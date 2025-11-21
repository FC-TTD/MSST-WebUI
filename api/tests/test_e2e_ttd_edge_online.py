"""
线上端到端测试 - ttd-edge 环境

测试 MSST 批量处理 API 在真实环境中的完整流程，包括：
- 同步模式和 SSE 流式模式
- Range 参数处理和文件范围验证
- 自定义 INPUT/OUTPUT 路径支持
- 项目级处理支持（/TTD/00_项目/xxx/input -> /TTD/00_项目/xxx/03_Stem/xxx）
- 真实文件输出验证
- CUDA 设备格式兼容性

环境变量配置：
    MSST_E2E_TTD_EDGE_ENABLE=1 启用测试
    MSST_E2E_BASE_URL=http://ttd-edge:8662/ API地址
    MSST_E2E_DEVICE_IDS=cuda:0 CUDA设备格式（重要：使用冒号）
    MSST_E2E_CONTAINER_MODE=1 容器模式

CUDA 设备格式注意：
    必须使用 "cuda:0" 格式，不是 "cuda0"
    这是 PyTorch 的标准设备表示法

项目级处理：
    支持任意项目目录结构
    输入：/TTD/00_项目/xxx/input（剧集文件）
    输出：/TTD/00_项目/xxx/03_Stem/xxx（分离结果）
    每个任务独立目录：task_xxx/Vocals.wav + Instrumental.wav

Range 功能：
    支持音频和视频文件处理
    音频格式 (.wav, .mp3, .flac, .m4a, .aac, .ogg)
    视频格式 (.mp4, .avi, .mkv, .mov, .wmv, .flv, .webm)
    MSST通过librosa+ffmpeg自动提取视频音频进行分离
    排除NAS系统文件（以@或.开头的文件，如.DS_Store, @eaDir）
    按文件名排序后索引，忽略非媒体文件
    支持范围验证和越界检查
    注意：视频文件必须包含音频轨道

文件输出验证：
    创建独立时间戳输出目录
    验证 API 返回与实际文件系统一致
    检查 Vocals/Instrumental 输出文件生成
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from api.client import MSSTApiClient


def _get_env_bool(name: str, default: bool = False) -> bool:
    """读取布尔环境变量（1/true/yes/on 为真，其余为假）。"""
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _require_env(name: str) -> str:
    """获取必填环境变量，若不存在则 pytest.skip。"""
    val = os.getenv(name)
    if not val:
        pytest.skip(f"线上 e2e 测试需要环境变量 {name}，未设置则跳过")
    return val


@pytest.fixture(scope="module")
def e2e_client():
    """创建线上 e2e 专用的 MSSTApiClient。"""
    if not _get_env_bool("MSST_E2E_TTD_EDGE_ENABLE"):
        pytest.skip("未设置 MSST_E2E_TTD_EDGE_ENABLE=1，跳过线上 e2e 测试")
    base_url = os.getenv("MSST_E2E_BASE_URL", "http://ttd-edge:8662/").rstrip("/")
    return MSSTApiClient(base_url=base_url, timeout=None)


@pytest.fixture(scope="module")
def e2e_paths():
    """读取测试用输入/输出路径，并做基本校验。支持容器内和宿主机路径映射。"""
    input_path = _require_env("MSST_E2E_INPUT_PATH")
    output_dir = _require_env("MSST_E2E_OUTPUT_DIR")
    # 容器模式：路径以 /app 开头，不做本地校验
    container_mode = _get_env_bool("MSST_E2E_CONTAINER_MODE")
    if not container_mode:
        # 宿主机模式：必须绝对路径且输入路径存在
        if not os.path.isabs(input_path):
            pytest.skip(f"MSST_E2E_INPUT_PATH 必须是绝对路径: {input_path}")
        if not os.path.isabs(output_dir):
            pytest.skip(f"MSST_E2E_OUTPUT_DIR 必须是绝对路径: {output_dir}")
        if not os.path.exists(input_path):
            pytest.skip(f"输入路径不存在: {input_path}")
        # 输出目录若不存在则尝试创建（权限不足时跳过）
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            pytest.skip(f"无法创建输出目录 {output_dir}: {e}")
    return {"input_path": input_path, "output_dir": output_dir}


@pytest.fixture(scope="module")
def e2e_model_config():
    """读取模型相关配置，提供合理默认值。"""
    return {
        "model_type": os.getenv("MSST_E2E_MODEL_TYPE", "vocal_models"),
        "model_name": os.getenv("MSST_E2E_MODEL_NAME", "melband_roformer_instvox_duality_v2.ckpt"),
        "extract_instrumental": ["Vocals", "Instrumental"],
        "device_ids": os.getenv("MSST_E2E_DEVICE_IDS", "cuda:0").split(","),
        "use_tta": _get_env_bool("MSST_E2E_USE_TTA", False),
        "range_start": int(os.getenv("MSST_E2E_RANGE_START", "1") or "1"),
        "range_end": int(os.getenv("MSST_E2E_RANGE_END", "3") or "3"),
    }


def test_ttd_edge_sync_end_to_end_online(e2e_client, e2e_paths, e2e_model_config):
    """验证同步模式（mode=sync）在 ttd-edge 容器上的完整链路。"""
    input_path = e2e_paths["input_path"]
    output_dir = e2e_paths["output_dir"]

    # 记录测试前的输出目录状态（用于校验有新文件生成）
    before_files = set()
    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in files:
                before_files.add(os.path.relpath(os.path.join(root, f), output_dir))

    resp = e2e_client.create_batch_sync(
        input_path=input_path,
        output_dir=output_dir,
        model_type=e2e_model_config["model_type"],
        model_name=e2e_model_config["model_name"],
        extract_instrumental=e2e_model_config["extract_instrumental"],
        range_start=e2e_model_config["range_start"],
        range_end=e2e_model_config["range_end"],
        params={
            "device_ids": e2e_model_config["device_ids"],
            "use_tta": e2e_model_config["use_tta"],
        },
    )

    # 严格校验最终状态为 success
    assert resp.get("status") == "success", f"任务未成功: {resp}"
    task_id = resp.get("task_id")
    assert isinstance(task_id, str) and task_id

    # 再通过状态查询接口确认一次
    status = e2e_client.get_task_status(task_id)
    assert status.get("status") == "success", f"状态查询不一致: {status}"

    # 容器模式下跳过本地文件系统检查，仅依赖 API 返回状态
    container_mode = _get_env_bool("MSST_E2E_CONTAINER_MODE")
    if not container_mode:
        # 校验输出目录确实有新文件（允许一定延迟）
        deadline = time.time() + 30  # 最多等 30 秒
        while True:
            after_files = set()
            if os.path.exists(output_dir):
                for root, _, files in os.walk(output_dir):
                    for f in files:
                        after_files.add(os.path.relpath(os.path.join(root, f), output_dir))
            new_files = after_files - before_files
            if new_files:
                # 至少有一个新文件生成，认为成功
                break
            if time.time() > deadline:
                pytest.fail("等待输出文件生成超时，可能推理未完成或输出路径错误")
            time.sleep(1)


def test_ttd_edge_sse_end_to_end_online(e2e_client, e2e_paths, e2e_model_config):
    """验证 SSE 流式模式（mode=sse）在 ttd-edge 容器上的完整链路。"""
    input_path = e2e_paths["input_path"]
    output_dir = e2e_paths["output_dir"]

    # 记录测试前的输出目录状态
    before_files = set()
    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in files:
                before_files.add(os.path.relpath(os.path.join(root, f), output_dir))

    events = list(
        e2e_client.create_batch_sse(
            input_path=input_path,
            output_dir=output_dir,
            model_type=e2e_model_config["model_type"],
            model_name=e2e_model_config["model_name"],
            extract_instrumental=e2e_model_config["extract_instrumental"],
            range_start=e2e_model_config["range_start"],
            range_end=e2e_model_config["range_end"],
            params={
                "device_ids": e2e_model_config["device_ids"],
                "use_tta": e2e_model_config["use_tta"],
            },
        )
    )

    # 校验 SSE 事件序列
    start_events = [ev for ev in events if ev.event == "start"]
    completed_events = [ev for ev in events if ev.event == "completed"]
    error_events = [ev for ev in events if ev.event == "error"]
    assert start_events, "未收到 start 事件"
    assert not error_events, f"SSE 流程报错: {error_events}"
    assert completed_events, "未收到 completed 事件"
    # 取最后一个 completed 事件作为最终状态
    final = completed_events[-1]
    assert final.data.get("status") == "success", f"SSE 最终状态非 success: {final.data}"
    task_id = final.data.get("task_id")
    assert isinstance(task_id, str) and task_id

    # 再通过状态查询接口确认一次
    status = e2e_client.get_task_status(task_id)
    assert status.get("status") == "success", f"SSE 完成后状态查询不一致: {status}"

    # 容器模式下跳过本地文件系统检查，仅依赖 API 返回状态
    container_mode = _get_env_bool("MSST_E2E_CONTAINER_MODE")
    if not container_mode:
        # 校验输出目录有新文件（允许一定延迟）
        deadline = time.time() + 30
        while True:
            after_files = set()
            if os.path.exists(output_dir):
                for root, _, files in os.walk(output_dir):
                    for f in files:
                        after_files.add(os.path.relpath(os.path.join(root, f), output_dir))
            new_files = after_files - before_files
            if new_files:
                break
            if time.time() > deadline:
                pytest.fail("SSE 模式等待输出文件生成超时")
            time.sleep(1)


def test_ttd_edge_custom_paths_and_range_behavior(e2e_client, e2e_paths, e2e_model_config):
    """测试自定义 INPUT/OUTPUT 路径和 Range 行为，验证真实文件输出。"""
    input_path = e2e_paths["input_path"]
    # 使用时间戳创建独立输出目录，避免历史文件干扰
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{e2e_paths['output_dir']}/e2e_test_{timestamp}"
    
    # 测试不使用 Range（处理所有文件）
    # range_start = 1
    # range_end = 1
    
    resp = e2e_client.create_batch_sync(
        input_path=input_path,
        output_dir=output_dir,
        model_type=e2e_model_config["model_type"],
        model_name=e2e_model_config["model_name"],
        extract_instrumental=e2e_model_config["extract_instrumental"],
        params={
            "device_ids": e2e_model_config["device_ids"],
            "use_tta": False,  # 关闭 TTA 加快测试
        },
    )
    
    # 严格校验最终状态为 success
    assert resp.get("status") == "success", f"任务未成功: {resp}"
    task_id = resp.get("task_id")
    assert isinstance(task_id, str) and task_id
    
    # 获取任务结果，检查文件列表
    result = e2e_client.get_task_result(task_id)
    assert result.get("status") == "success", f"任务结果非 success: {result}"
    
    files = result.get("files", [])
    # 不使用Range，应该处理所有音频文件，至少有1个文件组
    assert len(files) >= 1, f"输出文件数量不足，期望至少1个，实际{len(files)}个"
    
    # 验证文件信息
    for file_info in files:
        assert file_info.get("status") == "success", f"文件状态非 success: {file_info}"
        output_files = file_info.get("output_files", [])
        assert len(output_files) >= 2, f"输出文件数量不足: {file_info}"  # 至少 Vocals + Instrumental
    
    # 容器模式下通过 API 检查输出目录文件
    container_mode = _get_env_bool("MSST_E2E_CONTAINER_MODE")
    if container_mode:
        # 验证文件名包含预期的分离结果
        for file_info in files:
            output_filenames = file_info.get("output_files", [])
            assert len(output_filenames) >= 2, "输出文件名列表不完整"
            
            # 验证包含 Vocals 和 Instrumental 文件
            has_vocals = any("Vocals" in fname for fname in output_filenames)
            has_instrumental = any("Instrumental" in fname for fname in output_filenames)
            assert has_vocals, f"输出文件缺少 Vocals: {output_filenames}"
            assert has_instrumental, f"输出文件缺少 Instrumental: {output_filenames}"


def test_ttd_edge_range_all_files(e2e_client, e2e_paths, e2e_model_config):
    """测试 Range 处理所有文件的行为。"""
    input_path = e2e_paths["input_path"]
    # 使用时间戳创建独立输出目录
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{e2e_paths['output_dir']}/e2e_all_test_{timestamp}"
    
    # 测试 Range=1-3（处理所有3个文件）
    resp = e2e_client.create_batch_sync(
        input_path=input_path,
        output_dir=output_dir,
        model_type=e2e_model_config["model_type"],
        model_name=e2e_model_config["model_name"],
        extract_instrumental=e2e_model_config["extract_instrumental"],
        range_start=1,
        range_end=3,
        params={
            "device_ids": e2e_model_config["device_ids"],
            "use_tta": False,
        },
    )
    
    assert resp.get("status") == "success", f"处理所有文件任务未成功: {resp}"
    task_id = resp.get("task_id")
    
    result = e2e_client.get_task_result(task_id)
    assert result.get("status") == "success", f"任务结果非 success: {result}"
    
    files = result.get("files", [])
    # 3个文件应该产生至少3个输出文件组
    assert len(files) >= 3, f"处理所有文件时输出数量不足，期望至少3个，实际{len(files)}个"
    
    # 验证每个文件组都有合理的输出文件数量
    for file_info in files:
        assert file_info.get("status") == "success", f"文件状态非 success: {file_info}"
        output_files = file_info.get("output_files", [])
        assert len(output_files) >= 2, f"输出文件数量不足: {file_info}"  # 至少 Vocals + Instrumental
