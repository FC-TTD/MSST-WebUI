from __future__ import annotations

import json
import multiprocessing
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

from api.models import FileResult, ModelInfo, TaskCreateRequest, TaskResultResponse
from api.storage import get_storage
from utils.constant import MODELS_INFO, WEBUI_CONFIG
from webui.msst import run_folder_batch_inference, run_inference
from webui.utils import get_msst_model, i18n, load_configs, logger


def _validate_host_path(path: str) -> str | None:
    if path.startswith("/Volume/") or path.startswith("/Volumes/"):
        return i18n("非法路径（疑似 macOS Volume 路径）: ") + path
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return i18n("非法路径（疑似 Windows 盘符路径）: ") + path
    return None


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return default


def _finalize_task_output(task_output_dir: str, output_dir: str) -> None:
    if not task_output_dir or not os.path.exists(task_output_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    for root, dirnames, filenames in os.walk(task_output_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(("@", "."))]
        filenames = [f for f in filenames if not f.startswith(("@", "."))]
        for filename in filenames:
            src = os.path.join(root, filename)
            rel_path = os.path.relpath(src, task_output_dir)
            dst = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.replace(src, dst)
            except Exception:
                try:
                    shutil.copy2(src, dst)
                    os.remove(src)
                except Exception:
                    logger.exception(f"Failed to move output file: {src} -> {dst}")

    shutil.rmtree(task_output_dir, ignore_errors=True)


def run_msst_batch_sync(req: TaskCreateRequest) -> TaskResultResponse:
    """同步（blocking）执行一次批量 MSST 分离，并记录任务结果。

    说明：
    - 当前实现直接复用 webui.msst.run_folder_batch_inference，保持与 Gradio 相同的推理链路；
    - API 允许调用方自定义 input_path/output_dir，但要求服务端能访问这些路径；
    - 为简化首版实现，暂不返回过程进度，仅在完成后一次性返回结果。
    """

    storage = get_storage()

    # 先创建任务记录，初始状态为 running
    task_id = storage.create_task(status="running", message=None)

    input_path = os.path.abspath(req.input_path)
    output_dir = os.path.abspath(req.output_dir)

    path_err = _validate_host_path(req.input_path) or _validate_host_path(req.output_dir)
    if path_err:
        logger.error(path_err)
        storage.update_task(task_id, status="failed", error=path_err)
        return TaskResultResponse(task_id=task_id, status="failed", files=[])

    # 简单校验路径
    if not os.path.exists(input_path):
        msg = i18n("输入路径不存在: ") + input_path
        logger.error(msg)
        storage.update_task(task_id, status="failed", error=msg)
        return TaskResultResponse(task_id=task_id, status="failed", files=[])

    os.makedirs(output_dir, exist_ok=True)

    range_start_raw = req.params.get("range_start")
    range_end_raw = req.params.get("range_end")
    input_path_for_infer = input_path
    cleanup_dir: str | None = None
    if os.path.isdir(input_path) and (range_start_raw is not None or range_end_raw is not None):
        try:
            range_start = int(range_start_raw) if range_start_raw is not None else 1
            range_end = int(range_end_raw) if range_end_raw is not None else 0
        except Exception:
            msg = i18n("range_start/range_end 必须为整数")
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            return TaskResultResponse(task_id=task_id, status="failed", files=[])

        if range_start < 1:
            range_start = 1

        names = sorted(os.listdir(input_path))
        # 过滤出音频和视频文件，避免非媒体文件和NAS系统文件干扰Range索引
        media_names = []
        for name in names:
            # 跳过以@或.开头的文件（NAS系统文件）
            if name.startswith(('@', '.')):
                continue
            file_path = os.path.join(input_path, name)
            if os.path.isfile(file_path):
                # 检查文件扩展名是否为音频或视频格式
                # librosa支持通过ffmpeg提取视频中的音频
                if name.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg',  # 音频格式
                                           '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')):  # 视频格式
                    media_names.append(name)
        names = media_names
        total = len(names)
        if total == 0:
            msg = i18n("输入目录中没有找到媒体文件: ") + input_path
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            return TaskResultResponse(task_id=task_id, status="failed", files=[])

        if range_end_raw is None:
            range_end = total

        if range_end < range_start or range_start > total:
            msg = i18n("range_start/range_end 范围无效")
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            return TaskResultResponse(task_id=task_id, status="failed", files=[])

        if range_end > total:
            range_end = total

        if not (range_start == 1 and range_end == total):
            import tempfile
            tmp_base = os.path.join(tempfile.gettempdir(), f".msst_api_range_{task_id}")
            os.makedirs(tmp_base, exist_ok=True)
            for idx in range(range_start - 1, range_end):
                name = names[idx]
                src = os.path.join(input_path, name)
                dst = os.path.join(tmp_base, name)
                try:
                    if os.path.isdir(src):
                        os.symlink(src, dst, target_is_directory=True)
                    else:
                        os.symlink(src, dst)
                except Exception:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            input_path_for_infer = tmp_base
            cleanup_dir = tmp_base

    # 从顶级字段和 params 中获取可选参数，提供合理默认值
    # 输出音轨优先使用顶级字段，兼容旧版 params["instrumental"] 写法
    extract_instrumental: List[str] = (req.extract_instrumental or req.params.get("instrumental") or [])

    # 设备选择：默认不使用 CPU，缺省时自动使用 GPU0
    gpu_id = req.params.get("device_ids")
    force_cpu = bool(req.params.get("force_cpu", False))
    if not force_cpu and (not gpu_id):
        gpu_id = [0]

    output_format = req.params.get("output_format") or "wav"
    # 复用 WebUI 已保存的 use_tta 设置，忽略 API 传参
    try:
        _webui_cfg = load_configs(WEBUI_CONFIG)
        use_tta = bool(_webui_cfg.get("inference", {}).get("use_tta", True))
    except Exception:
        use_tta = True

    task_output_dir = os.path.join(output_dir, f"task_{task_id}")

    try:
        os.makedirs(task_output_dir, exist_ok=True)

        message, _ = run_folder_batch_inference(
            req.model_name,
            input_path_for_infer,
            task_output_dir,
            extract_instrumental,
            gpu_id,
            output_format,
            force_cpu,
            use_tta,
        )

        logger.info(message)
        storage.update_task(task_id, status="success", progress=1.0)

        # 收集实际生成的输出文件
        files: List[FileResult] = []
        try:
            # 扫描当前任务的输出目录，收集新生成的文件
            if os.path.exists(task_output_dir):
                output_files = []
                for root, dirnames, filenames in os.walk(task_output_dir):
                    dirnames[:] = [d for d in dirnames if not d.startswith(("@", "."))]
                    filenames = [f for f in filenames if not f.startswith(("@", "."))]
                    for filename in filenames:
                        if filename.endswith(('.wav', '.mp3')):
                            file_path = os.path.join(root, filename)
                            output_files.append(os.path.basename(file_path))
                
                # 根据文件名推断输入文件并分组
                input_file_groups = {}
                for output_file in output_files:
                    # 尝试从输出文件名推断输入文件
                    # 例如: test1_Vocals.wav -> test1
                    input_name = None
                    if "_Vocals" in output_file:
                        input_name = output_file.split("_Vocals")[0]
                    elif "_Instrumental" in output_file:
                        input_name = output_file.split("_Instrumental")[0]
                    else:
                        # 如果无法推断，使用通用名称
                        input_name = f"file_{len(input_file_groups) + 1}"
                    
                    if input_name not in input_file_groups:
                        input_file_groups[input_name] = []
                    input_file_groups[input_name].append(output_file)
                
                # 为每个输入文件创建一个 FileResult
                for input_name, files_list in input_file_groups.items():
                    files.append(FileResult(
                        input_file=input_name,
                        output_files=files_list,
                        status="success"
                    ))
                    
            logger.info(f"Collected {len(files)} file results from {task_output_dir}")
        except Exception as e:
            logger.warning(f"Failed to collect output files: {e}")
        
        storage.save_results(task_id, files)
        return TaskResultResponse(task_id=task_id, status="success", files=files)
    except Exception as e:
        err_msg = f"MSST batch inference failed: {e}"
        logger.exception(err_msg)
        storage.update_task(task_id, status="failed", error=err_msg)
        return TaskResultResponse(task_id=task_id, status="failed", files=[])
    finally:
        _finalize_task_output(task_output_dir, output_dir)
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def list_models(model_class: str | None = None) -> List[ModelInfo]:
    """列出可用模型。

    - 基于 MODELS_INFO（data/models_info.json）
    - 可选按 model_class 过滤，例如 "VR_Models"、"multi_stem_models" 等。
    """

    from webui.utils import load_configs as _load_configs  # 避免和上面 import 冲突

    try:
        config = _load_configs(MODELS_INFO)
    except Exception as e:  # pragma: no cover - 防御性分支
        logger.exception("Failed to load models info")
        raise

    results: List[ModelInfo] = []
    for key, info in config.items():
        mc = info.get("model_class")
        if model_class and mc != model_class:
            continue
        base_fields = {
            "id": key,
            "model_name": info.get("model_name", key),
            "model_class": mc or "",
            "model_size": info.get("model_size"),
            "sha256": info.get("sha256"),
            "is_installed": info.get("is_installed"),
            "target_position": info.get("target_position"),
            "link": info.get("link"),
        }
        extra = {k: v for k, v in info.items() if k not in base_fields}
        results.append(ModelInfo(extra=extra, **base_fields))
    return results


def _build_store_dict(store_dir: str, extract_instrumental: List[str], config_path: str) -> Dict[str, str]:
    """根据选中的音轨构建 store_dict，逻辑参考 webui.msst.start_inference。

    如果未选择任何有效音轨，则默认导出模型支持的全部音轨到同一目录。
    """

    from webui.utils import load_configs as _load_configs  # 避免循环引用

    if isinstance(store_dir, str):
        store_dict: Dict[str, str] = {}
        model_config = _load_configs(config_path)
        for inst in extract_instrumental:
            if inst in model_config.training.get("instruments"):
                store_dict[inst] = store_dir
        if not store_dict:
            store_dict = {k: store_dir for k in model_config.training.get("instruments")}
        return store_dict
    return store_dir  # 兼容未来扩展


def _sse_event(event: str, data: Dict) -> str:
    """构造标准 SSE 事件字符串。"""

    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def run_msst_batch_sse(req: TaskCreateRequest):
    """以 SSE 事件流的方式执行一次批量 MSST 分离。

    - 使用 multiprocessing + webui.msst.run_inference 复用现有推理逻辑；
    - 通过 Manager().dict() 回调读取进度，更新 SQLite，并推送 SSE 事件；
    - 仍然创建 Task 记录，便于后续通过 /tasks/{task_id} 查询。
    """

    storage = get_storage()
    task_id = storage.create_task(status="running", message=None)

    input_path = os.path.abspath(req.input_path)
    output_dir = os.path.abspath(req.output_dir)

    path_err = _validate_host_path(req.input_path) or _validate_host_path(req.output_dir)
    if path_err:
        logger.error(path_err)
        storage.update_task(task_id, status="failed", error=path_err)
        yield _sse_event("error", {"task_id": task_id, "message": path_err})
        return

    if not os.path.exists(input_path):
        msg = i18n("输入路径不存在: ") + input_path
        logger.error(msg)
        storage.update_task(task_id, status="failed", error=msg)
        yield _sse_event("error", {"task_id": task_id, "message": msg})
        return

    os.makedirs(output_dir, exist_ok=True)

    task_output_dir = os.path.join(output_dir, f"task_{task_id}")

    range_start_raw = req.params.get("range_start")
    range_end_raw = req.params.get("range_end")
    input_path_for_infer = input_path
    cleanup_dir: str | None = None
    if os.path.isdir(input_path) and (range_start_raw is not None or range_end_raw is not None):
        try:
            range_start = int(range_start_raw) if range_start_raw is not None else 1
            range_end = int(range_end_raw) if range_end_raw is not None else 0
        except Exception:
            msg = i18n("range_start/range_end 必须为整数")
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            yield _sse_event("error", {"task_id": task_id, "message": msg})
            return

        if range_start < 1:
            range_start = 1

        names = sorted(os.listdir(input_path))
        # 过滤出音频和视频文件，避免非媒体文件和NAS系统文件干扰Range索引
        media_names = []
        for name in names:
            # 跳过以@或.开头的文件（NAS系统文件）
            if name.startswith(('@', '.')):
                continue
            file_path = os.path.join(input_path, name)
            if os.path.isfile(file_path):
                # 检查文件扩展名是否为音频或视频格式
                # librosa支持通过ffmpeg提取视频中的音频
                if name.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg',  # 音频格式
                                           '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')):  # 视频格式
                    media_names.append(name)
        names = media_names
        total = len(names)
        if total == 0:
            msg = i18n("输入目录中没有找到音频文件: ") + input_path
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            yield _sse_event("error", {"task_id": task_id, "message": msg})
            return

        if range_end_raw is None:
            range_end = total

        if range_end < range_start or range_start > total:
            msg = i18n("range_start/range_end 范围无效")
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            yield _sse_event("error", {"task_id": task_id, "message": msg})
            return

        if range_end > total:
            range_end = total

        if not (range_start == 1 and range_end == total):
            import tempfile
            tmp_base = os.path.join(tempfile.gettempdir(), f".msst_api_range_{task_id}")
            os.makedirs(tmp_base, exist_ok=True)
            for idx in range(range_start - 1, range_end):
                name = names[idx]
                src = os.path.join(input_path, name)
                dst = os.path.join(tmp_base, name)
                try:
                    if os.path.isdir(src):
                        os.symlink(src, dst, target_is_directory=True)
                    else:
                        os.symlink(src, dst)
                except Exception:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            input_path_for_infer = tmp_base
            cleanup_dir = tmp_base

    # 输出音轨优先使用顶级字段，兼容旧版 params["instrumental"]
    extract_instrumental: List[str] = (req.extract_instrumental or req.params.get("instrumental") or [])
    gpu_id = req.params.get("device_ids")
    output_format = req.params.get("output_format") or "wav"
    force_cpu = bool(req.params.get("force_cpu", False))
    # 复用 WebUI 已保存的 use_tta 设置，忽略 API 传参
    use_tta = True

    try:
        # 构建推理所需参数（参考 webui.msst.start_inference）
        if not req.model_name:
            msg = i18n("请选择模型")
            storage.update_task(task_id, status="failed", error=msg)
            yield _sse_event("error", {"task_id": task_id, "message": msg})
            return

        gpu_ids: List[int] = []
        if not force_cpu:
            # 默认不使用 CPU：缺省时自动选择 GPU0
            if not gpu_id:
                gpu_id = [0]
            try:
                for gpu in gpu_id:
                    gpu_ids.append(int(str(gpu).split(":", 1)[0].replace("cuda", "")))
            except Exception:
                gpu_ids = [0]
        else:
            gpu_ids = [0]

        gpu_ids = list(set(gpu_ids))
        device = "auto" if not force_cpu else "cpu"

        model_path, config_path, model_type, _ = get_msst_model(req.model_name)
        webui_config = load_configs(WEBUI_CONFIG)
        debug = webui_config["settings"].get("debug", False)
        wav_bit_depth = webui_config["settings"].get("wav_bit_depth", "FLOAT")
        flac_bit_depth = webui_config["settings"].get("flac_bit_depth", "PCM_24")
        mp3_bit_rate = webui_config["settings"].get("mp3_bit_rate", "320k")

        os.makedirs(task_output_dir, exist_ok=True)
        
        store_dict = _build_store_dict(task_output_dir, extract_instrumental, config_path)

        start_time = time.time()
        logger.info("Starting MSST inference process (SSE mode)...")

        yield _sse_event("start", {"task_id": task_id, "status": "running"})

        with multiprocessing.Manager() as manager:
            callback = manager.dict()  # type: ignore[var-annotated]
            callback["info"] = {"index": -1, "total": -1, "name": ""}
            callback["progress"] = 0.0
            callback["flag"] = (0, None)

            proc = multiprocessing.Process(
                target=run_inference,
                args=(
                    model_type,
                    config_path,
                    model_path,
                    device,
                    gpu_ids,
                    output_format,
                    use_tta,
                    store_dict,
                    debug,
                    wav_bit_depth,
                    flac_bit_depth,
                    mp3_bit_rate,
                    input_path_for_infer,
                    callback,
                    "recursive" if req.params.get("batch_mode") == "recursive" else "folder",
                ),
                name="msst_inference_api",
            )

            proc.start()
            logger.debug(f"Inference process (SSE) started, PID: {proc.pid}")

            while proc.is_alive():
                flag = callback["flag"]
                if flag[0]:
                    break
                info = callback["info"]
                progress = float(callback.get("progress", 0.0))
                if info["index"] != -1:
                    processed = info["index"]
                    total = info["total"]
                    current_file = info["name"]
                    storage.update_task(
                        task_id,
                        progress=progress,
                        total_files=total,
                        processed_files=processed,
                    )
                    yield _sse_event(
                        "progress",
                        {
                            "task_id": task_id,
                            "status": "running",
                            "progress": progress,
                            "processed_files": processed,
                            "total_files": total,
                            "current_file": current_file,
                        },
                    )
                time.sleep(0.5)

            proc.join()
            flag = callback["flag"]

        if flag[0] == 1:
            duration = round(time.time() - start_time, 2)
            storage.update_task(task_id, status="success", progress=1.0)
            
            # 收集实际生成的输出文件
            files: List[FileResult] = []
            try:
                # 扫描当前任务的输出目录，收集新生成的文件
                if os.path.exists(task_output_dir):
                    output_files = []
                    for root, dirnames, filenames in os.walk(task_output_dir):
                        dirnames[:] = [d for d in dirnames if not d.startswith(("@", "."))]
                        filenames = [f for f in filenames if not f.startswith(("@", "."))]
                        for filename in filenames:
                            if filename.endswith(('.wav', '.mp3')):
                                file_path = os.path.join(root, filename)
                                output_files.append(os.path.basename(file_path))
                    
                    # 根据文件名推断输入文件并分组
                    input_file_groups = {}
                    for output_file in output_files:
                        # 尝试从输出文件名推断输入文件
                        # 例如: test1_Vocals.wav -> test1
                        input_name = None
                        if "_Vocals" in output_file:
                            input_name = output_file.split("_Vocals")[0]
                        elif "_Instrumental" in output_file:
                            input_name = output_file.split("_Instrumental")[0]
                        else:
                            # 如果无法推断，使用通用名称
                            input_name = f"file_{len(input_file_groups) + 1}"
                        
                        if input_name not in input_file_groups:
                            input_file_groups[input_name] = []
                        input_file_groups[input_name].append(output_file)
                    
                    # 为每个输入文件创建一个 FileResult
                    for input_name, files_list in input_file_groups.items():
                        files.append(FileResult(
                            input_file=input_name,
                            output_files=files_list,
                            status="success"
                        ))
                        
                logger.info(f"Collected {len(files)} file results from {task_output_dir}")
                storage.save_results(task_id, files)
            except Exception as e:
                logger.warning(f"Failed to collect output files: {e}")

            _finalize_task_output(task_output_dir, output_dir)
            
            files_payload = [
                (f.model_dump() if hasattr(f, "model_dump") else f.dict())  # type: ignore[attr-defined]
                for f in files
            ]
            yield _sse_event(
                "completed",
                {"task_id": task_id, "status": "success", "duration": duration, "files": files_payload},
            )
        elif flag[0] == -1:
            err_msg = str(flag[1])
            storage.update_task(task_id, status="failed", error=err_msg)
            _finalize_task_output(task_output_dir, output_dir)
            yield _sse_event(
                "error",
                {"task_id": task_id, "status": "failed", "message": err_msg},
            )
        else:
            msg = "MSST inference process exited unexpectedly"
            storage.update_task(task_id, status="failed", error=msg)
            _finalize_task_output(task_output_dir, output_dir)
            yield _sse_event(
                "error",
                {"task_id": task_id, "status": "failed", "message": msg},
            )
    except Exception as e:  # pragma: no cover - 防御性分支
        err_msg = f"MSST batch inference (SSE) failed: {e}"
        logger.exception(err_msg)
        storage.update_task(task_id, status="failed", error=err_msg)
        _finalize_task_output(task_output_dir, output_dir)
        yield _sse_event(
            "error",
            {"task_id": task_id, "status": "failed", "message": err_msg},
        )
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
