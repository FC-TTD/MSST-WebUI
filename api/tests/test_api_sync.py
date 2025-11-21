"""Basic tests for the synchronous MSST batch API.

这些测试尽量不依赖真实模型和音频文件：
- storage 层使用临时 SQLite 文件
- API 层主要验证输入路径校验与任务状态持久化逻辑

运行方式示例：

    pytest api/test_api_sync.py

需要依赖 fastapi[testclient] / pytest 等测试依赖。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import register_routes
from api.storage.sqlite import SQLiteStorage


def create_test_app() -> FastAPI:
    """构建一个仅包含 API 路由的轻量 FastAPI 应用，用于单元测试。

    测试不需要完整拉起 Gradio WebUI 和前端，仅关注 /api/v1/... 行为。
    """

    app = FastAPI()
    register_routes(app)
    return app


def test_sqlite_storage_create_and_get_status(tmp_path: Path) -> None:
    """SQLiteStorage 能够创建任务并返回状态。"""

    db_path = tmp_path / "tasks.db"
    storage = SQLiteStorage(str(db_path))

    task_id = storage.create_task(status="running", message="test")

    status = storage.get_task_status(task_id)
    assert status is not None
    assert status.task_id == task_id
    assert status.status == "running"
    assert status.error is None


def test_api_msst_batch_invalid_input_path() -> None:
    """当输入路径不存在时，应返回 status=failed。"""

    app = create_test_app()
    client = TestClient(app)

    # 使用明显不存在的路径
    payload = {
        "input_path": "/path/does/not/exist",
        "output_dir": "results_test/",
        "model_name": "dummy-model",  # 名称本身不会被真正使用，因为会在路径校验前失败
        "params": {},
    }

    resp = client.post("/api/v1/tasks/msst-batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # 同步接口：在路径校验失败时直接标记任务失败
    assert data["status"] == "failed"
    assert isinstance(data["task_id"], str) and data["task_id"]


def test_api_msst_batch_uses_absolute_paths(tmp_path: Path, monkeypatch) -> None:
    """验证 API 会将输入/输出路径归一化为绝对路径传入服务层（间接测试）。

    这里通过创建一个实际存在的空目录，确保不会触发“路径不存在”的错误，
    但又避免真正跑 MSST 模型：通过 monkeypatch run_folder_batch_inference。
    """

    from api import services

    calls: list[dict[str, str]] = []

    def fake_run_folder_batch_inference(
        selected_model,
        input_folder,
        store_dir,
        extract_instrumental,
        gpu_id,
        output_format,
        force_cpu,
        use_tta,
    ) -> str:  # pragma: no cover - 行为很简单
        calls.append({"input_folder": input_folder, "store_dir": store_dir})
        return "ok"

    monkeypatch.setattr(services, "run_folder_batch_inference", fake_run_folder_batch_inference)

    app = create_test_app()
    client = TestClient(app)

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    payload = {
        "input_path": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": "dummy-model",
        "params": {},
    }

    resp = client.post("/api/v1/tasks/msst-batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"

    assert len(calls) == 1
    # 被传入 fake_run_folder_batch_inference 的路径应为绝对路径
    assert os.path.isabs(calls[0]["input_folder"])
    assert os.path.isabs(calls[0]["store_dir"])


def test_api_msst_batch_range_start_end(tmp_path: Path, monkeypatch) -> None:
    """验证 range_start/range_end 在目录输入时按 1-based 闭区间生效。

    构造一个包含多集子目录的输入目录，只应处理指定范围内的子项。
    """

    from api import services

    calls: list[dict[str, object]] = []

    def fake_run_folder_batch_inference(
        selected_model,
        input_folder,
        store_dir,
        extract_instrumental,
        gpu_id,
        output_format,
        force_cpu,
        use_tta,
    ) -> str:  # pragma: no cover - 行为很简单
        entries = sorted(os.listdir(input_folder))
        calls.append({"input_folder": input_folder, "entries": entries})
        return "ok"

    monkeypatch.setattr(services, "run_folder_batch_inference", fake_run_folder_batch_inference)

    app = create_test_app()
    client = TestClient(app)

    # 构造三集子目录：ep01, ep02, ep03
    input_root = tmp_path / "episodes"
    input_root.mkdir()
    for name in ["ep01", "ep02", "ep03"]:
        (input_root / name).mkdir()

    output_dir = tmp_path / "out"

    payload = {
        "input_path": str(input_root),
        "output_dir": str(output_dir),
        "model_name": "dummy-model",
        "params": {"range_start": 2, "range_end": 3},  # 仅处理 ep02, ep03
    }

    resp = client.post("/api/v1/tasks/msst-batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"

    assert len(calls) == 1
    entries = calls[0]["entries"]
    assert entries == ["ep02", "ep03"]


def test_api_msst_batch_range_start_end_files(tmp_path: Path, monkeypatch) -> None:
    """验证 range_start/range_end 对纯文件目录同样按 1-based 闭区间生效。"""

    from api import services

    calls: list[dict[str, object]] = []

    def fake_run_folder_batch_inference(
        selected_model,
        input_folder,
        store_dir,
        extract_instrumental,
        gpu_id,
        output_format,
        force_cpu,
        use_tta,
    ) -> str:  # pragma: no cover - 行为很简单
        entries = sorted(os.listdir(input_folder))
        calls.append({"input_folder": input_folder, "entries": entries})
        return "ok"

    monkeypatch.setattr(services, "run_folder_batch_inference", fake_run_folder_batch_inference)

    app = create_test_app()
    client = TestClient(app)

    # 构造三个文件：a.wav, b.wav, c.wav
    input_root = tmp_path / "files"
    input_root.mkdir()
    for name in ["a.wav", "b.wav", "c.wav"]:
        (input_root / name).write_text("test", encoding="utf-8")

    output_dir = tmp_path / "out_files"

    payload = {
        "input_path": str(input_root),
        "output_dir": str(output_dir),
        "model_name": "dummy-model",
        "params": {"range_start": 1, "range_end": 2},  # 仅处理 a.wav, b.wav
    }

    resp = client.post("/api/v1/tasks/msst-batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"

    assert len(calls) == 1
    entries = calls[0]["entries"]
    assert entries == ["a.wav", "b.wav"]
