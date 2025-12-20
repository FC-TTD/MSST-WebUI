from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.config as config_module
from api.app import register_routes
from api.storage import get_storage
from api.task_runtime import TaskRuntime, register_task_runtime, unregister_task_runtime


def create_test_app() -> FastAPI:
    app = FastAPI()
    register_routes(app)
    return app


def test_cancel_sse_task_returns_partial_results(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "tasks_cancel.db"
    monkeypatch.setattr(config_module, "DEFAULT_DB_PATH", str(db_path))

    output_dir = tmp_path / "out"
    task_id = get_storage().create_task(status="running", message=None)

    task_output_dir = output_dir / f"task_{task_id}"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create partial outputs (simulate already processed files)
    (task_output_dir / "ep01_Vocals.wav").write_bytes(b"RIFF")
    (task_output_dir / "ep01_Instrumental.wav").write_bytes(b"RIFF")

    runtime = TaskRuntime(
        task_id=task_id,
        pid=999999,
        task_output_dir=str(task_output_dir),
        output_dir=str(output_dir),
    )
    register_task_runtime(runtime)

    killed = []

    def fake_kill(pid: int, sig: int):  # type: ignore[override]
        killed.append((pid, sig))
        return None

    monkeypatch.setattr(os, "kill", fake_kill)

    app = create_test_app()
    client = TestClient(app)

    resp = client.post(f"/api/v1/tasks/{task_id}/cancel")
    assert resp.status_code == 200
    data = resp.json()

    assert data["task_id"] == task_id
    assert data["status"] == "canceled"

    # Should return the partial outputs
    assert any(fr["input_file"] == "ep01" for fr in data["files"])
    ep01 = next(fr for fr in data["files"] if fr["input_file"] == "ep01")
    assert "ep01_Vocals.wav" in ep01["output_files"]
    assert "ep01_Instrumental.wav" in ep01["output_files"]

    # output files should be moved to output_dir
    assert (output_dir / "ep01_Vocals.wav").exists()
    assert (output_dir / "ep01_Instrumental.wav").exists()
    # task-specific temp dir should be removed
    assert not task_output_dir.exists()

    # result endpoint should return stored results
    resp2 = client.get(f"/api/v1/tasks/{task_id}/result")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "canceled"
    assert any(fr["input_file"] == "ep01" for fr in data2["files"])

    # cleanup registry in case test fails early
    unregister_task_runtime(task_id)
