import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import services
from api.app import register_routes
from api.models import TaskCreateRequest, TaskCreateResponse


def _app() -> FastAPI:
    app = FastAPI()
    register_routes(app)
    return app


def _request() -> TaskCreateRequest:
    return TaskCreateRequest(
        input_path="/TTD/input.wav",
        output_dir="/TTD/output",
        model_type="vocal_models",
        model_name="melband_roformer_instvox_duality_v2.ckpt",
    )


def test_async_mode_returns_task_id_immediately(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.app.start_msst_batch_async",
        lambda _req: TaskCreateResponse(task_id="async-1", status="queued"),
    )
    response = TestClient(_app()).post(
        "/api/v1/tasks/msst-batch?mode=async",
        json=_request().model_dump(),
    )
    assert response.status_code == 200
    assert response.json() == {"task_id": "async-1", "status": "queued", "message": None}


def test_async_mode_drains_existing_sse_lifecycle_with_stable_task_id(monkeypatch) -> None:
    drained = threading.Event()
    observed: list[str] = []

    class Storage:
        def create_task(self, **_kwargs) -> str:
            return "async-2"

        def update_task(self, *_args, **_kwargs) -> None:
            raise AssertionError("successful lifecycle must not be marked failed")

    def lifecycle(_req, existing_task_id=None):
        observed.append(existing_task_id)
        yield "queued"
        drained.set()

    monkeypatch.setattr(services, "get_storage", lambda: Storage())
    monkeypatch.setattr(services, "run_msst_batch_sse", lifecycle)

    response = services.start_msst_batch_async(_request())

    assert response == TaskCreateResponse(task_id="async-2", status="queued", message=None)
    assert drained.wait(timeout=1)
    assert observed == ["async-2"]


def test_async_mode_marks_task_failed_when_lifecycle_raises(monkeypatch) -> None:
    failed = threading.Event()
    updates: list[tuple[str, dict[str, str]]] = []

    class Storage:
        def create_task(self, **_kwargs) -> str:
            return "async-3"

        def update_task(self, task_id: str, **fields: str) -> None:
            updates.append((task_id, fields))
            failed.set()

    def lifecycle(_req, existing_task_id=None):
        assert existing_task_id == "async-3"
        raise RuntimeError("background failure")
        yield  # pragma: no cover

    monkeypatch.setattr(services, "get_storage", lambda: Storage())
    monkeypatch.setattr(services, "run_msst_batch_sse", lifecycle)

    services.start_msst_batch_async(_request())

    assert failed.wait(timeout=1)
    assert updates == [("async-3", {"status": "failed", "error": "background failure"})]
