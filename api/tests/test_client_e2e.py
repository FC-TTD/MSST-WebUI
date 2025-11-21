from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import register_routes
from api.client import MSSTApiClient
from api.models import TaskCreateRequest, TaskResultResponse


def create_test_app() -> FastAPI:
    app = FastAPI()
    register_routes(app)
    return app


def test_client_sync_end_to_end(monkeypatch, tmp_path: Path) -> None:
    """端到端验证 MSSTApiClient.create_batch_sync -> HTTP -> API -> services。

    通过 monkeypatch services.run_msst_batch_sync，检查顶级字段和 params 是否正确传入。
    """

    import api.app as app_module

    captured: List[TaskCreateRequest] = []

    def fake_run_msst_batch_sync(req: TaskCreateRequest) -> TaskResultResponse:  # type: ignore[override]
        captured.append(req)
        # 直接返回一个成功结果，避免真实推理
        return TaskResultResponse(task_id="task-sync-1", status="success", files=[])

    # router 在 api.app 中直接引用 run_msst_batch_sync，这里需要 patch api.app 上的符号
    monkeypatch.setattr(app_module, "run_msst_batch_sync", fake_run_msst_batch_sync)

    app = create_test_app()
    test_client = TestClient(app)

    api_client = MSSTApiClient(base_url=str(test_client.base_url), session=test_client)  # type: ignore[arg-type]

    input_dir = tmp_path / "input_sync"
    input_dir.mkdir()
    (input_dir / "a.wav").write_text("test", encoding="utf-8")

    output_dir = tmp_path / "out_sync"

    resp = api_client.create_batch_sync(
        input_path=str(input_dir),
        output_dir=str(output_dir),
        model_type="vocal_models",
        model_name="melband_roformer_instvox_duality_v2.ckpt",
        extract_instrumental=["Vocals", "Instrumental"],
        range_start=1,
        range_end=1,
        params={"device_ids": ["cuda0"], "use_tta": True},
    )

    assert resp["status"] == "success"
    assert resp["task_id"] == "task-sync-1"

    assert len(captured) == 1
    req = captured[0]
    assert req.input_path == str(input_dir)
    assert req.output_dir == str(output_dir)
    assert req.model_type == "vocal_models"
    assert req.model_name == "melband_roformer_instvox_duality_v2.ckpt"
    assert req.extract_instrumental == ["Vocals", "Instrumental"]
    assert req.params.get("range_start") == 1
    assert req.params.get("range_end") == 1
    assert req.params.get("device_ids") == ["cuda0"]
    assert req.params.get("use_tta") is True


def test_client_sse_end_to_end(tmp_path: Path) -> None:
    """端到端验证 MSSTApiClient.create_batch_sse 的 payload 构造与 SSE 解析。

    这里不依赖 FastAPI TestClient 的 stream 能力，而是通过 DummySession
    模拟一次 HTTP POST + text/event-stream 响应：
    - 检查 client 发送的 JSON payload 是否包含顶级字段和 params；
    - 检查 SSE 文本能被正确解析为 SSEEvent 序列。
    """

    captured_kwargs: list[dict] = []

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200

        # create_batch_sse 只会在 status_code 非 2xx 时调用 json/text，这里无需实现

        def iter_lines(self, decode_unicode: bool = False):  # type: ignore[override]
            # 最小合法 SSE 流：
            lines = [
                "event: completed\n",
                'data: {"task_id": "task-sse-1", "status": "success"}\n',
                "\n",
            ]
            for line in lines:
                if decode_unicode:
                    yield line
                else:
                    yield line.encode("utf-8")

    class DummySession:
        def post(self, url: str, **kwargs):  # type: ignore[override]
            captured_kwargs.append({"url": url, **kwargs})
            return DummyResponse()

    api_client = MSSTApiClient(base_url="http://testserver")
    api_client._session = DummySession()  # type: ignore[assignment]

    input_dir = tmp_path / "input_sse"
    input_dir.mkdir()
    (input_dir / "b.wav").write_text("test", encoding="utf-8")

    output_dir = tmp_path / "out_sse"

    events = list(
        api_client.create_batch_sse(
            input_path=str(input_dir),
            output_dir=str(output_dir),
            model_type="vocal_models",
            model_name="mel_band_roformer_instrumental_becruily.ckpt",
            extract_instrumental=["Vocals"],
            range_start=1,
            range_end=1,
            params={"device_ids": ["cuda1"]},
        )
    )

    # 校验 SSE 解析结果
    assert any(ev.event == "completed" and ev.data.get("status") == "success" for ev in events)

    # 校验 payload 构造
    assert len(captured_kwargs) == 1
    call = captured_kwargs[0]
    assert call["url"].endswith("/api/v1/tasks/msst-batch")
    assert call["params"] == {"mode": "sse"}

    payload = call["json"]
    assert payload["input_path"] == str(input_dir)
    assert payload["output_dir"] == str(output_dir)
    assert payload["model_name"] == "mel_band_roformer_instrumental_becruily.ckpt"
    assert payload["model_type"] == "vocal_models"
    assert payload["extract_instrumental"] == ["Vocals"]
    assert payload["params"]["range_start"] == 1
    assert payload["params"]["range_end"] == 1
    assert payload["params"]["device_ids"] == ["cuda1"]
