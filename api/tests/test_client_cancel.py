from __future__ import annotations

from typing import Any, Dict

from api.client import MSSTApiClient


def test_client_cancel_task_builds_request() -> None:
    captured: list[dict[str, Any]] = []

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200

        def json(self) -> Dict[str, Any]:
            return {"task_id": "t1", "status": "canceled"}

    class DummySession:
        def request(self, method: str, url: str, timeout=None, **kwargs):  # type: ignore[override]
            captured.append({"method": method, "url": url, "kwargs": kwargs})
            return DummyResponse()

    c = MSSTApiClient(base_url="http://testserver", session=DummySession())  # type: ignore[arg-type]
    resp = c.cancel_task("t1")

    assert resp["status"] == "canceled"
    assert len(captured) == 1
    assert captured[0]["method"] == "POST"
    assert captured[0]["url"].endswith("/api/v1/tasks/t1/cancel")
