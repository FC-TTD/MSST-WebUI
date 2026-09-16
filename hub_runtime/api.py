"""API routes copied from api/app.py at 41d0334; execution wiring only."""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.models import ErrorResponse, ModelListResponse, TaskCreateRequest, TaskCreateResponse, TaskResultResponse, TaskStatusResponse
from api.services import InferenceBusyError, list_models
from .tasks import Tasks, streaming_response
from api.storage import get_storage


def build_api(runtime, tasks=None):
    app = FastAPI()
    tasks = tasks or Tasks(runtime, get_storage())
    app.state.hub_tasks = tasks
    cancel_msst_sse_task = tasks.cancel_task
    run_msst_batch_sse = tasks.run_sse
    run_msst_batch_sync = tasks.run_sync
    start_msst_batch_async = tasks.start
    router = APIRouter(prefix="/api/v1", tags=["msst"])


    @router.post("/tasks/msst-batch", response_model=TaskCreateResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    async def create_msst_batch_task(
        req: TaskCreateRequest,
        mode: str = Query("sync", pattern="^(sync|sse|async)$", description="返回模式：sync=同步JSON，sse=事件流，async=立即返回 task ID 后轮询"),
    ) -> TaskCreateResponse | StreamingResponse:
        """创建一次批量 MSST 分离任务。

        当前默认模式为 sync（blocking JSON）：请求会一直阻塞直到任务完成，再返回最终状态。
        当 mode=sse 时，返回标准 SSE 事件流，渐进式推送任务进度。
        当 mode=async 时，立即返回 task ID，调用方通过任务查询接口轮询状态和结果。
        """
        try:
            if mode == "async":
                return start_msst_batch_async(req)
            if mode == "sse":
                return streaming_response(run_msst_batch_sse(req))

            result: TaskResultResponse = run_msst_batch_sync(req)
            return TaskCreateResponse(task_id=result.task_id, status=result.status, message=None)
        except InferenceBusyError as e:
            raise HTTPException(status_code=429, detail=ErrorResponse(error_message=str(e)).dict())


    @router.get("/tasks/{task_id}", response_model=TaskStatusResponse, responses={404: {"model": ErrorResponse}})
    async def get_task_status(task_id: str) -> TaskStatusResponse:
        storage = get_storage()
        status = storage.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail=ErrorResponse(error_message="task not found").dict())
        return status


    @router.get("/tasks/{task_id}/result", response_model=TaskResultResponse, responses={404: {"model": ErrorResponse}})
    async def get_task_result(task_id: str) -> TaskResultResponse:
        storage = get_storage()
        result = storage.get_task_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail=ErrorResponse(error_message="task not found").dict())
        return result


    @router.post("/tasks/{task_id}/cancel", response_model=TaskResultResponse, responses={404: {"model": ErrorResponse}})
    async def cancel_task(task_id: str) -> TaskResultResponse:
        result = cancel_msst_sse_task(task_id)
        if not result:
            raise HTTPException(status_code=404, detail=ErrorResponse(error_message="task not found or not cancelable").dict())
        return result


    @router.get("/models", response_model=ModelListResponse)
    async def get_models(model_class: str | None = Query(None, description="模型类别，例如 VR_Models/multi_stem_models 等")) -> ModelListResponse:
        """列出可用模型，支持按 model_class 过滤。"""

        models = list_models(model_class=model_class)
        return ModelListResponse(models=models)
    app.include_router(router)
    return app
