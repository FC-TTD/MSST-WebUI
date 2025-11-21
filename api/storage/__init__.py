from __future__ import annotations

import abc
from typing import List, Optional

from api.config import DEFAULT_DB_PATH, DEFAULT_STORAGE_BACKEND
from api.models import FileResult, TaskStatusResponse, TaskResultResponse


class BaseStorage(abc.ABC):
    @abc.abstractmethod
    def create_task(self, status: str, message: str | None) -> str:
        """创建任务记录，返回 task_id。"""

    @abc.abstractmethod
    def update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        total_files: Optional[int] = None,
        processed_files: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """更新任务元数据。"""

    @abc.abstractmethod
    def save_results(self, task_id: str, files: List[FileResult]) -> None:
        """保存任务结果。"""

    @abc.abstractmethod
    def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """获取任务状态信息。"""

    @abc.abstractmethod
    def get_task_result(self, task_id: str) -> Optional[TaskResultResponse]:
        """获取任务结果信息。"""


def get_storage() -> BaseStorage:
    backend = DEFAULT_STORAGE_BACKEND.lower()
    if backend == "sqlite":
        from api.storage.sqlite import SQLiteStorage

        return SQLiteStorage(DEFAULT_DB_PATH)
    raise ValueError(f"Unsupported storage backend: {backend}")
