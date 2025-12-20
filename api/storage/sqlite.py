from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from api.models import FileResult, TaskResultResponse, TaskStatusResponse
from api.storage import BaseStorage


_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    progress REAL,
    total_files INTEGER,
    processed_files INTEGER,
    error TEXT,
    message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_FILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    input_file TEXT NOT NULL,
    output_files TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
);
"""


class SQLiteStorage(BaseStorage):
    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(_TASKS_SCHEMA)
            cur.execute(_FILES_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def create_task(self, status: str, message: str | None) -> str:
        task_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO tasks (task_id, status, progress, total_files, processed_files, error, message, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, status, None, None, None, None, message, now, now),
            )
            conn.commit()
        finally:
            conn.close()
        return task_id

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
        fields = []
        values = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(progress)
        if total_files is not None:
            fields.append("total_files = ?")
            values.append(total_files)
        if processed_files is not None:
            fields.append("processed_files = ?")
            values.append(processed_files)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(task_id)
        sql = f"UPDATE tasks SET {' ,'.join(fields)} WHERE task_id = ?"
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, values)
            conn.commit()
        finally:
            conn.close()

    def save_results(self, task_id: str, files: List[FileResult]) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM task_files WHERE task_id = ?", (task_id,))
            for f in files:
                cur.execute(
                    "INSERT INTO task_files (task_id, input_file, output_files, status, error) VALUES (?, ?, ?, ?, ?)",
                    (
                        task_id,
                        f.input_file,
                        json.dumps(f.output_files, ensure_ascii=False),
                        f.status,
                        f.error,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT task_id, status, progress, total_files, processed_files, error, created_at, updated_at, message FROM tasks WHERE task_id = ?",
                (task_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return TaskStatusResponse(
            task_id=row[0],
            status=row[1],
            progress=row[2],
            total_files=row[3],
            processed_files=row[4],
            error=row[5],
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
        )

    def get_task_result(self, task_id: str) -> Optional[TaskResultResponse]:
        status = self.get_task_status(task_id)
        if not status:
            return None
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT input_file, output_files, status, error FROM task_files WHERE task_id = ?",
                (task_id,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        files: List[FileResult] = []
        for r in rows:
            files.append(
                FileResult(
                    input_file=r[0],
                    output_files=json.loads(r[1]) if r[1] else [],
                    status=r[2],
                    error=r[3],
                )
            )
        return TaskResultResponse(task_id=task_id, status=status.status, files=files)
