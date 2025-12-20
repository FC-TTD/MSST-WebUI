from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TaskRuntime:
    task_id: str
    pid: int
    task_output_dir: str
    output_dir: str


_registry_lock = threading.Lock()
_registry: Dict[str, TaskRuntime] = {}
_task_locks: Dict[str, threading.Lock] = {}
_cancel_events: Dict[str, threading.Event] = {}


def register_task_runtime(runtime: TaskRuntime) -> None:
    with _registry_lock:
        _registry[runtime.task_id] = runtime
        _task_locks.setdefault(runtime.task_id, threading.Lock())
        _cancel_events.setdefault(runtime.task_id, threading.Event())


def unregister_task_runtime(task_id: str) -> None:
    with _registry_lock:
        _registry.pop(task_id, None)
        _cancel_events.pop(task_id, None)


def get_task_runtime(task_id: str) -> Optional[TaskRuntime]:
    with _registry_lock:
        return _registry.get(task_id)


def get_task_lock(task_id: str) -> threading.Lock:
    with _registry_lock:
        lock = _task_locks.get(task_id)
        if lock is None:
            lock = threading.Lock()
            _task_locks[task_id] = lock
        return lock


def get_cancel_event(task_id: str) -> threading.Event:
    with _registry_lock:
        ev = _cancel_events.get(task_id)
        if ev is None:
            ev = threading.Event()
            _cancel_events[task_id] = ev
        return ev
