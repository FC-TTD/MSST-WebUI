"""Run original MSST functions inside the SDK's leased process group.

This module owns no inference queue. API semaphore and Gradio queues remain
native. The narrow multiprocessing facade records children for terminal proof;
it does not change their targets, parameters, precision or start method.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import os
import re
import threading
import time
from uuid import uuid4


UI_ENTRIES = {
    "msst.run_inference_single": ("msst", "run_inference_single"),
    "msst.run_multi_inference": ("msst", "run_multi_inference"),
    "msst.run_folder_batch_inference": ("msst", "run_folder_batch_inference"),
    "vr.vr_inference_single": ("vr", "vr_inference_single"),
    "vr.vr_inference_multi": ("vr", "vr_inference_multi"),
    "preset.preset_inference": ("preset", "preset_inference"),
    "preset.preset_inference_audio": ("preset", "preset_inference_audio"),
    "ensemble.inference_audio_func": ("ensemble", "inference_audio_func"),
    "ensemble.inference_folder_func": ("ensemble", "inference_folder_func"),
    "tools.some_inference": ("tools", "some_inference"),
    "train.validate_model": ("train", "validate_model"),
}
STOP_ENTRIES = {
    "msst": ("msst", "stop_msst_inference"),
    "vr": ("vr", "stop_vr_inference"),
    "preset": ("preset", "stop_preset"),
    "ensemble": ("ensemble", "stop_ensemble_func"),
    "valid": ("train", "stop_msst_valid"),
}
_scope = ContextVar("msst_native_children", default=None)


def _identity(pid):
    try:
        with open(f"/proc/{pid}/stat") as source:
            fields = source.read().rsplit(")", 1)[1].split()
        return int(fields[19]), fields[0], int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def _descendants(pid):
    """Read identities before native cancellation can reparent descendants."""
    records = {}
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            identity = _identity(int(entry.name))
            if identity is not None:
                records[int(entry.name)] = identity
    result, frontier = {}, {pid}
    while frontier:
        children = {child for child, value in records.items() if value[2] in frontier}
        for child in children:
            result[child] = records[child][0]
        frontier = children
        # Each PID has exactly one parent; protect against a reused/stale cycle.
        records = {child: value for child, value in records.items() if child not in children}
    return result


def _alive(pid, ticks):
    state = _identity(pid)
    return state is not None and state[0] == ticks and state[1] != "Z"


class NativeExecutionUnknown(RuntimeError):
    """Completion remains uncertain until the actor is reconciled/replaced."""


class _TrackedProcess:
    def __init__(self, process):
        self.process = process
        self.descendants = {}

    def __getattr__(self, name):
        return getattr(self.process, name)

    def capture(self):
        if self.process.pid and self.process.is_alive():
            self.descendants.update(_descendants(self.process.pid))

    def start(self):
        return self.process.start()

    def join(self, *args, **kwargs):
        self.capture()
        return self.process.join(*args, **kwargs)

    def terminate(self):
        self.capture()
        return self.process.terminate()

    def kill(self):
        self.capture()
        return self.process.kill()


class _Multiprocessing:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Process(self, *args, **kwargs):
        children = _scope.get()
        if children is None:
            raise RuntimeError("Native process launch requires a managed invocation")
        process = _TrackedProcess(self.original.Process(*args, **kwargs))
        children.append(process)
        return process


class _Gradio:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Progress(self, *args, **kwargs):
        from ttd_model_runtime.engine import engine_progress
        return engine_progress() or self.original.Progress(*args, **kwargs)


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _local_devices(values, *, labels=False):
    """Validate caller intent before native parsers silently fall back to 0."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Select the single GPU assigned to this managed service")
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    for value in values:
        token = str(value).strip()
        valid = token == "0" or re.fullmatch(r"cuda:?0", token) or re.fullmatch(r"0:\s*.+", token)
        if token.startswith("GPU-"):
            valid = token == assigned and "," not in assigned
        if isinstance(value, bool) or not valid:
            raise ValueError("Managed MSST supports only its assigned GPU (local cuda:0); multiple/unassigned GPUs are unsupported")
    return ["0: Assigned GPU"] if labels else [0]


class NativeSupervisor:
    def __init__(self, modules=None, *, join_timeout=3.0):
        self.modules = modules if modules is not None else {
            name: importlib.import_module(path) for name, path in {
                "services": "api.services", "msst": "webui.msst", "vr": "webui.vr",
                "preset": "webui.preset", "ensemble": "webui.ensemble",
                "tools": "webui.tools", "train": "webui.train",
            }.items()
        }
        self.join_timeout = join_timeout
        self._guard = threading.Lock()  # Metadata only; never holds inference.
        self._running = {}
        self._jobs = {}
        self._uncertain = []
        for module in self.modules.values():
            if hasattr(module, "multiprocessing"):
                original = module.multiprocessing
                if isinstance(original, _Multiprocessing):
                    original = original.original
                module.multiprocessing = _Multiprocessing(original)
            if hasattr(module, "gr"):
                original = module.gr
                if isinstance(original, _Gradio):
                    original = original.original
                module.gr = _Gradio(original)
        # The published sync API declares an unavailable-dependencies stub and
        # never resolves it. Bind only this intended native forwarding seam.
        self.modules["services"].run_folder_batch_inference = self.modules["msst"].run_folder_batch_inference

    def _request(self, request):
        from api.models import TaskCreateRequest
        request = dict(request)
        params = dict(request.get("params") or {})
        if not params.get("force_cpu", False) and params.get("device_ids") is not None:
            params["device_ids"] = _local_devices(params["device_ids"])
        request["params"] = params
        return TaskCreateRequest(**request)

    def _uncertain_exit(self, detail):
        with self._guard:
            self._uncertain.append(detail)
        raise NativeExecutionUnknown(detail)

    def _settle(self, children):
        deadline = time.monotonic() + self.join_timeout
        for child in children:
            if child.pid is None:
                continue
            child.capture()
            child.join(timeout=max(0, deadline - time.monotonic()))
            while any(_alive(pid, ticks) for pid, ticks in tuple(child.descendants.items())):
                if time.monotonic() >= deadline:
                    self._uncertain_exit("Native descendants have not exited")
                time.sleep(0.02)
            if child.is_alive():
                self._uncertain_exit("Native child has not exited")

    @contextmanager
    def _execution(self):
        invocation, children = uuid4().hex, []
        with self._guard:
            self._running[invocation] = children
        token = _scope.set(children)
        try:
            yield children
        finally:
            try:
                self._settle(children)
            except BaseException as exc:
                with self._guard:
                    self._uncertain.append(str(exc))
                raise
            finally:
                _scope.reset(token)
                with self._guard:
                    self._running.pop(invocation, None)

    def run_sync(self, request):
        with self._execution():
            return _plain(self.modules["services"].run_msst_batch_sync(self._request(request)))

    def run_sse(self, request, task_id=None):
        with self._execution():
            iterator = self.modules["services"].run_msst_batch_sse(self._request(request), existing_task_id=task_id)
            try:
                yield from iterator
            finally:
                iterator.close()

    def run_ui(self, entry, args):
        module, function = UI_ENTRIES[entry]
        args = list(args)
        if module == "msst" and not args[6]:
            args[4] = _local_devices(args[4], labels=True)
        elif entry == "train.validate_model":
            args[5] = _local_devices(args[5], labels=True)
        elif module in ("preset", "ensemble"):
            force_cpu = args[3] if module == "preset" else args[2]
            if not force_cpu:
                native = self.modules[module]
                devices = native.load_configs(native.WEBUI_CONFIG)["inference"].get("device")
                if devices:
                    # Preset code reads this setting itself; verify it is local.
                    _local_devices(devices, labels=True)
        with self._execution():
            return getattr(self.modules[module], function)(*args)

    def _capture_running(self):
        with self._guard:
            children = [child for group in self._running.values() for child in list(group)]
            children += [child for job in self._jobs.values() for child in job["children"]]
        for child in children:
            child.capture()

    def cancel_task(self, task_id):
        self._capture_running()
        return _plain(self.modules["services"].cancel_msst_sse_task(task_id))

    def stop_ui(self, kind):
        module, function = STOP_ENTRIES[kind]
        self._capture_running()
        return getattr(self.modules[module], function)()

    def start_training(self, args):
        """Yield acknowledgement, then retain the stream until actual exit.

        The parent may present the first message immediately, but its background
        activity owner must consume/close this stream, not end at the first yield.
        """
        args = list(args)
        args[6] = _local_devices(args[6], labels=True)
        job_id, children = uuid4().hex, []
        token = _scope.set(children)
        try:
            message = self.modules["train"].start_training(*args)
        except BaseException:
            self._settle(children)
            raise
        finally:
            _scope.reset(token)
        with self._guard:
            self._jobs[job_id] = {"children": children, "message": message}
        try:
            yield {"job_id": job_id, "message": message, "status": "started"}
            yield self.wait_job(job_id)
        finally:
            with self._guard:
                unfinished = self._jobs.get(job_id)
            if unfinished is not None:
                self._settle(unfinished["children"])
                with self._guard:
                    self._jobs.pop(job_id, None)

    def wait_job(self, job_id):
        with self._guard:
            job = self._jobs[job_id]
        for child in job["children"]:
            if child.pid is not None:
                child.join()
        self._settle(job["children"])
        with self._guard:
            self._jobs.pop(job_id)
        return {"job_id": job_id, "status": "finished", "message": job["message"]}

    def completion(self):
        # SDK Runtime._finish invokes this again: a failed call cannot erase
        # uncertainty and accidentally release its lease as an ordinary error.
        with self._guard:
            uncertain = list(self._uncertain)
        if uncertain:
            raise NativeExecutionUnknown(uncertain[-1])

    def release(self):
        self.completion()
        with self._guard:
            if self._running or self._jobs:
                raise NativeExecutionUnknown("Native work still owns this engine")
