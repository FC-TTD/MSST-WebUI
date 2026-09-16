"""CPU process evidence for MSST ownership, not model/GPU acceptance."""
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hub_runtime.native import NativeExecutionUnknown, NativeSupervisor, _local_devices


def _sleep(seconds):
    time.sleep(seconds)


def _orphan(pid_file):
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    Path(pid_file).write_text(str(child.pid))
    time.sleep(30)


def supervisor(callback=None, *, timeout=0.05):
    native = SimpleNamespace(multiprocessing=multiprocessing.get_context("spawn"))
    native.run_folder_batch_inference = lambda *a: None
    native.some_inference = callback or (lambda *a: None)
    services = SimpleNamespace(multiprocessing=multiprocessing.get_context("spawn"))
    model = NativeSupervisor({"msst": native, "services": services, "train": native, "tools": native}, join_timeout=timeout)
    return model, native, services


def test_no_blanket_lock_and_children_finish():
    model, native, _ = supervisor(timeout=1)
    barrier = threading.Barrier(2)
    def callback():
        barrier.wait(timeout=2)
        child = native.multiprocessing.Process(target=_sleep, args=(0.02,))
        child.start()
        child.join()
        return child.exitcode
    native.some_inference = callback
    results = []
    threads = [threading.Thread(target=lambda: results.append(model.run_ui("tools.some_inference", []))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert results == [0, 0]
    model.completion()
    model.release()


def test_native_early_return_is_sticky_unknown():
    model, native, _ = supervisor()
    children = []
    def callback():
        child = native.multiprocessing.Process(target=_sleep, args=(30,))
        children.append(child)
        child.start()
        return "already returned"
    native.some_inference = callback
    try:
        with pytest.raises(NativeExecutionUnknown):
            model.run_ui("tools.some_inference", [])
        with pytest.raises(NativeExecutionUnknown):
            model.completion()
    finally:
        children[0].terminate()
        children[0].join()
    # Later exit is not sufficient to silently rewrite the unknown terminal.
    with pytest.raises(NativeExecutionUnknown):
        model.completion()


def test_sse_close_runs_native_finally_and_waits_child():
    model, native, services = supervisor(timeout=1)
    model._request = lambda value: value
    children = []
    def run(request, existing_task_id=None):
        child = services.multiprocessing.Process(target=_sleep, args=(30,))
        children.append(child)
        child.start()
        try:
            yield existing_task_id
        finally:
            child.terminate()
            child.join()
    services.run_msst_batch_sse = run
    stream = model.run_sse({}, "same-task-id")
    assert next(stream) == "same-task-id"
    assert children[0].is_alive()
    stream.close()
    assert not children[0].is_alive()
    model.completion()
    model.release()


def test_native_cancel_does_not_hide_live_grandchild(tmp_path):
    model, native, services = supervisor(timeout=0.05)
    pid_file = tmp_path / "grandchild.pid"
    children, errors = [], []
    def callback():
        child = native.multiprocessing.Process(target=_orphan, args=(str(pid_file),))
        children.append(child)
        child.start()
        child.join()
    native.some_inference = callback
    services.cancel_msst_sse_task = lambda task_id: (children[0].terminate(), {"status": "canceled"})[1]
    def invoke():
        try:
            model.run_ui("tools.some_inference", [])
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_file.exists()
    grandchild = int(pid_file.read_text())
    try:
        assert model.cancel_task("task") == {"status": "canceled"}
        thread.join(timeout=5)
        assert len(errors) == 1
        assert isinstance(errors[0], NativeExecutionUnknown)
        with pytest.raises(NativeExecutionUnknown):
            model.completion()
    finally:
        os.kill(grandchild, signal.SIGKILL)
        if children[0].is_alive():
            children[0].terminate()
        children[0].join()


def test_training_stream_retains_job_after_start_ack():
    model, native, _ = supervisor(timeout=1)
    def start_training(*args):
        child = native.multiprocessing.Process(target=_sleep, args=(0.1,))
        child.start()
        return "native startup message"
    native.start_training = start_training
    stream = model.start_training([None] * 6 + [[0]])
    first = next(stream)
    assert first["message"] == "native startup message"
    assert first["status"] == "started"
    with pytest.raises(NativeExecutionUnknown):
        model.release()
    final = next(stream)
    assert final["job_id"] == first["job_id"]
    assert final["status"] == "finished"
    with pytest.raises(StopIteration):
        next(stream)
    model.release()


def test_unsupported_ui_target_cannot_run_arbitrary_function():
    model, _, _ = supervisor()
    with pytest.raises(KeyError):
        model.run_ui("os.system", ["echo forbidden"])


def test_sync_reuses_real_folder_function_and_plain_result():
    model, native, services = supervisor()
    model._request = lambda value: value
    assert services.run_folder_batch_inference is native.run_folder_batch_inference
    services.run_msst_batch_sync = lambda request: SimpleNamespace(model_dump=lambda: {"task_id": request["id"]})
    assert model.run_sync({"id": "native-id"}) == {"task_id": "native-id"}


@pytest.mark.parametrize("value", [1, "cuda:1", "GPU-other", "bad", True])
def test_device_selection_never_silently_falls_back(value):
    with pytest.raises(ValueError):
        _local_devices([value])


def test_device_uuid_and_ordinal_are_equivalent(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned")
    for value in [0, "0", "cuda0", "cuda:0", "0: RTX 3090", "GPU-assigned"]:
        assert _local_devices([value]) == [0]


def test_real_sdk_roundtrip_spawn_child_and_group_exit(capfd):
    from hub_native_fixtures import load_fixture
    from hub_runtime.adapter import completion, cleanup, release
    from ttd_model_runtime.engine import ProcessModel
    lease = {"gpu": "GPU-fixture", "generation": 1}
    engine = ProcessModel(load_fixture, completion, cleanup, release, lease).start()
    try:
        assert engine.run_ui("tools.some_inference", []) == {"exit_code": 0}
        engine.complete()
    finally:
        engine.close()
    assert engine.engine_status()["group_empty"]
    assert "resource_tracker:" not in capfd.readouterr().err


def test_sdk_finish_persists_unknown_even_if_native_call_raised_ordinary_error():
    from hub_runtime.adapter import completion
    from ttd_model_runtime import Runtime, HubError
    model, _, _ = supervisor()
    with pytest.raises(NativeExecutionUnknown):
        model._uncertain_exit("child exit not proven")
    runtime = Runtime(lambda: model, completion=completion, gpu_process=False)
    runtime._model = model
    runtime.service = "msst"
    runtime._db = sqlite3.connect(":memory:")
    runtime._db.execute("CREATE TABLE activities (id TEXT, state TEXT, pending INTEGER)")
    runtime._db.execute("CREATE TABLE admissions (key TEXT, state TEXT)")
    runtime._db.execute("INSERT INTO activities VALUES ('activity', 'running', 0)")
    runtime._db.execute("INSERT INTO admissions VALUES ('request', 'running')")
    runtime._save_status = lambda: runtime._db.commit()
    runtime.flush = lambda: None
    try:
        with pytest.raises(HubError) as failure:
            runtime._finish({"id": "activity", "key": "request"}, "failed")
        assert failure.value.code == "execution_unknown"
        assert runtime._db.execute("SELECT state FROM activities").fetchone()[0] == "unknown"
        assert runtime._status["residency"] == "unknown"
    finally:
        runtime._db.close()
