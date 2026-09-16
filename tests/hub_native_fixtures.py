"""Importable CPU native-process fixture for the real SDK transport."""
import multiprocessing
import time
from types import SimpleNamespace

from hub_runtime.native import NativeSupervisor

_native = None


def sleep_worker():
    time.sleep(0.02)


def native_call():
    child = _native.multiprocessing.Process(target=sleep_worker)
    child.start()
    child.join()
    return {"exit_code": child.exitcode}


def load_fixture():
    global _native
    _native = SimpleNamespace(
        multiprocessing=multiprocessing.get_context("spawn"),
        semaphore=multiprocessing.get_context("spawn").Semaphore(2),
        some_inference=native_call,
        run_folder_batch_inference=native_call,
    )
    return NativeSupervisor({"services": SimpleNamespace(), "msst": _native, "tools": _native})
