"""GPU-process-only adapter preserving native MSST child-process execution."""
import multiprocessing
import os
import sys


def load_model():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Managed MSST requires its assigned CUDA GPU")
    multiprocessing.set_start_method("spawn", force=True)
    from .native import NativeSupervisor
    return NativeSupervisor()


def completion(model):
    model.completion()
    # SOME runs CUDA in this supervisor rather than a per-task subprocess.
    # Synchronize an already initialized context; do not create one for the
    # usual child-process-only separation path.
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def release(model):
    model.release()
    _stop_owned_resource_tracker()


def _stop_owned_resource_tracker():
    """Reap CPython's spawn helper before SDK verifies an empty process group.

    Verified on CPython 3.11 by a real SDK spawn/close test. This is an unload
    accommodation, never a per-request inference hook. Remove it when Python
    exposes a public owned-tracker shutdown API or SDK provides equivalent
    acknowledged helper teardown. Missing/changed private APIs fail closed.
    """
    from multiprocessing import resource_tracker, synchronize, util
    from .native import NativeExecutionUnknown, _identity

    if sys.implementation.name != "cpython":
        raise NativeExecutionUnknown("MSST tracker cleanup requires verified CPython semantics")
    tracker = getattr(resource_tracker, "_resource_tracker", None)
    if tracker is None or not hasattr(tracker, "_stop"):
        raise NativeExecutionUnknown("Owned tracker shutdown API unavailable")
    pid = getattr(tracker, "_pid", None)
    fd = getattr(tracker, "_fd", None)
    if pid is None and fd is None:
        return
    state = _identity(pid) if isinstance(pid, int) and pid > 0 else None
    if state is None or state[2] != os.getpid() or os.getpgrp() != os.getpid():
        raise NativeExecutionUnknown("Cannot prove tracker belongs to this SDK engine")
    if os.getpgid(pid) != os.getpgrp():
        raise NativeExecutionUnknown("Tracker is outside the SDK engine group")
    with open(f"/proc/{pid}/cmdline", "rb") as source:
        if b"multiprocessing.resource_tracker" not in source.read():
            raise NativeExecutionUnknown("Tracker process identity mismatch")
    registry = getattr(util, "_finalizer_registry", None)
    sem_cleanup = getattr(synchronize.SemLock, "_cleanup", None)
    if registry is None or sem_cleanup is None:
        raise NativeExecutionUnknown("Owned semaphore cleanup API unavailable")
    # Run only this engine's semaphore finalizers. Calling Finalize itself
    # unregisters it, avoiding double sem_unlink at interpreter shutdown, and
    # propagates failures (util._run_finalizers would merely print them).
    for finalizer in list(registry.values()):
        if (getattr(finalizer, "_pid", None) == os.getpid()
                and getattr(finalizer, "_callback", None) is sem_cleanup):
            finalizer()
    current = _identity(pid)
    if current is None or current[0] != state[0] or current[2] != state[2]:
        raise NativeExecutionUnknown("Tracker identity changed before owned shutdown")
    tracker._stop()  # CPython closes the alive pipe and waitpid()s its own child.
    if getattr(tracker, "_pid", None) is not None or _identity(pid) is not None:
        raise NativeExecutionUnknown("Tracker exit not confirmed")


def cleanup():
    # Native task processes own weights and release them when they exit.
    # No CPU relocation, precision change, new cache or queue is introduced.
    pass
