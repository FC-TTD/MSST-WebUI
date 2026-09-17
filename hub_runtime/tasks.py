"""Own real native work beyond an HTTP task receipt, without changing its queue.

An owner holds the Runtime execution scope and engine reference until the remote
generator/process tree finishes. Control calls use that same reference and never
acquire a fresh GPU lease just to cancel an old task.
"""
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass, field
import logging
import queue
import threading
from ttd_model_runtime import NativeCancelled

logger = logging.getLogger(__name__)


def plain(value):
    return value.model_dump() if hasattr(value, 'model_dump') else value.dict()


@dataclass
class Owner:
    kind: str
    task_id: str | None = None
    engine: object = None
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


class NativeTaskFailed(RuntimeError):
    def __init__(self, result):
        super().__init__('native MSST task failed')
        self.result = result


class OwnedStream:
    """HTTP iteration never owns a Runtime ContextVar across separate next calls."""
    def __init__(self, tasks, req, owner):
        self.tasks, self.req, self.owner = tasks, req, owner
        self.context = copy_context()
        self.queue = queue.Queue(maxsize=2)
        self.closed = threading.Event()
        self.guard = threading.Lock()
        self.thread = None
        self.exhausted = False

    def _put(self, kind, value=None):
        while not self.closed.is_set():
            try:
                self.queue.put((kind, value), timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def _run(self):
        iterator = self.tasks._stream(self.req, self.owner)
        failure = None
        try:
            for event in iterator:
                if not self._put('event', event):
                    break
        except NativeCancelled:
            pass  # Native canceled event/result is preserved; no stream error.
        except BaseException as exc:
            failure = exc
            self.tasks._record_error(self.owner.task_id, exc)
            self._put('error', exc)
        finally:
            try:
                iterator.close()
            except BaseException as exc:
                self.tasks._record_error(self.owner.task_id, exc)
                self._put('error', exc)
            else:
                if failure is None and self.closed.is_set() and self.tasks._status(self.owner.task_id) in ('queued', 'running'):
                    # HTTP close is only a request to native finally; write the
                    # terminal state after cleanup and SDK completion succeed.
                    self.tasks.storage.update_task(self.owner.task_id, status='canceled')
            self._put('end')

    def __iter__(self):
        return self

    def __next__(self):
        with self.guard:
            if self.closed.is_set() or self.exhausted:
                raise StopIteration
            if self.thread is None:
                self.thread = threading.Thread(target=self.context.run, args=(self._run,),
                    name='hub-msst-sse-'+self.owner.task_id, daemon=False)
                try:
                    self.thread.start()
                except BaseException:
                    self.close_unstarted()
                    raise
        while not self.closed.is_set():
            try:
                kind, value = self.queue.get(timeout=.1)
                break
            except queue.Empty:
                pass
        else:
            raise StopIteration
        if kind == 'event':
            return value
        self.exhausted = True
        if kind == 'error':
            raise value
        raise StopIteration

    def close_unstarted(self):
        self.closed.set()
        self.owner.cancel.set()
        self.tasks.storage.update_task(self.owner.task_id, status='canceled')
        self.tasks._remove(self.owner)

    def close(self):
        with self.guard:
            if self.thread is None:
                self.close_unstarted()
            else:
                self.closed.set()
                # Native generator close/finally runs on the owner, not here.


def streaming_response(stream):
    from fastapi.responses import StreamingResponse

    class Response(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                stream.close()

    return Response(stream, media_type='text/event-stream')


class Tasks:
    def __init__(self, runtime, storage):
        self.runtime = runtime
        self.storage = storage
        self._guard = threading.RLock()
        self._owners = {}

    def pending_api_count(self):
        # Async receipts may precede the worker entering Runtime.execution.
        # Gradio UI work is counted separately by the shared SDK integration.
        with self._guard:
            return sum(owner.kind == 'api' for owner in self._owners.values())

    def _add(self, owner):
        with self._guard:
            self._owners[id(owner)] = owner

    def _remove(self, owner):
        owner.done.set()
        with self._guard:
            self._owners.pop(id(owner), None)

    def _status(self, task_id):
        state = self.storage.get_task_status(task_id)
        return state.status if state is not None else None

    def _stream(self, req, owner):
        iterator = None
        try:
            if owner.cancel.is_set():
                self.storage.update_task(owner.task_id, status='canceled')
                return
            with self.runtime.execution():
                engine = self.runtime.get()
                with self._guard:
                    owner.engine = engine
                if owner.cancel.is_set():
                    self.storage.update_task(owner.task_id, status='canceled')
                    raise NativeCancelled()
                iterator = engine.run_sse(plain(req), owner.task_id)
                try:
                    yield from iterator
                finally:
                    iterator.close()
                # Native errors are often SSE events, not raised exceptions.
                # Preserve API task state while reporting a failed GPU activity.
                if self._status(owner.task_id) == 'failed':
                    raise RuntimeError('native MSST task failed')
                if self._status(owner.task_id) == 'canceled':
                    raise NativeCancelled()
        finally:
            self._remove(owner)

    def run_sse(self, req):
        task_id = self.storage.create_task(status='queued', message=None)
        owner = Owner('api', task_id)
        self._add(owner)
        return OwnedStream(self, req, owner)

    def _record_error(self, task_id, exc):
        if getattr(exc, 'code', None) == 'execution_unknown':
            self.storage.update_task(task_id, status='running', error='execution_unknown')
        elif self._status(task_id) not in ('failed', 'canceled', 'success'):
            self.storage.update_task(task_id, status='failed', error=getattr(exc, 'code', type(exc).__name__))

    def start(self, req):
        from api.models import TaskCreateResponse
        task_id = self.storage.create_task(status='queued', message=None)
        owner = Owner('api', task_id)
        self._add(owner)
        context = copy_context()

        def execute():
            try:
                for _ in self._stream(req, owner):
                    pass
            except NativeCancelled:
                pass
            except BaseException as exc:
                # SDK completion is authoritative for unknown execution. Do not
                # turn it into a false successful/canceled native task result.
                self._record_error(task_id, exc)
                logger.exception('Managed MSST background task ended with an error: %s', task_id)

        thread = threading.Thread(target=context.run, args=(execute,),
                                  name='hub-msst-'+task_id, daemon=False)
        try:
            thread.start()
        except BaseException:
            self._remove(owner)
            self.storage.update_task(task_id, status='failed', error='owner_start_failed')
            raise
        return TaskCreateResponse(task_id=task_id, status='queued', message=None)

    def run_sync(self, req):
        from api.models import TaskResultResponse
        try:
            with self.runtime.execution():
                result = self.runtime.get().run_sync(plain(req))
                if result.get('status') == 'failed':
                    raise NativeTaskFailed(result)
                if result.get('status') == 'canceled':
                    raise NativeCancelled()
        except NativeTaskFailed as exc:
            result = exc.result
        except NativeCancelled:
            pass
        return TaskResultResponse(**result)

    def cancel_task(self, task_id):
        from api.models import TaskResultResponse
        with self._guard:
            owner = next((o for o in self._owners.values() if o.task_id == task_id), None)
            if owner is not None:
                owner.cancel.set()
                engine = owner.engine
            else:
                engine = None
        if owner is None:
            # No live engine is resurrected to answer a historical task query.
            return self.storage.get_task_result(task_id)
        if engine is None:
            self.storage.update_task(task_id, status='canceled')
            return TaskResultResponse(task_id=task_id, status='canceled', files=[])
        result = engine.cancel_task(task_id)
        return TaskResultResponse(**result) if result is not None else None

    def run_ui(self, entry, *args):
        from ttd_model_runtime.engine import progress_scope
        import gradio as gr
        owner = Owner('valid' if entry == 'train.validate_model' else entry.split('.', 1)[0])
        self._add(owner)
        try:
            with self.runtime.execution():
                with self._guard:
                    owner.engine = self.runtime.get()
                progress = gr.Progress()
                # Treat the progress object as callable, never as a collection;
                # Gradio 4's __len__ assumes an active tqdm iterable.
                with progress_scope(lambda *values, **options: progress(*values, **options)):
                    return owner.engine.run_ui(entry, list(args))
        finally:
            self._remove(owner)

    def stop_ui(self, kind):
        with self._guard:
            engines = {id(o.engine):o.engine for o in self._owners.values()
                       if o.kind == kind and o.engine is not None and not o.done.is_set()}
        result = None
        for engine in engines.values():
            result = engine.stop_ui(kind)
        return result

    def start_training(self, *args):
        """Return native startup feedback, while a background owner waits for exit."""
        owner = Owner('train')
        self._add(owner)
        context = copy_context()
        started = Future()

        def execute():
            try:
                with self.runtime.execution():
                    with self._guard:
                        owner.engine = self.runtime.get()
                    jobs = owner.engine.start_training(list(args))
                    try:
                        job = next(jobs)
                        started.set_result(job['message'])
                        for _ in jobs:
                            pass
                    finally:
                        jobs.close()
            except BaseException as exc:
                if not started.done():
                    started.set_exception(exc)
                else:
                    logger.exception('Managed MSST training owner failed')
            finally:
                self._remove(owner)

        thread = threading.Thread(target=context.run, args=(execute,),
                                  name='hub-msst-training', daemon=False)
        try:
            thread.start()
        except BaseException:
            self._remove(owner)
            raise
        return started.result()
