"""Exercise actual Starlette iteration/disconnect with one native scope owner."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import threading
from types import SimpleNamespace
import unittest

from starlette.concurrency import iterate_in_threadpool
from hub_runtime.tasks import Tasks, streaming_response


class Unknown(RuntimeError):
    code = "execution_unknown"


class Storage:
    def __init__(self):
        self.rows = {}
    def create_task(self, status, message):
        key = str(len(self.rows) + 1)
        self.rows[key] = SimpleNamespace(status=status)
        return key
    def update_task(self, key, **fields):
        self.rows[key].__dict__.update(fields)
    def get_task_status(self, key):
        return self.rows.get(key)
    get_task_result = get_task_status


class Request:
    def model_dump(self):
        return {}


class Runtime:
    def __init__(self, engine, *, unknown=False):
        self.engine, self.unknown = engine, unknown
        self.active = 0
        self.states = []
        self.threads = []
        self.scope = ContextVar("fixture_sdk_scope")
    @contextmanager
    def execution(self):
        token = self.scope.set("active")
        self.active += 1
        self.threads.append(threading.get_ident())
        state = "succeeded"
        try:
            yield
        except BaseException:
            state = "failed"
            raise
        finally:
            # SDK execution() resets before _finish; wrong-context iteration
            # would throw here and leave active > 0, as it did before the fix.
            self.scope.reset(token)
            self.active -= 1
            self.threads.append(threading.get_ident())
            self.states.append("unknown" if self.unknown else state)
            if self.unknown:
                raise Unknown("completion is uncertain")
    def get(self):
        return self.engine


class StreamLifecycleTest(unittest.TestCase):
    def finish(self, stream):
        stream.close()
        if stream.thread is not None:
            stream.thread.join(3)
            self.assertFalse(stream.thread.is_alive(), "stream owner did not exit")

    def test_real_starlette_iteration_uses_one_scope_context_to_completion(self):
        storage = Storage()
        trace = ContextVar("request_trace")
        seen = []
        class Engine:
            def run_sse(self, req, task_id):
                for i in range(3):
                    seen.append(trace.get())
                    yield str(i)
                storage.update_task(task_id, status="success")
        runtime = Runtime(Engine())
        tasks = Tasks(runtime, storage)
        token = trace.set("original-request")
        stream = tasks.run_sse(Request())
        trace.reset(token)
        async def consume():
            return [item async for item in iterate_in_threadpool(stream)]
        try:
            self.assertEqual(asyncio.run(consume()), ["0", "1", "2"])
        finally:
            self.finish(stream)
        self.assertEqual(seen, ["original-request"] * 3)
        self.assertEqual(runtime.active, 0)
        self.assertEqual(runtime.states, ["succeeded"])
        self.assertEqual(len(set(runtime.threads)), 1)
        self.assertEqual(tasks._owners, {})

    def test_response_failure_before_first_consumption_cleans_queued_owner(self):
        storage = Storage()
        runtime = Runtime(None)
        tasks = Tasks(runtime, storage)
        stream = tasks.run_sse(Request())
        response = streaming_response(stream)
        async def send(event):
            raise OSError("client closed before response started")
        async def receive():
            await asyncio.Future()
        async def invoke():
            try:
                await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
            except BaseException:
                pass
        asyncio.run(invoke())
        self.assertTrue(stream.owner.done.is_set())
        self.assertIsNone(stream.thread)
        self.assertEqual(storage.rows[stream.owner.task_id].status, "canceled")
        self.assertEqual(tasks._owners, {})
        self.assertEqual(runtime.active, 0)

    def test_http_disconnect_closes_on_owner_and_retains_activity_until_cleanup(self):
        storage = Storage()
        closing, release = threading.Event(), threading.Event()
        cleanup_threads = []
        class Engine:
            def run_sse(self, req, task_id):
                storage.update_task(task_id, status="running")
                try:
                    while True:
                        yield "data: progress\n\n"
                finally:
                    cleanup_threads.append(threading.get_ident())
                    closing.set()
                    if not release.wait(3):
                        raise TimeoutError("test cleanup release missing")
                    storage.update_task(task_id, status="canceled")
        runtime = Runtime(Engine())
        tasks = Tasks(runtime, storage)
        stream = tasks.run_sse(Request())
        response = streaming_response(stream)
        async def invoke():
            disconnect = asyncio.Event()
            async def send(event):
                if event["type"] == "http.response.body":
                    disconnect.set()
                    await asyncio.sleep(0)
            async def receive():
                await disconnect.wait()
                return {"type": "http.disconnect"}
            await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
        try:
            asyncio.run(invoke())
            self.assertTrue(closing.wait(3))
            self.assertEqual(runtime.active, 1)
            self.assertFalse(stream.owner.done.is_set())
        finally:
            release.set()
            self.finish(stream)
        self.assertEqual(runtime.active, 0)
        self.assertEqual(cleanup_threads, [runtime.threads[0]])
        self.assertEqual(tasks._owners, {})

    def test_completion_unknown_is_not_rewritten_as_success_or_canceled(self):
        storage = Storage()
        class Engine:
            def run_sse(self, req, task_id):
                storage.update_task(task_id, status="running")
                yield "data: progress\n\n"
        runtime = Runtime(Engine(), unknown=True)
        tasks = Tasks(runtime, storage)
        stream = tasks.run_sse(Request())
        try:
            with self.assertRaises(Unknown):
                list(stream)
        finally:
            self.finish(stream)
        row = storage.rows[stream.owner.task_id]
        self.assertEqual(row.status, "running")
        self.assertEqual(row.error, "execution_unknown")
        self.assertEqual(runtime.states, ["unknown"])

    def test_closed_stream_records_unknown_from_native_cleanup(self):
        storage = Storage()
        class Engine:
            def run_sse(self, req, task_id):
                storage.update_task(task_id, status="running")
                while True:
                    yield "progress"
        runtime = Runtime(Engine(), unknown=True)
        tasks = Tasks(runtime, storage)
        stream = tasks.run_sse(Request())
        next(stream)
        self.finish(stream)
        row = storage.rows[stream.owner.task_id]
        self.assertEqual(row.status, "running")
        self.assertEqual(row.error, "execution_unknown")
        self.assertEqual(runtime.states, ["unknown"])

    def test_proven_stream_cleanup_does_not_leave_native_task_running(self):
        storage = Storage()
        class Engine:
            def run_sse(self, req, task_id):
                storage.update_task(task_id, status="running")
                # Native api.services finally stops/reaps the subprocess but
                # does not update task status when closed through GeneratorExit.
                try:
                    while True:
                        yield "progress"
                finally:
                    pass
        runtime = Runtime(Engine())
        tasks = Tasks(runtime, storage)
        stream = tasks.run_sse(Request())
        next(stream)
        self.finish(stream)
        self.assertEqual(runtime.active, 0)
        self.assertTrue(stream.owner.done.is_set())
        self.assertEqual(storage.rows[stream.owner.task_id].status, "canceled")

    def test_sync_failed_result_preserves_api_shape_and_records_failed_activity(self):
        class Engine:
            def run_sync(self, request):
                return {"task_id": "failed-task", "status": "failed", "files": []}
        runtime = Runtime(Engine())
        result = Tasks(runtime, Storage()).run_sync(Request())
        self.assertEqual(result.status, "failed")
        self.assertEqual(runtime.states, ["failed"])
        runtime = Runtime(Engine(), unknown=True)
        with self.assertRaises(Unknown):
            Tasks(runtime, Storage()).run_sync(Request())
        self.assertEqual(runtime.states, ["unknown"])


if __name__ == "__main__":
    unittest.main()
