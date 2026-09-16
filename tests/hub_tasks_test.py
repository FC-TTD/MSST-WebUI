from contextlib import contextmanager
from contextvars import ContextVar
import threading
from types import SimpleNamespace
import unittest

from hub_runtime.tasks import Tasks


class Storage:
    def __init__(self):
        self.rows = {}
    def create_task(self, status, message):
        task_id = str(len(self.rows)+1)
        self.rows[task_id] = SimpleNamespace(status=status)
        return task_id
    def update_task(self, task_id, **fields):
        self.rows[task_id].__dict__.update(fields)
    def get_task_status(self, task_id):
        return self.rows.get(task_id)
    def get_task_result(self, task_id):
        return self.rows.get(task_id)


class Request:
    def model_dump(self):
        return {'model_name': 'native'}


class TasksTest(unittest.TestCase):
    def test_async_receipt_holds_activity_until_native_exit_and_preserves_context(self):
        trace = ContextVar('trace')
        entered, release, ended = threading.Event(), threading.Event(), threading.Event()
        storage = Storage()
        seen = []
        class Engine:
            def run_sse(self, req, task_id):
                seen.append(trace.get())
                entered.set()
                yield 'queued'
                if not release.wait(5):
                    raise TimeoutError('fixture owner did not release')
                storage.update_task(task_id, status='success')
        class Runtime:
            active = 0
            @contextmanager
            def execution(self):
                self.active += 1
                try:
                    yield
                finally:
                    self.active -= 1
                    ended.set()
            def get(self):
                return Engine()
        runtime=Runtime(); tasks=Tasks(runtime,storage)
        token=trace.set('original-request')
        try:
            result=tasks.start(Request())
            self.assertEqual(result.status,'queued')
            self.assertTrue(entered.wait(5))
            self.assertEqual(runtime.active,1)
            self.assertFalse(ended.is_set())
        finally:
            release.set(); trace.reset(token)
        self.assertTrue(ended.wait(5))
        self.assertEqual(runtime.active,0)
        self.assertEqual(seen,['original-request'])

    def test_cancel_uses_existing_owner_engine_without_new_execution(self):
        from hub_runtime.tasks import Owner
        calls=[]
        class Runtime:
            def execution(self):
                raise AssertionError('cancel must not load model or begin execution')
        class Engine:
            def cancel_task(self, task_id):
                calls.append(task_id)
                return {'task_id':task_id,'status':'canceled','files':[]}
        storage=Storage(); task_id=storage.create_task('running',None)
        tasks=Tasks(Runtime(),storage)
        owner=Owner('api',task_id,Engine());tasks._add(owner)
        self.assertEqual(tasks.cancel_task(task_id).status,'canceled')
        self.assertEqual(calls,[task_id])
        self.assertFalse(owner.done.is_set(), 'cancel receipt prematurely ended owner')

    def test_native_failed_state_finishes_as_failure(self):
        storage=Storage(); states=[]
        class Engine:
            def run_sse(self, req, task_id):
                storage.update_task(task_id,status='failed')
                yield 'error'
        class Runtime:
            @contextmanager
            def execution(self):
                try:
                    yield
                except Exception:
                    states.append('failed');raise
                else:
                    states.append('succeeded')
            def get(self):return Engine()
        tasks=Tasks(Runtime(),storage)
        with self.assertRaises(RuntimeError):list(tasks.run_sse(Request()))
        self.assertEqual(states,['failed'])

    def test_native_cancel_result_is_not_counted_as_success(self):
        from ttd_model_runtime import NativeCancelled
        storage=Storage(); states=[]
        class Engine:
            def run_sse(self, req, task_id):
                storage.update_task(task_id,status='canceled')
                yield 'event: canceled\n\n'
        class Runtime:
            @contextmanager
            def execution(self):
                try:yield
                except NativeCancelled:
                    states.append('cancelled');raise
                else:states.append('succeeded')
            def get(self):return Engine()
        tasks=Tasks(Runtime(),storage)
        stream=tasks.run_sse(Request())
        self.assertEqual(list(stream),['event: canceled\n\n'])
        stream.thread.join(3)
        self.assertEqual(states,['cancelled'])
        self.assertEqual(storage.rows[stream.owner.task_id].status,'canceled')


if __name__=='__main__':unittest.main()
