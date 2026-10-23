"""Run real bake-operator methods against fake contexts in background Blender.

No RNA operator instance, GUI operation, real modal loop, or solver is used.
"""
import io
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arm_clearance as package
from arm_clearance.progress import ConsoleProgress


class WindowManager:
    def __init__(self, failure=None):
        self.failure = failure
        self.timer = object()
        self.added, self.removed, self.handlers = [], [], []
    def event_timer_add(self, interval, window=None):
        if self.failure == 'timer':
            raise RuntimeError('injected timer setup failure')
        self.added.append(self.timer)
        return self.timer
    def event_timer_remove(self, timer):
        self.removed.append(timer)
    def modal_handler_add(self, operator):
        if self.failure == 'handler':
            raise RuntimeError('injected handler setup failure')
        self.handlers.append(operator)


class ModalProgressTests(unittest.TestCase):
    def setUp(self):
        self.original = {name:getattr(package, name) for name in
                         ('ConsoleProgress', 'run_job', 'show_system_console')}
        self.stream = io.StringIO()
        self.loggers, self.advances, self.reports, self.redraws = [], [], [], []
        def logger_factory():
            logger = ConsoleProgress(scope='modal-test', stream=self.stream)
            self.loggers.append(logger)
            return logger
        package.ConsoleProgress = logger_factory
        package.show_system_console = lambda *args, **kwargs: False
        self.settings = SimpleNamespace(busy=False, status='选择骨架和网格', console_auto_open=False)
        self.wm = WindowManager()
        self.context = SimpleNamespace(
            scene=SimpleNamespace(arm_clearance=self.settings, frame_start=321, frame_end=494),
            window_manager=self.wm, window=object(),
            screen=SimpleNamespace(is_animation_playing=False,
                                   areas=[SimpleNamespace(tag_redraw=lambda:self.redraws.append(True))]))
        self.operator = SimpleNamespace(_timer=None, _job=None, _console=None,
            report=lambda levels,text:self.reports.append((set(levels),text)))
        self.operator.cleanup = MethodType(package.AC_OT_bake.cleanup, self.operator)
        def pending(scene, settings, status, *, console=None):
            self.advances.append('advanced')
            try:
                status('主优化 第1/4组 1/150步')
                console.update(settings.status)
                yield None
            finally:
                settings.status = '已取消，恢复先前动作'
                console.finish('cancelled', settings.status)
        package.run_job = pending

    def tearDown(self):
        if self.operator._job is not None:
            try:
                self.operator.cleanup(self.context)
            except Exception:
                pass
        for name,value in self.original.items():
            setattr(package, name, value)

    def execute(self):
        return package.AC_OT_bake.execute(self.operator, self.context)

    def assert_clean(self, expected_removed=1):
        self.assertIsNone(self.operator._job)
        self.assertIsNone(self.operator._timer)
        self.assertFalse(self.settings.busy)
        self.assertEqual(len(self.wm.removed), expected_removed)
        self.assertTrue(self.operator._console.finished)

    def test_escape_before_first_yield_updates_cancelled_status_and_cleans_timer(self):
        self.assertEqual(self.execute(), {'RUNNING_MODAL'})
        self.assertTrue(self.settings.busy)
        self.assertEqual(self.advances, [])
        result = package.AC_OT_bake.modal(self.operator, self.context, SimpleNamespace(type='ESC'))
        self.assertEqual(result, {'CANCELLED'})
        self.assertEqual(self.advances, [])
        self.assert_clean()
        self.assertEqual(self.operator._console.outcome, 'cancelled')
        self.assertIn('取消', self.settings.status)
        self.assertEqual(self.stream.getvalue().count('[cancelled]'), 1)

    def test_setup_failures_finish_error_and_remove_any_created_timer(self):
        for failure in ('timer', 'handler'):
            with self.subTest(failure=failure):
                self.wm = WindowManager(failure)
                self.context.window_manager = self.wm
                self.operator._timer = self.operator._job = self.operator._console = None
                self.settings.status, self.settings.busy = '待运行', False
                self.assertEqual(self.execute(), {'CANCELLED'})
                self.assert_clean(0 if failure == 'timer' else 1)
                self.assertEqual(self.operator._console.outcome, 'error')
                self.assertIn('setup failure', self.settings.status)
                self.assertEqual(self.reports[-1][0], {'ERROR'})
                self.assertEqual(self.advances, [])

    def test_close_exception_still_clears_busy_and_timer(self):
        self.assertEqual(self.execute(), {'RUNNING_MODAL'})
        def fail_close():
            raise RuntimeError('injected close failure')
        self.operator._job = SimpleNamespace(close=fail_close)
        with self.assertRaisesRegex(RuntimeError, 'injected close failure'):
            self.operator.cleanup(self.context)
        self.assert_clean()
        self.assertEqual(self.operator._console.outcome, 'error')
        self.assertEqual(self.stream.getvalue().count('[error]'), 1)
        self.assertNotIn('[cancelled]', self.stream.getvalue())

    def test_cleanup_preserves_existing_complete_or_review_outcome_and_status(self):
        for outcome in ('complete', 'needs_review'):
            with self.subTest(outcome=outcome):
                self.wm = WindowManager()
                self.context.window_manager = self.wm
                self.operator._timer = self.operator._job = self.operator._console = None
                self.assertEqual(self.execute(), {'RUNNING_MODAL'})
                self.settings.status = '终态保留 ' + outcome
                self.operator._console.finish(outcome, self.settings.status)
                before = self.stream.getvalue()
                self.operator.cleanup(self.context)
                self.assert_clean()
                self.assertEqual(self.operator._console.outcome, outcome)
                self.assertEqual(self.settings.status, '终态保留 ' + outcome)
                self.assertEqual(self.stream.getvalue(), before)

    def test_normal_modal_completion_preserves_outcome_status_and_report_level(self):
        for validation, outcome, level in (('passed','complete','INFO'),
                                           ('needs_review','needs_review','WARNING')):
            with self.subTest(outcome=outcome):
                self.wm = WindowManager()
                self.context.window_manager = self.wm
                self.operator._timer = self.operator._job = self.operator._console = None
                def completed(scene, settings, status, *, console=None):
                    yield None
                    settings.status = '最终状态 ' + outcome
                    console.finish(outcome, settings.status)
                    return {'arm_clearance_validation_status': validation}
                package.run_job = completed
                self.assertEqual(self.execute(), {'RUNNING_MODAL'})
                self.assertEqual(package.AC_OT_bake.modal(self.operator, self.context,
                                 SimpleNamespace(type='TIMER')), {'FINISHED'})
                self.assert_clean()
                self.assertEqual(self.operator._console.outcome, outcome)
                self.assertEqual(self.settings.status, '最终状态 ' + outcome)
                self.assertEqual(self.reports[-1], ({level}, '最终状态 ' + outcome))

    def test_modal_exception_preserves_error_final_state(self):
        def failed(scene, settings, status, *, console=None):
            yield None
            settings.status = '处理失败，已恢复先前动作：injected solver failure'
            console.finish('error', settings.status)
            raise RuntimeError('injected solver failure')
        package.run_job = failed
        self.assertEqual(self.execute(), {'RUNNING_MODAL'})
        self.assertEqual(package.AC_OT_bake.modal(self.operator, self.context,
                         SimpleNamespace(type='TIMER')), {'CANCELLED'})
        self.assert_clean()
        self.assertEqual(self.operator._console.outcome, 'error')
        self.assertIn('injected solver failure', self.settings.status)
        self.assertEqual(self.reports[-1][0], {'ERROR'})
        self.assertEqual(self.stream.getvalue().count('[error]'), 1)

    def test_external_cancel_and_repeated_cleanup_are_idempotent(self):
        self.assertEqual(self.execute(), {'RUNNING_MODAL'})
        package.AC_OT_bake.cancel(self.operator, self.context)
        self.assert_clean()
        self.assertEqual(self.operator._console.outcome, 'cancelled')
        status, output = self.settings.status, self.stream.getvalue()
        self.operator.cleanup(self.context)
        self.assert_clean()
        self.assertEqual(self.settings.status, status)
        self.assertEqual(self.stream.getvalue(), output)


if __name__ == '__main__':
    unittest.main(argv=[__file__], verbosity=2)
