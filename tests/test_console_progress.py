"""Pure Python regression tests for main-thread console progress output.

Run with Python, or Blender --background --factory-startup --python this_file.
The progress module is loaded directly without importing the Blender addon.
"""
import importlib.util
import io
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('arm_clearance_console_test', ROOT / 'arm_clearance/progress.py')
PROGRESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROGRESS
SPEC.loader.exec_module(PROGRESS)
ConsoleProgress = PROGRESS.ConsoleProgress


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


class FlushSpy(io.StringIO):
    def __init__(self):
        super().__init__()
        self.events = []
    def write(self, value):
        self.events.append(('write', value))
        return super().write(value)
    def flush(self):
        self.events.append(('flush', None))
        return super().flush()


class BrokenStream:
    def __init__(self, operation, exception):
        self.operation, self.exception = operation, exception
        self.writes, self.flushes = 0, 0
    def write(self, value):
        self.writes += 1
        if self.operation == 'write':
            raise self.exception()
        return len(value)
    def flush(self):
        self.flushes += 1
        if self.operation == 'flush':
            raise self.exception()


def unicode_error():
    return UnicodeEncodeError('ascii', '手臂', 0, 2, 'unrepresentable test output')


class ConsoleProgressTests(unittest.TestCase):
    def make(self, interval=1.):
        clock, stream = Clock(), FlushSpy()
        logger = ConsoleProgress(scope='测试角色', stream=stream, clock=clock, interval=interval)
        return logger, stream, clock

    def assert_flushed(self, stream):
        self.assertTrue(stream.events)
        self.assertEqual(stream.events[-1][0], 'flush')

    def test_start_is_idempotent_and_flushes_immediately(self):
        logger, stream, clock = self.make()
        logger.start('开始分析')
        first = stream.getvalue()
        self.assertIn('开始分析', first)
        self.assert_flushed(stream)
        logger.start('不能重复启动')
        self.assertEqual(stream.getvalue(), first)
        self.assertFalse(logger.finished)

    def test_stage_group_and_repair_round_changes_bypass_throttle(self):
        logger, stream, clock = self.make()
        logger.start()
        messages = ('读取原动作，提取身体、双臂与完整手掌手指',
                    '主优化 第1/4组 1/150步',
                    '主优化 第2/4组 1/150步',
                    '追加修复 第1/3轮 第1/1组 1/150步',
                    '追加修复 第2/3轮 第1/1组 1/150步',
                    '全分辨率臂手复核 321/494（1/347）')
        for message in messages:
            with self.subTest(message=message):
                before = stream.getvalue()
                clock.advance(.01)
                logger.update(message)
                self.assertNotEqual(stream.getvalue(), before)
                self.assertIn(message, stream.getvalue())
                self.assertEqual(logger.last_message, message)
                self.assert_flushed(stream)
        positions = [stream.getvalue().index(message) for message in messages]
        self.assertEqual(positions, sorted(positions))
        logger.finish('complete')

    def test_regular_iterations_are_throttled_but_latest_message_is_retained(self):
        logger, stream, clock = self.make()
        logger.start()
        logger.update('主优化 第1/4组 1/150步')
        before = stream.getvalue()
        clock.advance(.1)
        logger.update('主优化 第1/4组 2/150步')
        self.assertEqual(stream.getvalue(), before)
        self.assertEqual(logger.last_message, '主优化 第1/4组 2/150步')
        clock.advance(1.)
        logger.update('主优化 第1/4组 3/150步')
        self.assertIn('主优化 第1/4组 3/150步', stream.getvalue())
        self.assert_flushed(stream)
        logger.finish('complete')

    def test_direction_and_line_search_share_the_iteration_throttle(self):
        logger, stream, clock = self.make()
        logger.start()
        logger.update('主优化 第1/4组 1/150步 计算修正方向')
        before = stream.getvalue()
        clock.advance(.1)
        logger.update('主优化 第1/4组 1/150步 线搜索 1/16')
        self.assertEqual(stream.getvalue(), before)
        clock.advance(.1)
        logger.update('主优化 第1/4组 1/150步 线搜索 2/16')
        self.assertEqual(stream.getvalue(), before)
        self.assertEqual(logger.last_message, '主优化 第1/4组 1/150步 线搜索 2/16')
        clock.advance(1.)
        logger.update('主优化 第1/4组 1/150步 线搜索 3/16')
        self.assertIn('线搜索 3/16', stream.getvalue())
        self.assert_flushed(stream)
        logger.finish('complete')

    def test_final_iteration_is_printed_without_waiting_for_interval(self):
        logger, stream, clock = self.make()
        logger.start()
        logger.update('主优化 第1/4组 149/150步')
        before = stream.getvalue()
        clock.advance(.01)
        logger.update('主优化 第1/4组 150/150步')
        self.assertNotEqual(stream.getvalue(), before)
        self.assertIn('主优化 第1/4组 150/150步', stream.getvalue())
        self.assert_flushed(stream)
        first_final = stream.getvalue()
        logger.update('主优化 第1/4组 150/150步 线搜索 1/16')
        self.assertEqual(stream.getvalue(), first_final)
        logger.finish('complete')

    def test_each_terminal_outcome_is_emitted_once_and_finish_is_idempotent(self):
        for outcome in ('complete', 'needs_review', 'cancelled', 'error'):
            with self.subTest(outcome=outcome):
                logger, stream, clock = self.make()
                logger.start()
                logger.update('正在分析')
                before = stream.getvalue()
                detail = '终态说明 ' + outcome
                logger.finish(outcome, detail)
                final = stream.getvalue()
                self.assertTrue(logger.finished)
                self.assertEqual(logger.outcome, outcome)
                self.assertNotEqual(final, before)
                self.assertEqual(final.count(detail), 1)
                self.assertEqual(final.count('[' + outcome + ']'), 1)
                self.assert_flushed(stream)
                logger.finish('error', '不得覆盖第一次终态')
                logger.finish(outcome, detail)
                logger.update('结束后不能继续输出')
                self.assertEqual(stream.getvalue(), final)
                self.assertEqual(logger.outcome, outcome)

    def test_write_and_flush_errors_do_not_escape_or_prevent_final_state(self):
        for operation in ('write', 'flush'):
            for exception in (lambda: OSError('closed test stream'), unicode_error):
                with self.subTest(operation=operation, exception=exception().__class__.__name__):
                    stream, clock = BrokenStream(operation, exception), Clock()
                    logger = ConsoleProgress(scope='容错测试', stream=stream, clock=clock, interval=1.)
                    logger.start('开始')
                    logger.update('主优化 第1/4组 1/150步')
                    clock.advance(2.)
                    logger.update('主优化 第1/4组 2/150步')
                    logger.finish('needs_review', '烘焙已保留')
                    logger.finish('error', '不可覆盖')
                    self.assertTrue(logger.finished)
                    self.assertEqual(logger.outcome, 'needs_review')
                    self.assertGreater(stream.writes, 0)
                    if operation == 'flush':
                        self.assertGreater(stream.flushes, 0)

    def test_logging_does_not_start_background_threads(self):
        before = {thread.ident for thread in threading.enumerate()}
        logger, stream, clock = self.make()
        logger.start('开始')
        for index in range(1, 4):
            clock.advance(1.)
            logger.update('主优化 第1/4组 %d/150步' % index)
        logger.finish('cancelled', '用户取消')
        after = {thread.ident for thread in threading.enumerate()}
        self.assertEqual(after, before)


if __name__ == '__main__':
    unittest.main(argv=[__file__], verbosity=2)
