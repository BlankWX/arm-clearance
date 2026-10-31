"""Flushed, throttled main-thread diagnostics. No Blender dependency or threads."""
import re
import sys
import time


class ConsoleProgress:
    def __init__(self, scope='', stream=None, clock=time.perf_counter, interval=1.0):
        self.scope = scope
        self.stream = stream
        self.clock = clock
        self.interval = max(0., float(interval))
        self.finished = False
        self.outcome = ''
        self.last_message = ''
        self._started = False
        self._start_time = 0.
        self._last_time = float('-inf')
        self._phase = None
        self._completed = set()

    def _emit(self, tag, message):
        now = self.clock()
        elapsed = max(0., now - self._start_time)
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        scope = (' ' + self.scope) if self.scope else ''
        line = '[Arm Clearance%s][%02d:%02d:%02d][%s] %s\n' % (
            scope, hours, minutes, seconds, tag, message)
        stream = self.stream if self.stream is not None else sys.stdout
        try:
            try:
                stream.write(line)
            except UnicodeEncodeError:
                encoding = getattr(stream, 'encoding', None) or 'ascii'
                stream.write(line.encode(encoding, errors='backslashreplace').decode(encoding))
            stream.flush()
        except Exception:
            # A closed/redirected console must never discard a valid bake.
            pass
        self._last_time = now

    def start(self, message=None):
        if self._started or self.finished:
            return self
        self._started = True
        self._start_time = self.clock()
        self.last_message = str(message or '开始分析并烘焙')
        self._emit('start', self.last_message)
        return self

    @staticmethod
    def _key(message):
        solve = re.match(r'^(.*?第\d+/\d+组)\s+(?:第)?(\d+)/(\d+)步', message)
        if solve:
            return solve.group(1) + ':solve', (int(solve.group(2)), int(solve.group(3)))
        # Keep group/round identities while ignoring changing frame counters.
        identities = re.findall(r'第\d+/\d+[组轮]|复核轮次\s*\d+/\d+', message)
        normalized = re.sub(r'\d+(?:\.\d+)?', '#', message)
        without_groups = re.sub(r'第\d+/\d+[组轮]|复核轮次\s*\d+/\d+', '', message)
        counts = re.findall(r'(\d+)/(\d+)', without_groups)
        return (normalized, tuple(identities)), (tuple(map(int, counts[-1])) if counts else None)

    def update(self, message):
        if self.finished:
            return
        if not self._started:
            self.start()
        self.last_message = str(message)
        phase, count = self._key(self.last_message)
        completion = (phase, count)
        final = bool(count and count[1] > 0 and count[0] == count[1] and completion not in self._completed)
        if phase != self._phase or final or self.clock() - self._last_time >= self.interval:
            self._emit('progress', self.last_message)
            self._phase = phase
            if final:
                self._completed.add(completion)

    def finish(self, outcome, detail=''):
        if self.finished:
            return
        if not self._started:
            self.start()
        self.finished = True
        self.outcome = str(outcome)
        if detail:
            self.last_message = str(detail)
        labels = {'complete': '完成并通过复核', 'needs_review': '烘焙已保留，需要复核',
                  'cancelled': '已取消', 'error': '运行出错'}
        self._emit(self.outcome, labels.get(self.outcome, self.outcome) +
                   (('；' + str(detail)) if detail else ''))


def show_system_console(toggle, background=False):
    """Show a hidden Windows console without toggling a visible console off."""
    if sys.platform != 'win32' or background:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        get_window = ctypes.windll.kernel32.GetConsoleWindow
        get_window.restype = wintypes.HWND
        is_visible = ctypes.windll.user32.IsWindowVisible
        is_visible.argtypes = [wintypes.HWND]
        handle = get_window()
        if not handle or not is_visible(handle):
            toggle()
        return True
    except Exception:
        return False
