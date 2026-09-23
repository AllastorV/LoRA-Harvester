"""Coalesce fast worker progress without dropping completion/cancellation state.

One instance per worker. This is a notification filter, never a work scheduler:
all images are still processed. The most recent message is flushed on exit.
"""
from __future__ import annotations
from time import monotonic


class ProgressThrottle:
    def __init__(self, callback, interval=0.10, clock=monotonic):
        if interval < 0:
            raise ValueError('Progress interval must be non-negative.')
        self.callback, self.interval, self.clock = callback, interval, clock
        self._last_time = None
        self._last_total = None
        self._pending = None

    def __call__(self, current, total, message):
        self._pending = (current, total, message)
        now = self.clock()
        if (self._last_time is None or total != self._last_total
                or (total > 0 and current >= total)
                or now - self._last_time >= self.interval):
            self.flush(now)

    def flush(self, now=None):
        if self._pending is not None:
            value, self._pending = self._pending, None
            self._last_time = self.clock() if now is None else now
            self._last_total = value[1]
            self.callback(*value)
