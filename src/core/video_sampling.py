"""Sequential, frame-exact sampling without retrieving discarded pixels.

Do NOT seek between samples: inter-frame codecs may need preceding frames.
``grab`` still advances/decodes as required by the backend; only retrieval and
BGR materialisation are skipped. This is not hardware decoding or extra frame
skipping. A read-only adapter remains supported for plugins and tests.
"""
from __future__ import annotations
from time import perf_counter


class SelectiveVideoReader:
    def __init__(self, capture, *, enabled=True):
        self.capture = capture
        self.selective = bool(enabled and callable(getattr(capture, 'grab', None))
                              and callable(getattr(capture, 'retrieve', None)))
        self.advanced = 0
        self.retrieved = 0
        self.discarded = 0
        self.seconds = 0.0

    def read(self, selected=True):
        """Advance exactly one source frame; unselected frames return (True, None)."""
        started = perf_counter()
        try:
            if self.selective:
                ok = self.capture.grab()
                if not ok:
                    return False, None
                self.advanced += 1
                if not selected:
                    self.discarded += 1
                    return True, None
                ok, frame = self.capture.retrieve()
            else:
                ok, frame = self.capture.read()
                if not ok:
                    return False, None
                self.advanced += 1
                if not selected:
                    self.discarded += 1
                    return True, None
            if not ok or frame is None:
                return False, None
            self.retrieved += 1
            return True, frame
        finally:
            self.seconds += perf_counter() - started

    def snapshot(self):
        return {'mode': 'grab/retrieve' if self.selective else 'read',
                'advanced': self.advanced, 'retrieved': self.retrieved,
                'discarded': self.discarded, 'seconds': self.seconds}
