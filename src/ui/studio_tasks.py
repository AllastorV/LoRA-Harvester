"""Small cooperative Qt task adapter shared by dataset and comparison panels."""
from __future__ import annotations
import threading
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QMessageBox
from . import theme


class StudioWorker(QThread):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self.work = work
        self.cancel = threading.Event()

    def stop(self):
        self.cancel.set()
        self.requestInterruption()

    def run(self):
        from src.core.progress_throttle import ProgressThrottle
        progress = ProgressThrottle(self.progress.emit)
        try:
            result = self.work(self.cancel, progress)
            progress.flush()
            self.completed.emit(result)
        except Exception as exc:
            progress.flush()
            self.failed.emit(str(exc))


class StudioTaskWidget(QWidget):
    busy_changed = pyqtSignal(bool)

    def __init__(self, lang='en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._thread = None
        self._completion = None
        self._controls = []
        self._translations = []

    def text(self, tr, en):
        return tr if self.lang == 'tr' else en

    def label_text(self, widget, tr, en):
        self._translations.append((widget.setText, tr, en))
        widget.setText(self.text(tr, en))
        return widget

    def tip_text(self, widget, tr, en):
        self._translations.append((widget.setToolTip, tr, en))
        widget.setToolTip(self.text(tr, en))
        return widget

    def start_task(self, work, completion):
        if self._thread is not None or getattr(self.window(), '_close_pending', False):
            return False
        if any(t.isRunning() for t in self.window().findChildren(QThread)):
            self.error(self.text('Önce diğer çalışan işlemi bitir veya durdur.',
                                 'Finish or stop the other running task first.'))
            return False
        self._completion = completion
        thread = StudioWorker(work, self)
        self._thread = thread
        thread.progress.connect(self._progress)
        thread.failed.connect(self.error)
        thread.completed.connect(self._completed)
        thread.finished.connect(self._finished)
        for control in self._controls:
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.busy_changed.emit(True)
        observer = getattr(self, '_ai_before_worker_start', None)
        if observer is not None:
            observer(thread)
        thread.start()
        return True

    def _progress(self, current, total, message):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        self.status.setText(message)

    def _completed(self, result):
        # Slots must not leak Python exceptions into Qt's event loop.
        try:
            if self._completion:
                self._completion(result)
        except Exception as exc:
            self.error(str(exc))

    def _finished(self):
        thread, self._thread = self._thread, None
        self._completion = None
        for control in self._controls:
            control.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1); self.progress.setValue(1)
        self.busy_changed.emit(False)
        if thread is not None:
            thread.deleteLater()

    def is_busy(self):
        return self._thread is not None

    def stop(self):
        if self._thread:
            self._thread.stop()
            self.status.setText(self.text('Durduruluyor; geçerli güvenli adım tamamlanacak.',
                                          'Stopping after the current safe step.'))

    def error(self, message):
        self.status.setText(str(message))
        QMessageBox.warning(self, 'LoRA-Harvester', str(message))

    def update_language(self, lang):
        self.lang = lang
        for setter, tr, en in self._translations:
            setter(self.text(tr, en))

    def refresh_styles(self):
        return theme.refresh_styles(self)
