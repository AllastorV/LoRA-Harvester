"""Qt thread adapter. Only signals cross into the GUI thread."""
from __future__ import annotations
import threading
from PyQt5.QtCore import QThread, pyqtSignal
from src.core.clothing_backend import ClothingCancelled, OllamaClothingBackend
from src.core.clothing_captions import apply_caption
from src.core.clothing_service import (ClothingSession, run_clothing_batch,
                                       record_job, undo_last_job)


class ClothingTaggerWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_message = pyqtSignal(str)
    image_result = pyqtSignal(object, object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation, store, settings, *, paths=None, profile=None,
                 previews=None, feedback=None, parent=None):
        super().__init__(parent)
        self.operation, self.store, self.settings = operation, store, settings
        self.paths = list(paths or [])
        self.profile = profile
        self.previews = list(previews or [])
        self.feedback = dict(feedback or {})
        self.cancel_event = threading.Event()
        self.backend = OllamaClothingBackend(settings, self.cancel_event)

    def stop(self):
        self.requestInterruption()
        self.backend.cancel()

    def run(self):
        try:
            if self.operation == 'analyze':
                result = run_clothing_batch(
                    self.paths, store=self.store, settings=self.settings,
                    cancel_event=self.cancel_event, log=self.log_message.emit,
                    on_result=self.image_result.emit, on_progress=self.progress.emit,
                    backend=self.backend)
            elif self.operation == 'feedback':
                from src.core.clothing_feedback import ClothingFeedbackStore
                from src.core.clothing_captions import prepare_caption
                from src.core.clothing_service import save_selection
                data = dict(self.feedback)
                image, bbox = data.pop('image'), data.pop('bbox')
                corrected = ClothingFeedbackStore(self.store).remember(image, bbox, **data)
                save_selection(self.store, image, bbox)
                preview = prepare_caption(corrected) if corrected.status == 'matched' else None
                self.image_result.emit(corrected, preview)
                result = {'saved_correction': True, 'caption_written': False}
            elif self.operation == 'forget_feedback':
                from src.core.clothing_feedback import ClothingFeedbackStore
                count = ClothingFeedbackStore(self.store).forget(self.feedback['image'], self.feedback['bbox'])
                result = {'forgotten': count, 'caption_written': False}
            elif self.operation == 'references':
                with ClothingSession(self.store, self.settings, self.cancel_event,
                                     self.log_message.emit, self.backend) as session:
                    result = session.matcher.analyze_references(self.profile)
            elif self.operation == 'check':
                result = self.backend.check()
            elif self.operation == 'pull':
                def report(data):
                    total = data.get('total', 0)
                    completed = data.get('completed', 0)
                    percent = int(completed * 100 / total) if total else 0
                    self.progress.emit(percent, 100, str(data.get('status', 'Downloading...')))
                result = self.backend.pull(report)
            elif self.operation == 'apply':
                journals, errors, unchanged = [], [], 0
                try:
                    for i, preview in enumerate(self.previews):
                        self.backend.check_cancelled()
                        try:
                            journal = apply_caption(preview, self.store)
                            if journal:
                                journals.append(journal)
                            else:
                                unchanged += 1
                        except Exception as exc:
                            errors.append(f'{preview.image}: {exc}')
                        self.progress.emit(i + 1, len(self.previews), preview.image)
                finally:
                    record_job(self.store, journals)
                result = {'written': len(journals), 'unchanged': unchanged, 'errors': errors}
            elif self.operation == 'undo':
                result = undo_last_job(self.store)
            else:
                raise ValueError('Unknown clothing operation.')
            self.completed.emit(result)
        except ClothingCancelled:
            self.completed.emit({'cancelled': True})
        except Exception as exc:
            self.failed.emit(str(exc))
