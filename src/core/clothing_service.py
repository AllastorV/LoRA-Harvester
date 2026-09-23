"""Shared synchronous service used by background Qt jobs and pipeline post-passes."""
from __future__ import annotations
import json
from contextlib import ExitStack
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from .clothing_backend import ClothingCancelled, OllamaClothingBackend
from .clothing_cache import CachedVision, ClothingCache
from .clothing_captions import apply_caption, prepare_caption, undo_caption
from .clothing_io import FileLock, atomic_json, file_digest
from .clothing_matcher import ClothingMatcher, ClothingResult
from .clothing_profiles import ClothingProfileStore


class ClothingSession:
    def __init__(self, store=None, settings=None, cancel_event=None, log=None, backend=None):
        self.store = store or ClothingProfileStore()
        self.settings = settings or self.store.load_settings()
        self.cancel_event = cancel_event or threading.Event()
        self.log = log or (lambda text: None)
        self.backend = backend or OllamaClothingBackend(self.settings, self.cancel_event)
        self.cache = None
        self.lock = FileLock(self.store.root / 'inference.lock')
        self.matcher = None

    def __enter__(self):
        self.lock.__enter__()
        try:
            self.log('Checking local vision model...')
            self.backend.check()
            if self.settings.use_cache:
                self.cache = ClothingCache(self.store.root / 'cache.sqlite3')
            self.matcher = ClothingMatcher(self.store, self.settings,
                                            CachedVision(self.backend, self.cache), self.log)
            return self
        except Exception:
            self.lock.__exit__()
            raise

    def __exit__(self, *args):
        try:
            if self.cache:
                self.cache.close()
            if self.settings.unload_after_job:
                try:
                    # A cancelled HTTP request must not keep the model resident for five minutes.
                    if self.cancel_event.is_set() and isinstance(self.backend, OllamaClothingBackend):
                        OllamaClothingBackend(self.settings).unload()
                    else:
                        self.backend.unload()
                except Exception as exc:
                    self.log(f'Model unload was not confirmed: {exc}')
        finally:
            self.lock.__exit__()


def load_selection(store, image):
    path = store.root / 'selections.json'
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding='utf-8'))
    item = records.get(str(Path(image).resolve()))
    if item and item.get('digest') == file_digest(Path(image)):
        return item.get('bbox')
    return None


def save_selection(store, image, bbox):
    from .clothing_io import validate_box
    validate_box(bbox)
    key = str(Path(image).resolve())
    path = store.root / 'selections.json'
    with FileLock(store.root / 'selection.lock'):
        records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if bbox is None:
            records.pop(key, None)
        else:
            records[key] = {'digest': file_digest(Path(image)), 'bbox': list(bbox)}
        atomic_json(path, records)


def record_job(store, journals):
    if not journals:
        return
    path = store.root / 'undo_jobs.json'
    with FileLock(store.root / 'history.lock'):
        jobs = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        jobs.append({'created': time.time(), 'journals': list(journals)})
        atomic_json(path, jobs)


def undo_last_job(store=None, log=None):
    store = store or ClothingProfileStore()
    path = store.root / 'undo_jobs.json'
    if not path.exists():
        return {'undone': 0, 'errors': []}
    errors, undone = [], 0
    with FileLock(store.root / 'history.lock'):
        jobs = json.loads(path.read_text(encoding='utf-8'))
        if not jobs:
            return {'undone': 0, 'errors': []}
        job = jobs[-1]
        remaining = []
        for journal in reversed(job['journals']):
            try:
                undo_caption(journal)
                undone += 1
            except Exception as exc:
                remaining.append(journal)
                errors.append(str(exc))
        if remaining:
            job['journals'] = list(reversed(remaining))
        else:
            jobs.pop()
        atomic_json(path, jobs)
    return {'undone': undone, 'errors': errors}


def run_clothing_batch(paths, *, store=None, settings=None, apply=False,
                       selections=None, cancel_event=None, log=None, on_result=None,
                       on_progress=None, backend=None, sequential=None):
    """Analyze exact final saved image paths. Default is preview-only, never write.

    Multi-image jobs process the outfit library in display order and retain
    accepted matches between passes. A single-image analysis still compares
    all outfits so the user can inspect competing candidates.
    """
    store = store or ClothingProfileStore()
    settings = settings or store.load_settings()
    cancel_event = cancel_event or threading.Event()
    paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
    if sequential is None:
        sequential = len(paths) > 1
    if sequential and paths:
        from .clothing_sequential import run_sequential_batch
        return run_sequential_batch(
            paths, store=store, settings=settings, apply=apply,
            selections=selections, cancel_event=cancel_event,
            log=log or (lambda text: None), on_result=on_result,
            on_progress=on_progress, backend=backend)
    profiles = [p for p in store.load() if p.enabled]
    stats = {'total': len(paths), 'matched': 0, 'review': 0, 'no_match': 0,
             'errors': 0, 'written': 0, 'cancelled': False}
    if not paths:
        return stats
    if not profiles:
        raise ValueError('Create and enable at least one outfit profile.')
    journals = []
    try:
        from .clothing_feedback import ClothingFeedbackStore
        feedback = ClothingFeedbackStore(store)
        with ExitStack() as stack:
            session = None
            for i, path in enumerate(paths):
                if cancel_event.is_set():
                    raise ClothingCancelled('Clothing operation cancelled.')
                preview = None
                try:
                    bbox = (selections or {}).get(path)
                    if bbox is None:
                        bbox = load_selection(store, path)
                    result = feedback.exact(Path(path), bbox, profiles)
                    if result is None:
                        if session is None:
                            session = stack.enter_context(ClothingSession(store, settings, cancel_event, log, backend))
                        result = session.matcher.analyze(Path(path), profiles, bbox)
                    if result.status == 'matched':
                        preview = prepare_caption(result)
                        if apply:
                            if cancel_event.is_set():
                                raise ClothingCancelled('Clothing operation cancelled.')
                            journal = apply_caption(preview, store)
                            if journal:
                                journals.append(journal)
                                stats['written'] += 1
                    stats[result.status] += 1
                except ClothingCancelled:
                    raise
                except Exception as exc:
                    stats['errors'] += 1
                    preview = None
                    result = ClothingResult(path, 'error', str(exc))
                if on_result:
                    on_result(result, preview)
                if on_progress:
                    on_progress(i + 1, len(paths), result.reason)
    except ClothingCancelled:
        stats['cancelled'] = True
    finally:
        record_job(store, journals)
    return stats


def optional_pipeline_pass(paths, stage, *, log=None, stop=None, store=None):
    """Opt-in pipeline hook. Errors here never destroy already extracted/captioned data."""
    store = store or ClothingProfileStore()
    log = log or (lambda text: None)
    settings = store.load_settings()
    enabled = settings.auto_after_video if stage == 'video' else settings.auto_after_caption
    if not enabled:
        return None
    if stop and stop():
        return {'cancelled': True}
    paths = list(paths)
    if not paths:
        return None
    log(f'Clothing post-pass: {len(paths)} final images. Only accepted matches will be written.')
    class StopEvent(threading.Event):
        def is_set(self):
            return super().is_set() or bool(stop and stop())
    event = StopEvent()
    def progress(i, total, message):
        if stop and stop():
            event.set()
        log(f'Clothing {i}/{total}: {message}')
    return run_clothing_batch(paths, store=store, settings=settings, apply=True,
                              cancel_event=event, log=log, on_progress=progress)
