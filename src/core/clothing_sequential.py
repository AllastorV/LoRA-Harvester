"""Ordered outfit passes with resumable per-image, per-profile decisions."""
from __future__ import annotations

from pathlib import Path

from .clothing_backend import ClothingCancelled
from .clothing_captions import apply_caption, prepare_caption
from .clothing_feedback import ClothingFeedbackStore
from .clothing_io import digest_json, file_digest
from .clothing_matcher import ClothingResult
from .clothing_cache import PROMPT_VERSION


_CACHE_VERSION = 'ordered-outfit-pass-v2'


def run_sequential_batch(paths, *, store, settings, apply, selections, cancel_event,
                         log, on_result, on_progress, backend):
    """Finish each library outfit over the remaining images before the next outfit.

    An accepted image leaves the queue immediately. Decisions are cached per
    image, profile, model and settings, so a cancelled run resumes without
    repeating completed comparisons. Uncertain decisions never auto-tag.
    """
    # Imported here because the public service also imports this module.
    from .clothing_service import ClothingSession, load_selection, record_job

    profiles = [profile for profile in store.load() if profile.enabled]
    if not profiles:
        raise ValueError('Create and enable at least one outfit profile.')
    stats = {'total': len(paths), 'matched': 0, 'review': 0, 'no_match': 0,
             'errors': 0, 'written': 0, 'cancelled': False,
             'comparisons': 0, 'cache_hits': 0}
    if not paths:
        return stats

    feedback = ClothingFeedbackStore(store)
    feedback_digest = feedback.fingerprint()
    library_digest = store.library_fingerprint(profiles)
    relevant_settings = {name: getattr(settings, name) for name in (
        'model', 'identity_threshold', 'part_threshold', 'person_mode',
        'max_side', 'num_ctx')}
    remaining = dict.fromkeys(paths)
    uncertain = {}
    failed = {}
    digests = {}
    auto_boxes = {}
    journals = []

    def finish(path, result):
        preview = None
        try:
            if result.status == 'matched':
                result.library_digest = library_digest
                preview = prepare_caption(result)
                if apply:
                    if cancel_event.is_set():
                        raise ClothingCancelled('Clothing operation cancelled.')
                    journal = apply_caption(preview, store)
                    if journal:
                        journals.append(journal)
                        stats['written'] += 1
            stats['errors' if result.status == 'error' else result.status] += 1
        except ClothingCancelled:
            raise
        except Exception as exc:
            result = ClothingResult(path, 'error', str(exc))
            preview = None
            stats['errors'] += 1
        remaining.pop(path, None)
        if on_result:
            on_result(result, preview)

    try:
        with ClothingSession(store, settings, cancel_event, log, backend) as session:
            for path in paths:
                if cancel_event.is_set():
                    raise ClothingCancelled('Clothing operation cancelled.')
                try:
                    bbox = (selections or {}).get(path)
                    if bbox is None:
                        bbox = load_selection(store, path)
                    exact = feedback.exact(Path(path), bbox, profiles)
                    if exact is not None:
                        finish(path, exact)
                except ClothingCancelled:
                    raise
                except Exception as exc:
                    finish(path, ClothingResult(path, 'error', str(exc)))

            for profile_index, profile in enumerate(profiles):
                stage_paths = list(remaining)
                if not stage_paths:
                    break
                profile_digest = store.fingerprint(profile)
                log(f'{profile.name} ({profile_index + 1}/{len(profiles)}): '
                    f'{len(stage_paths)} unmatched images')
                for index, path in enumerate(stage_paths):
                    if cancel_event.is_set():
                        raise ClothingCancelled('Clothing operation cancelled.')
                    try:
                        bbox = (selections or {}).get(path)
                        if bbox is None:
                            bbox = load_selection(store, path)
                        if path not in digests:
                            digests[path] = file_digest(Path(path))
                        if bbox is None and path in auto_boxes:
                            if file_digest(Path(path)) != digests[path]:
                                raise ValueError('Image changed during analysis; analyze it again.')
                            bbox = auto_boxes[path]
                        key = digest_json({
                            'kind': _CACHE_VERSION, 'prompt': PROMPT_VERSION,
                            'model': session.backend.fingerprint,
                            'settings': relevant_settings,
                            'profile': profile_digest,
                            'image': digests[path], 'bbox': list(bbox) if bbox else None,
                            'feedback': feedback_digest,
                        })
                        result = None
                        if session.cache is not None:
                            value = session.cache.get(key)
                            if isinstance(value, dict):
                                try:
                                    cached = ClothingResult(**value)
                                    if (cached.image == path
                                            and cached.image_digest == digests[path]
                                            and cached.status in ('matched', 'review', 'no_match')
                                            and (cached.status != 'matched'
                                                 or cached.profile_id == profile.id)):
                                        result = cached
                                        stats['cache_hits'] += 1
                                except (TypeError, ValueError):
                                    pass
                        if result is None:
                            result = session.matcher.analyze(Path(path), [profile], bbox)
                            if (session.cache is not None and result.decision_source == 'model'
                                    and result.status in ('matched', 'review', 'no_match')):
                                session.cache.put(key, result.to_dict())
                        if result.bbox:
                            auto_boxes[path] = result.bbox
                        result.library_digest = library_digest
                        stats['comparisons'] += 1
                        if result.status == 'matched':
                            if path in uncertain or path in failed:
                                reason = ('An earlier outfit comparison failed; review before tagging.'
                                          if path in failed else
                                          'An earlier outfit was also visually possible; review the color/design before tagging.')
                                finish(path, ClothingResult(
                                    path, 'review', reason, image_digest=digests[path],
                                    bbox=result.bbox, candidates=(uncertain[path].candidates
                                    if path in uncertain else []) + result.candidates))
                            else:
                                finish(path, result)
                        elif result.status == 'review':
                            uncertain.setdefault(path, result)
                        elif result.status == 'error':
                            failed.setdefault(path, result.reason)
                    except ClothingCancelled:
                        raise
                    except Exception as exc:
                        failed.setdefault(path, str(exc))
                    if on_progress:
                        on_progress(index + 1, len(stage_paths),
                                    f'{profile.name} ({profile_index + 1}/{len(profiles)}): '
                                    f'{index + 1}/{len(stage_paths)}; {len(remaining)} remaining')

            for path in list(remaining):
                if cancel_event.is_set():
                    raise ClothingCancelled('Clothing operation cancelled.')
                finish(path, uncertain.get(path) or (
                    ClothingResult(path, 'error', failed[path]) if path in failed else
                    ClothingResult(path, 'no_match',
                                   'No registered outfit matched the visible clothing.',
                                   image_digest=digests.get(path, ''))))
    except ClothingCancelled:
        stats['cancelled'] = True
    finally:
        record_job(store, journals)
    return stats
