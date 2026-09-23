"""Saved-caption synchronization without model inference or implicit dataset writes.

The editor keeps a byte snapshot separately from its draft. Writes use optimistic
concurrency and the same lock as Clothing. A caption-only change can rebase an
accepted outfit decision, but cannot make a stale visual decision valid again.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .clothing_captions import CaptionConflict, _caption_paths, prepare_caption
from .clothing_io import FileLock, atomic_bytes, digest_json


def path_key(path: str | Path) -> str:
    """Compare full paths, not basenames (also respects Windows case folding)."""
    return os.path.normcase(str(Path(path).resolve()))


def editor_text(raw: bytes | None) -> str:
    """QTextEdit uses LF. Keep the original BOM/newline bytes in the snapshot."""
    return (raw or b'').decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')


@dataclass(frozen=True)
class CaptionSnapshot:
    image: str
    raw: bytes | None

    @property
    def text(self) -> str:
        return editor_text(self.raw)

    @property
    def path(self) -> Path:
        return Path(self.image).with_suffix('.txt')


def read_caption_snapshot(image: str | Path) -> CaptionSnapshot:
    image = Path(image).resolve()
    if not image.is_file():
        raise CaptionConflict('Source image is missing. Reload the image list.')
    caption = image.with_suffix('.txt')
    if caption.is_symlink():
        raise CaptionConflict('Caption is a symbolic link; refusing to edit it.')
    try:
        raw = caption.read_bytes()
    except FileNotFoundError:
        raw = None
    result = CaptionSnapshot(str(image), raw)
    result.text  # Reject invalid UTF-8 now, not after the draft has replaced it.
    return result


def save_edited_caption(snapshot: CaptionSnapshot, text: str) -> CaptionSnapshot:
    """Save a user draft only against its exact disk baseline; never hide failures."""
    if not isinstance(text, str):
        raise TypeError('Caption text must be a string.')
    image, caption, meta, _ = _caption_paths(snapshot.image)
    with FileLock(meta / 'write.lock'):
        current = read_caption_snapshot(image)
        if current.raw != snapshot.raw:
            raise CaptionConflict('Caption changed on disk. Your draft was kept; reload/revert before saving.')
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        if snapshot.raw is not None and normalized == snapshot.text:
            return snapshot  # Preserve byte-identical clean captions, including BOM/CRLF.
        newline = '\r\n' if b'\r\n' in (snapshot.raw or b'') else '\n'
        data = normalized.replace('\n', newline).encode('utf-8')
        if (snapshot.raw or b'').startswith(b'\xef\xbb\xbf'):
            data = b'\xef\xbb\xbf' + data
        atomic_bytes(caption, data)
        return CaptionSnapshot(str(image), data)


class CaptionPreviewRebaser:
    """One local refresh transaction; hash the outfit library only once per batch.

    No Ollama, HTTP, embedding or model calls occur here. apply_caption retains
    its independent last-moment validation even after a successful rebase.
    """
    def __init__(self, store):
        self.store = store
        self._fingerprints = None
        self._library_digest = None
        self._feedback_digest = None

    def _load_context(self):
        if self._fingerprints is None:
            from .clothing_feedback import ClothingFeedbackStore
            profiles = [profile for profile in self.store.load() if profile.enabled]
            self._fingerprints = {p.id: self.store.fingerprint(p) for p in profiles}
            self._library_digest = digest_json(sorted(self._fingerprints.items(), key=lambda item: item[0]))
            self._feedback_digest = ClothingFeedbackStore(self.store).fingerprint()

    def prepare(self, result):
        if result.status != 'matched' or not result.tags:
            return None
        self._load_context()
        if not result.library_digest or result.library_digest != self._library_digest:
            raise CaptionConflict('Outfit library/references changed. Analyze again before applying.')
        if self._fingerprints.get(result.profile_id) != result.profile_digest:
            raise CaptionConflict('Outfit profile changed. Analyze again before applying.')
        if result.feedback_digest and result.feedback_digest != self._feedback_digest:
            raise CaptionConflict('Correction memory changed. Analyze again before applying.')
        from .clothing_service import load_selection
        selected = load_selection(self.store, result.image)
        if selected is not None and list(selected) != list(result.bbox):
            raise CaptionConflict('Main person/outfit selection changed. Analyze again before applying.')
        # prepare_caption independently verifies image bytes, caption encoding,
        # colliding filenames and ownership. A caption save is not a visual edit.
        return prepare_caption(result)
