"""Journal AI editor writes with the same optimistic concurrency/lock as Clothing."""
from __future__ import annotations
import base64
import hashlib
import json
import re
import uuid
from pathlib import Path
from .ai_control import ControlError, digest
from .caption_sync import CaptionSnapshot, read_caption_snapshot
from .clothing_captions import _caption_paths
from .clothing_io import FileLock, atomic_bytes, atomic_json


def image_stamp(image):
    stat = Path(image).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def revision(image, raw, draft, stamp=None):
    return digest({'image': str(Path(image).resolve()), 'stamp': image_stamp(image) if stamp is None else stamp,
                   'raw': None if raw is None else hashlib.sha256(raw).hexdigest(), 'draft': draft})


def _encode(raw):
    return None if raw is None else base64.b64encode(raw).decode('ascii')


def save(snapshot: CaptionSnapshot, text: str, request_id: str, expected_stamp=None):
    image, caption, meta, _ = _caption_paths(snapshot.image)
    if len(text.encode('utf-8')) > 262144:
        raise ControlError('caption_size', 'AI caption is too large.')
    with FileLock(meta / 'write.lock'):
        if expected_stamp is not None and image_stamp(image) != expected_stamp:
            raise ControlError('image_changed', 'Source changed after the AI preview.')
        current = read_caption_snapshot(image)
        if current.raw != snapshot.raw:
            raise ControlError('caption_conflict', 'Caption changed after preview; nothing was overwritten.')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        if current.raw is not None and text == current.text:
            return current, None
        newline = '\r\n' if b'\r\n' in (current.raw or b'') else '\n'
        data = text.replace('\n', newline).encode('utf-8')
        if (current.raw or b'').startswith(b'\xef\xbb\xbf'):
            data = b'\xef\xbb\xbf' + data
        journal = meta / 'history' / ('ai-caption-' + uuid.uuid4().hex + '.json')
        record = {'schema': 1, 'type': 'ai_editor', 'image': str(image), 'stamp': image_stamp(image),
                  'before': _encode(current.raw), 'after': _encode(data), 'request_id': request_id, 'phase': 'prepared'}
        atomic_json(journal, record)  # Persist recovery bytes BEFORE replacing caption.
        if expected_stamp is not None and image_stamp(image) != expected_stamp:
            raise ControlError('image_changed', 'Source changed before caption commit.')
        atomic_bytes(caption, data)
        record['phase'] = 'applied'
        try:
            atomic_json(journal, record)
        except OSError:
            pass  # The prepared journal can still safely undo using exact after bytes.
        return CaptionSnapshot(str(image), data), journal.name


def undo(image, name):
    if not re.fullmatch(r'ai-caption-[0-9a-f]{32}\.json', name):
        raise ControlError('journal', 'Use the exact AI journal filename returned by caption.set.')
    image, caption, meta, _ = _caption_paths(image)
    journal = meta / 'history' / name
    if journal.is_symlink() or not journal.is_file() or journal.stat().st_size > 2 * 1024 * 1024:
        raise ControlError('journal', 'Invalid journal.')
    with FileLock(meta / 'write.lock'):
        record = json.loads(journal.read_text(encoding='utf-8'))
        if record.get('schema') != 1 or record.get('type') != 'ai_editor' or record.get('image') != str(image):
            raise ControlError('journal', 'Journal belongs to another image or protocol.')
        if record.get('stamp') != image_stamp(image):
            raise ControlError('image_changed', 'Source image changed; undo refused.')
        before = None if record['before'] is None else base64.b64decode(record['before'], validate=True)
        after = base64.b64decode(record['after'], validate=True)
        raw = read_caption_snapshot(image).raw
        if raw == before and record['phase'] in ('undo_prepared', 'undone'):
            return CaptionSnapshot(str(image), before)
        if raw != after:
            raise ControlError('caption_conflict', 'Caption was edited after the AI write. Undo refused.')
        record['phase'] = 'undo_prepared'; atomic_json(journal, record)
        if before is None:
            caption.unlink(missing_ok=True)
        else:
            atomic_bytes(caption, before)
        record['phase'] = 'undone'; atomic_json(journal, record)
        return CaptionSnapshot(str(image), before)
