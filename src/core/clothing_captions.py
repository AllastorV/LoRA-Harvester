"""Preview-first caption editing, provenance and reversible, journaled writes.

Never deletes arbitrary clothing tags. Only tags introduced by this subsystem in
an earlier application are owned; pre-existing/manual tags remain user-owned.
"""
from __future__ import annotations
import base64
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from .clothing_io import (IMAGE_EXTENSIONS, FileLock, atomic_bytes, atomic_json,
                          digest_bytes, file_digest)
from .clothing_profiles import tag_key


class CaptionConflict(RuntimeError):
    pass


def _b64(data):
    return None if data is None else base64.b64encode(data).decode('ascii')


def _unb64(data):
    return None if data is None else base64.b64decode(data, validate=True)


def _optional_bytes(path):
    return path.read_bytes() if path.exists() else None


def _restore(path, data):
    if data is None:
        path.unlink(missing_ok=True)
    else:
        atomic_bytes(path, data)


def _caption_paths(image):
    image = Path(image).resolve()
    caption = image.with_suffix('.txt')
    if caption.is_symlink():
        raise CaptionConflict('Caption is a symbolic link; refusing to replace it.')
    for other in image.parent.iterdir():
        if (other.is_file() and other != image and other.suffix.lower() in IMAGE_EXTENSIONS
                and other.stem.casefold() == image.stem.casefold()):
            raise CaptionConflict(f'{image.name} and {other.name} share the same .txt caption. Rename one first.')
    meta = image.parent / '.lh-clothing'
    for location in (meta, meta / 'state', meta / 'history'):
        if location.is_symlink() or location.resolve() != location.absolute():
            raise CaptionConflict('Clothing metadata must not be a symbolic link or junction.')
    state = meta / 'state' / (digest_bytes(image.name.encode('utf-8')) + '.json')
    if state.is_symlink():
        raise CaptionConflict('Clothing state must not be a symbolic link.')
    return image, caption, meta, state


def merge_caption(existing: str, tags: list[str], previous_owned=None):
    """Promote literal profile tags; normalize only for equality, never for output."""
    previous_owned = {tag_key(t) for t in previous_owned or []}
    source = [part.strip() for part in existing.split(',') if part.strip()]
    base = [t for t in source if tag_key(t) not in previous_owned]
    original_user_keys = {tag_key(t) for t in base}
    output, seen = [], set()
    for tag in tags + base:
        key = tag_key(tag)
        if key and key not in seen:
            output.append(tag)
            seen.add(key)
    owned = [t for t in tags if tag_key(t) not in original_user_keys]
    # Machine-owned tags still present in the new output retain their provenance.
    owned = [t for t in tags if tag_key(t) in previous_owned or t in owned]
    return ', '.join(output), owned


@dataclass
class CaptionPreview:
    image: str
    image_digest: str
    profile_id: str
    profile_digest: str
    before: bytes | None
    after: bytes
    state_before: bytes | None
    state_after: dict
    tags: list[str]
    warnings: list[str] = field(default_factory=list)
    library_digest: str = ''
    feedback_digest: str = ''

    @property
    def old_text(self):
        return (self.before or b'').decode('utf-8-sig')

    @property
    def new_text(self):
        return self.after.decode('utf-8-sig')


def prepare_caption(result) -> CaptionPreview:
    if result.status != 'matched' or not result.tags:
        raise ValueError('Only an accepted match can produce a caption preview.')
    image, caption, _, state_path = _caption_paths(result.image)
    if file_digest(image) != result.image_digest:
        raise CaptionConflict('Image changed since analysis. Analyze again.')
    before, old_state = _optional_bytes(caption), _optional_bytes(state_path)
    existing = (before or b'').decode('utf-8-sig')  # reject non-UTF-8; never replace bad bytes
    previous = json.loads(old_state) if old_state else {}
    if previous and previous.get('schema_version') != 1:
        raise CaptionConflict('Unknown clothing provenance schema; no files changed.')
    # If the saved crop was replaced, ownership cannot safely be transferred to it.
    if previous and previous.get('image_digest') != result.image_digest:
        raise CaptionConflict('Image content changed since the previous outfit write. Undo/review its old caption first.')
    text, owned = merge_caption(existing, result.tags, previous.get('owned_tags', []))
    after = text.encode('utf-8') + b'\n'
    if before and before.startswith(b'\xef\xbb\xbf'):
        after = b'\xef\xbb\xbf' + after
    warnings = []
    if previous and digest_bytes(before or b'') != previous.get('caption_digest'):
        warnings.append('Caption was edited since the last outfit run; only previously auto-added tags will be replaced.')
    visible_keys = {tag_key(t) for t in result.tags}
    existing_keys = {tag_key(t.strip()) for t in existing.split(',')}
    for part in result.parts:
        if tag_key(part['tag']) in existing_keys and tag_key(part['tag']) not in visible_keys:
            warnings.append('An unverified pre-existing tag is preserved: ' + part['tag'])
    state_after = {'schema_version': 1, 'image_digest': result.image_digest,
                   'profile_id': result.profile_id, 'profile_digest': result.profile_digest,
                   'owned_tags': owned, 'output_tags': result.tags,
                   'caption_digest': digest_bytes(after)}
    return CaptionPreview(str(image), result.image_digest, result.profile_id,
                          result.profile_digest, before, after, old_state, state_after,
                          list(result.tags), warnings, getattr(result, 'library_digest', ''),
                          getattr(result, 'feedback_digest', ''))


def apply_caption(preview: CaptionPreview, store=None):
    image, caption, meta, state_path = _caption_paths(preview.image)
    with FileLock(meta / 'write.lock'):
        if (file_digest(image) != preview.image_digest
                or _optional_bytes(caption) != preview.before
                or _optional_bytes(state_path) != preview.state_before):
            raise CaptionConflict('Image/caption changed after preview. Analyze again; nothing was overwritten.')
        if store is not None:
            if preview.feedback_digest:
                from .clothing_feedback import ClothingFeedbackStore
                if ClothingFeedbackStore(store).fingerprint() != preview.feedback_digest:
                    raise CaptionConflict('Correction memory changed after preview. Analyze again.')
            if preview.library_digest and store.library_fingerprint() != preview.library_digest:
                raise CaptionConflict('Outfit library or competing references changed after preview. Analyze again.')
            profile = next((p for p in store.load() if p.id == preview.profile_id and p.enabled), None)
            if profile is None or store.fingerprint(profile) != preview.profile_digest:
                raise CaptionConflict('Outfit profile changed after preview. Analyze again.')
        if preview.before == preview.after:
            return None
        journal = meta / 'history' / (uuid.uuid4().hex + '.json')
        record = {'schema_version': 1, 'status': 'prepared',
                  'image_name': image.name, 'caption_name': caption.name,
                  'state_name': state_path.name,
                  'before': _b64(preview.before), 'after': _b64(preview.after),
                  'state_before': _b64(preview.state_before), 'state_after': preview.state_after}
        atomic_json(journal, record)  # Write-ahead backup exists before any caption mutation.
        try:
            atomic_bytes(caption, preview.after)
            atomic_json(state_path, preview.state_after)
            record['status'] = 'applied'
            atomic_json(journal, record)
        except Exception:
            # If rollback itself fails, the prepared journal remains available for recovery.
            _restore(caption, preview.before)
            _restore(state_path, preview.state_before)
            record['status'] = 'rolled_back'
            atomic_json(journal, record)
            raise
        return str(journal)


def undo_caption(journal_path):
    """Recoverable undo, including interrupted undo and direct-journal recovery.

    Mark undo_prepared before restoring any bytes. A retry accepts only the
    recorded pre/post states; unrelated edits and forged filenames are refused.
    """
    journal = Path(journal_path).resolve()
    if journal.parent.name != 'history' or journal.parent.parent.name != '.lh-clothing':
        raise CaptionConflict('Invalid clothing undo journal location.')
    meta, root = journal.parent.parent, journal.parent.parent.parent
    with FileLock(meta / 'write.lock'):
        record = json.loads(journal.read_text(encoding='utf-8'))
        status = record.get('status')
        if record.get('schema_version') != 1 or status not in ('applied', 'prepared', 'undo_prepared', 'undone'):
            raise CaptionConflict('This journal cannot be undone (invalid status/schema).')
        for key in ('image_name', 'caption_name', 'state_name'):
            value = record[key]
            if (not isinstance(value, str) or value in ('', '.', '..')
                    or Path(value).name != value or '\\' in value or '/' in value):
                raise CaptionConflict('Unsafe path in undo journal.')
        image, caption, actual_meta, state = _caption_paths(root / record['image_name'])
        if (actual_meta != meta or caption.name != record['caption_name']
                or state.name != record['state_name']):
            raise CaptionConflict('Undo journal does not name the matching image/caption/state pair.')
        if not image.is_file() or file_digest(image) != record['state_after']['image_digest']:
            raise CaptionConflict('Source image changed after the outfit write; review manually before undo.')
        after, before = _unb64(record['after']), _unb64(record['before'])
        state_before = _unb64(record['state_before'])
        current = _optional_bytes(caption)
        current_state = _optional_bytes(state)
        if status == 'undone':
            if current != before or current_state != state_before:
                raise CaptionConflict('Files changed since this undo completed.')
            return str(caption)  # direct recovery must not strand the parent undo job
        recovering = status in ('prepared', 'undo_prepared')
        if current != after and not (recovering and current == before):
            raise CaptionConflict('Caption changed after the outfit write; undo would overwrite edits.')
        state_matches = (current_state is not None and json.loads(current_state) == record['state_after'])
        if not state_matches and current_state != state_before:
            raise CaptionConflict('A newer clothing operation changed ownership; undo it first.')
        record['status'] = 'undo_prepared'
        atomic_json(journal, record)
        _restore(caption, before)
        _restore(state, state_before)
        record['status'] = 'undone'
        atomic_json(journal, record)
        return str(caption)
