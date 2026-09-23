"""Collision-safe image/sidecar transfers shared by sorters and exporters.

A basename identifies the complete image + .txt + .json pair. Staging finishes
before destination publication; caught failures roll back the complete pair.
"""
from __future__ import annotations
import os
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from src.core.clothing_io import FileLock, digest_bytes, file_digest

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


def unique_image_path(path: Path) -> Path:
    """Do not reuse a stem owned by another extension or an orphan sidecar."""
    path = Path(path)
    taken = set()
    if path.parent.is_dir():
        # Scan names, not a Path object + suffix property for every entry.
        # Deliberately no stale directory cache: orphan sidecars and files
        # created by external tools must still reserve their casefolded stem.
        extensions = IMAGE_EXTENSIONS | {'.txt', '.json'}
        with os.scandir(path.parent) as entries:
            for entry in entries:
                stem, extension = os.path.splitext(entry.name)
                if extension.lower() in extensions:
                    taken.add(stem.casefold())
    candidate, number = path, 1
    while candidate.stem.casefold() in taken or candidate.exists():
        candidate = path.with_name(f'{path.stem}_{number}{path.suffix}')
        number += 1
    return candidate


def _publish_exclusive(source: Path, destination: Path) -> None:
    """Never replace a file created by another process during this operation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return
    except FileExistsError:
        raise
    except OSError:
        pass  # FAT/network shares may not support hard links.
    created = False
    try:
        with destination.open('xb') as target:
            created = True
            with source.open('rb') as src:
                shutil.copyfileobj(src, target)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def transfer_image_pair(source: Path, destination_dir: Path, *, copy: bool = True,
                        image_bytes: bytes | None = None, target_name: str | None = None,
                        expected_image_digest: str | None = None) -> Path:
    """Return the final image path, preserving .txt/.json and clothing ownership.

    Existing destination captions are never overwritten. Move uses staged copy
    then source deletion, with rollback on a caught I/O failure (not a claim of
    multi-file filesystem atomicity across power loss).
    """
    if image_bytes is not None and (not copy or not isinstance(image_bytes, bytes) or not image_bytes):
        raise ValueError('Derived images require a non-empty byte payload and copy=True.')
    if target_name is not None and (Path(target_name).name != target_name
                                   or '/' in target_name or '\\' in target_name
                                   or Path(target_name).suffix.lower() not in IMAGE_EXTENSIONS):
        raise ValueError('Target must be a plain image filename.')
    source = Path(source)
    if source.is_symlink():
        raise ValueError('Refusing to transfer a symbolic-link image.')
    source = source.resolve()
    destination_dir = Path(destination_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.parent == destination_dir and image_bytes is None and target_name is None:
        return source
    destination_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for folder in sorted({source.parent, destination_dir}, key=str):
            stack.enter_context(FileLock(folder / '.lh-clothing' / 'write.lock'))
        if expected_image_digest is not None and file_digest(source) != expected_image_digest:
            raise ValueError('Source image changed during processing; output was not saved.')
        # Without sidecars there is no caption-ownership ambiguity to scan for.
        if any(source.with_suffix(ext).exists() for ext in ('.txt', '.json')):
            wanted = source.stem.casefold()
            with os.scandir(source.parent) as entries:
                ambiguous = any(entry.name != source.name
                    and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS
                    and os.path.splitext(entry.name)[0].casefold() == wanted
                    for entry in entries)
            if ambiguous:
                raise ValueError(f'Ambiguous caption ownership for {source.name}; rename same-stem images first.')
        destination = unique_image_path(destination_dir / (target_name or source.name))
        pairs = [(source, destination)]
        for extension in ('.txt', '.json'):
            sidecar = source.with_suffix(extension)
            if sidecar.is_symlink():
                raise ValueError('Refusing to transfer a symbolic-link sidecar.')
            if sidecar.is_file():
                pairs.append((sidecar, destination.with_suffix(extension)))
        # Ownership follows an unchanged image; old undo journals remain at the
        # original folder and are not rewritten to target a moved asset.
        state_name = lambda p: digest_bytes(p.name.encode('utf-8')) + '.json'
        old_state = source.parent / '.lh-clothing' / 'state' / state_name(source)
        new_state = destination_dir / '.lh-clothing' / 'state' / state_name(destination)
        if old_state.is_file():
            if new_state.exists() or old_state.is_symlink():
                raise ValueError('Destination clothing ownership conflicts with this transfer.')
            pairs.append((old_state, new_state))
        # A derived image inherits only provenance verified against its source.
        # This also keeps ownership usable after lossless PNG upscaling.
        original_digests = {original: file_digest(original) for original, _ in pairs}
        if expected_image_digest is not None and original_digests[source] != expected_image_digest:
            raise ValueError('Source image changed during processing; output was not saved.')
        replacements = {} if image_bytes is None else {source: image_bytes}
        if image_bytes is not None and old_state.is_file():
            state = json.loads(old_state.read_text(encoding='utf-8'))
            if not isinstance(state, dict) or state.get('image_digest') != original_digests[source]:
                pairs = [(a, b) for a, b in pairs if a != old_state]
            else:
                state['derived_from_image_digest'] = original_digests[source]
                state['image_digest'] = digest_bytes(image_bytes)
                replacements[old_state] = json.dumps(state, ensure_ascii=False,
                                                     allow_nan=False, indent=2).encode('utf-8')
        staged, published, deleted = [], [], []
        try:
            for original, final in pairs:
                fd, name = tempfile.mkstemp(prefix='.lh-transfer-', suffix='.tmp', dir=destination_dir)
                os.close(fd)
                temp = Path(name)
                staged.append((original, final, temp))
                if original in replacements:
                    with temp.open('wb') as target:
                        target.write(replacements[original])
                        target.flush()
                        os.fsync(target.fileno())
                else:
                    shutil.copy2(original, temp)
                    if original_digests[original] != file_digest(temp):
                        raise OSError(f'Source changed while copying: {original}')
            for original, _, _ in staged:
                if file_digest(original) != original_digests[original]:
                    raise OSError(f'Source changed while staging: {original}')
            for original, final, temp in staged:
                _publish_exclusive(temp, final)
                published.append(final)
            if not copy:
                for original, final, temp in staged:
                    if file_digest(original) != file_digest(temp):
                        raise OSError(f'Source changed before move: {original}')
                for original, final, temp in staged:
                    original.unlink()
                    deleted.append((original, temp))
            return destination
        except Exception:
            # Recover removed source members before removing the destination.
            # On rollback failure, leave destination copies for recovery.
            for original, temp in deleted:
                if not original.exists():
                    _publish_exclusive(temp, original)
            for final in reversed(published):
                final.unlink(missing_ok=True)
            raise
        finally:
            for _, _, temp in staged:
                temp.unlink(missing_ok=True)


def save_image_bytes_unique(path: Path, data: bytes) -> Path:
    """Publish one newly encoded image without replacing any dataset member."""
    path = Path(path)
    if not isinstance(data, bytes) or not data:
        raise ValueError('Encoded image is empty.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.parent / '.lh-clothing' / 'write.lock'):
        destination = unique_image_path(path)
        fd, name = tempfile.mkstemp(prefix='.lh-image-', suffix='.tmp', dir=path.parent)
        temp = Path(name)
        try:
            with os.fdopen(fd, 'wb') as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            _publish_exclusive(temp, destination)
            return destination
        finally:
            temp.unlink(missing_ok=True)
