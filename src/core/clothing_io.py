"""Small, dependency-light I/O helpers for the optional clothing subsystem."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(data) -> str:
    return digest_bytes(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False).encode('utf-8'))


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    """Replace one file atomically; temporary file stays on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, data) -> None:
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, indent=2,
                                  allow_nan=False).encode('utf-8'))


def load_rgb(path: Path) -> Image.Image:
    """Unicode paths, EXIF orientation and alpha are handled identically everywhere."""
    with Image.open(path) as source:
        if source.width * source.height > 60_000_000:
            raise ValueError('Image exceeds 60 megapixels; resize it before analysis.')
        image = ImageOps.exif_transpose(source)
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            return bg.convert('RGB')
        return image.convert('RGB').copy()


def validate_box(box):
    if box is None:
        return None
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) for v in box)):
        raise ValueError('Invalid bounding box.')
    x1, y1, x2, y2 = (float(v) for v in box)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('Bounding box must be normalized to [0, 1].')
    return (x1, y1, x2, y2)


def crop_rgb(image: Image.Image, box=None) -> Image.Image:
    box = validate_box(box)
    if box is None:
        return image.copy()
    w, h = image.size
    x1, y1, x2, y2 = box
    pixels = (int(x1 * w), int(y1 * h), min(w, math.ceil(x2 * w)),
              min(h, math.ceil(y2 * h)))
    if pixels[2] - pixels[0] < 8 or pixels[3] - pixels[1] < 8:
        raise ValueError('Selected area is too small (minimum 8 x 8 pixels).')
    return image.crop(pixels)


def find_images(folder: Path, recursive: bool = False):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError('Image folder does not exist.')
    candidates = root.rglob('*') if recursive else root.iterdir()
    return sorted((p for p in candidates if p.is_file() and not p.is_symlink()
                   and p.suffix.lower() in IMAGE_EXTENSIONS
                   and '.lh-clothing' not in p.relative_to(root).parts),
                  key=lambda p: str(p).casefold())


class FileLock:
    """Non-blocking OS lock, released by the OS on crashes. Never deletes its inode."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        if self.handle.seek(0, 2) == 0:
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError('Another clothing operation is using this resource.') from exc
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            try:
                self.handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None
