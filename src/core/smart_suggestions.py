"""Read-only, evidence-based dataset diagnostics for the Studio desktop shell.

This is NOT an image-quality model or an inference engine. Header dimensions,
caption contents, explicit outfit tags and optional SHA256 duplicates are the
only evidence. Suggestions never delete, edit, export or start a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Callable, Iterable

from PIL import Image

from .dataset_balance import caption_labels, UNKNOWN, AMBIGUOUS
from .dataset_scanner import IMAGE_EXTS

SCHEMA_VERSION = 1
SKIP_DIRS = {'__pycache__', '.git', 'venv', '.venv', '_rejected', '_approved'}
MAX_CAPTION_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 120_000_000


class ScanCancelled(RuntimeError):
    pass


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise ScanCancelled('Dataset scan cancelled; no source files were changed.')


def _tag_key(tag):
    return re.sub(r'[\s_]+', ' ', tag.strip().casefold())


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class ScanOptions:
    recursive: bool = True
    minimum_side: int = 512
    exact_duplicates: bool = False

    def validate(self):
        if type(self.recursive) is not bool or type(self.exact_duplicates) is not bool:
            raise ValueError('Scan flags must be booleans.')
        if type(self.minimum_side) is not int or not 0 <= self.minimum_side <= 8192:
            raise ValueError('Minimum side must be an integer from 0 to 8192.')
        return self


@dataclass
class DatasetRecord:
    path: str
    relative_path: str
    caption: str = ''
    caption_state: str = 'missing'
    width: int = 0
    height: int = 0
    byte_size: int = 0
    tag_count: int = 0
    outfit: str = UNKNOWN
    pose: str = UNKNOWN
    angle: str = UNKNOWN
    issues: list[str] = field(default_factory=list)
    error: str = ''
    duplicate_group: str = ''
    signature: tuple = ()


@dataclass(frozen=True)
class Suggestion:
    key: str
    severity: str
    title_tr: str
    title_en: str
    detail_tr: str
    detail_en: str
    paths: tuple[str, ...] = ()
    route: str = 'library'

    @property
    def count(self):
        return len(self.paths)


@dataclass
class DatasetReport:
    root: str
    options: ScanOptions
    records: list[DatasetRecord]
    suggestions: list[Suggestion]
    created_at: str
    elapsed_seconds: float
    cache_hits: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self):
        return len(self.records)

    @property
    def captioned(self):
        return sum(r.caption_state == 'present' for r in self.records)

    @property
    def missing(self):
        return sum(r.caption_state in ('missing', 'empty') for r in self.records)

    @property
    def duplicate_extras(self):
        counts = Counter(r.duplicate_group for r in self.records if r.duplicate_group)
        return sum(n - 1 for n in counts.values())

    @property
    def flagged(self):
        return sum(bool(r.issues) for r in self.records)

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


class MetadataCache:
    """Optional local performance cache, not a source of write authorization.

    Stat signatures include inode, ctime and mtime. Caption content is always
    reread. Cache failure only disables the optimization, not diagnostics.
    A separate connection is owned by each scan worker.
    """
    def __init__(self, path: Path | None):
        self.connection = None
        self.hits = 0
        self.warning = ''
        if path is None:
            return
        try:
            path = Path(path)
            if path.is_symlink():
                raise OSError('Metadata cache must not be a symbolic link.')
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(path), timeout=2)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('''CREATE TABLE IF NOT EXISTS image_metadata_v1
                (path TEXT PRIMARY KEY, signature TEXT, width INTEGER, height INTEGER,
                 digest TEXT NOT NULL DEFAULT '')''')
        except (OSError, sqlite3.Error) as exc:
            self.warning = str(exc)
            self.close()

    def get(self, path, signature):
        if self.connection is None:
            return None
        try:
            row = self.connection.execute(
                'SELECT width,height,digest FROM image_metadata_v1 WHERE path=? AND signature=?',
                (str(path), repr(signature))).fetchone()
            if row:
                self.hits += 1
            return row
        except sqlite3.Error as exc:
            self.warning = str(exc)
            self.close()
            return None

    def put(self, path, signature, width, height, digest):
        if self.connection is None:
            return
        try:
            self.connection.execute('INSERT OR REPLACE INTO image_metadata_v1 VALUES (?,?,?,?,?)',
                                    (str(path), repr(signature), width, height, digest))
        except sqlite3.Error as exc:
            self.warning = str(exc)
            self.close()

    def close(self):
        if self.connection:
            try:
                self.connection.commit()
                self.connection.close()
            except sqlite3.Error:
                pass
            self.connection = None


def iter_dataset_images(root: Path, recursive: bool, cancel=None, warnings=None):
    """Prune helper directories BEFORE traversal; never follow directory links."""
    def onerror(exc):
        if warnings is not None:
            warnings.append(str(exc))
    for directory, dirs, names in os.walk(root, topdown=True, followlinks=False, onerror=onerror):
        _cancelled(cancel)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.lh-')
                         and not Path(directory, d).is_symlink()) if recursive else []
        for name in sorted(names):
            _cancelled(cancel)
            path = Path(directory, name)
            if path.suffix.lower() in IMAGE_EXTS and not path.is_symlink() and path.is_file():
                yield path


def _read_caption(path):
    if path.is_symlink():
        raise ValueError('Symbolic-link captions are not read.')
    if not path.exists():
        return '', 'missing'
    before = _signature(path)
    if before[2] > MAX_CAPTION_BYTES:
        raise ValueError('Caption exceeds the 2 MiB diagnostic limit.')
    with path.open('rb') as handle:
        raw = handle.read(MAX_CAPTION_BYTES + 1)
    if len(raw) > MAX_CAPTION_BYTES or _signature(path) != before:
        raise ValueError('Caption changed during the scan; scan again.')
    text = raw.decode('utf-8-sig').strip()
    return text, ('present' if text else 'empty')


def _digest(path, cancel):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            _cancelled(cancel)
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_suggestions(records: list[DatasetRecord], options: ScanOptions) -> list[Suggestion]:
    """Deterministic rules. No invented global health or training-readiness score."""
    result = []

    def add(key, severity, tr, en, detail_tr, detail_en, route='library'):
        paths = tuple(r.path for r in records if key in r.issues)
        if paths:
            result.append(Suggestion(key, severity, tr.format(n=len(paths)), en.format(n=len(paths)),
                                     detail_tr, detail_en, paths, route))

    add('caption_collision', 'error', '{n} görsel aynı caption dosyasını paylaşıyor',
        '{n} images share a caption path',
        'Aynı klasörde aynı kök ada sahip görseller var. Yazmadan önce ad çakışmasını gider.',
        'Images with the same stem share a .txt file. Resolve names before writing.')
    add('unreadable_image', 'error', '{n} görsel başlığı okunamadı', '{n} image headers could not be read',
        'Dosya bozuk, desteklenmeyen veya çok büyük olabilir. Kaynak dosyayı incele; otomatik silinmez.',
        'Files may be invalid, unsupported or oversized. Inspect originals; nothing is deleted.')
    add('changed_image', 'warning', '{n} görsel tarama sırasında değişti', '{n} images changed during scanning',
        'Bu dosyaların eski ölçümleri kullanılmadı. İşlemler bittikten sonra yeniden tara.',
        'Stale measurements were discarded. Rescan after other operations finish.')
    add('unreadable_caption', 'error', '{n} caption okunamadı', '{n} captions could not be read',
        'İzin, kodlama veya dosya türünü kontrol et. Okunamayan caption boş sayılmaz.',
        'Check permissions, encoding or file type. Unreadable captions are not counted as empty.')
    add('missing_caption', 'warning', '{n} görselin caption dosyası yok', '{n} images have no caption file',
        'Listede incele; Edit ile düzenle veya Otomatik etiketleme aracını aç.',
        'Inspect the list; edit manually or open the existing automatic tagger.')
    add('empty_caption', 'warning', '{n} caption boş', '{n} captions are empty',
        'Dosya var, ancak yalnız boşluk içeriyor veya içerik yok. Kendiliğinden tag eklenmez.',
        'Files exist but contain no text. Tags are not added automatically.')
    add('small_image', 'info', '{n} görselin kısa kenarı düşük', '{n} images have a small short edge',
        f'Kısa kenar {options.minimum_side} pikselin altında. Bu boyut uyarısıdır, estetik kalite puanı değil.',
        f'Short edge is below {options.minimum_side} pixels. A size warning, not an aesthetic score.')
    add('duplicate_tags', 'info', '{n} caption tekrar eden tag içeriyor', '{n} captions contain repeated tags',
        'Büyük/küçük harf ve boşluk/alt çizgi farkları normalize edilerek sayıldı. Edit ile kontrol et.',
        'Case and whitespace/underscore variants were normalized. Review in Edit.')
    add('multiple_outfits', 'info', '{n} caption birden fazla kıyafet master tagi içeriyor',
        '{n} captions contain multiple outfit masters',
        'Çok kişili görüntü olabilir. Ana kıyafet kuralına göre incele; etiketler otomatik silinmez.',
        'The image may contain multiple people. Review the main outfit; tags are not removed.')
    add('duplicate_image', 'info', '{n} görsel birebir kopya gruplarında', '{n} images belong to exact duplicate groups',
        'SHA256 ile dosya byte eşitliği kontrol edildi; benzer poz/çizim tespiti değildir. Captionlar farklı olabilir.',
        'SHA256 checks byte-identical files, not similar poses/artwork. Captions may differ.')
    counts = Counter(r.outfit for r in records if r.outfit not in (UNKNOWN, AMBIGUOUS))
    if len(counts) >= 2:
        largest = max(counts.values())
        rare = {name for name, count in counts.items() if count * 4 < largest}
        paths = tuple(r.path for r in records if r.outfit in rare)
        if paths:
            result.append(Suggestion('outfit_imbalance', 'info',
                f'{len(rare)} kıyafet etiketi diğerlerine göre seyrek',
                f'{len(rare)} outfit labels are underrepresented',
                'Caption sayısı en büyük grubun dörtte birinden az. Bu bir dengeleme önerisidir; zorunlu hedef değil.',
                'Caption counts are below one quarter of the largest group. A balancing hint, not a required target.',
                paths, 'balance'))
    return result


def scan_suggestions(root: str | Path, *, options: ScanOptions | None = None,
                     masters: Iterable[str] = (), cache_path: Path | None = None,
                     cancel: threading.Event | None = None,
                     progress: Callable[[int, int, str], None] | None = None) -> DatasetReport:
    options = (options or ScanOptions()).validate()
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Choose an existing dataset directory.')
    started = time.perf_counter()
    masters = tuple(masters)
    warnings = []
    paths = list(iter_dataset_images(root, options.recursive, cancel, warnings))
    cache = MetadataCache(cache_path)
    records, hashes = [], defaultdict(list)
    captions = defaultdict(list)
    last_progress = 0.0
    try:
        for index, path in enumerate(paths):
            _cancelled(cancel)
            record = DatasetRecord(str(path), str(path.relative_to(root)))
            before = None
            try:
                before = _signature(path)
                record.signature, record.byte_size = before, before[2]
                cached = cache.get(path, before)
                width, height, digest = cached or (0, 0, '')
                if not width or not height:
                    with Image.open(path) as image:
                        width, height = image.size
                    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                        raise ValueError('Image dimensions exceed the diagnostic limit.')
                if options.exact_duplicates and not digest:
                    digest = _digest(path, cancel)
                if _signature(path) != before:
                    record.issues.append('changed_image')
                    record.error = 'Image changed during scan.'
                else:
                    record.width, record.height = width, height
                    if options.minimum_side and min(width, height) < options.minimum_side:
                        record.issues.append('small_image')
                    if options.exact_duplicates:
                        hashes[digest].append(record)
                    cache.put(path, before, width, height, digest)
            except ScanCancelled:
                raise
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                record.issues.append('unreadable_image')
                record.error = str(exc)
            cap_path = path.with_suffix('.txt')
            captions[os.path.normcase(str(cap_path))].append(record)
            try:
                record.caption, record.caption_state = _read_caption(cap_path)
                if record.caption_state != 'present':
                    record.issues.append(record.caption_state + '_caption')
                tokens = [t.strip() for t in re.split(r'[,\r\n]+', record.caption) if t.strip()]
                record.tag_count = len(tokens)
                if len(tokens) != len({_tag_key(t) for t in tokens}):
                    record.issues.append('duplicate_tags')
                try:
                    labels = caption_labels(record.caption, masters)
                    record.outfit, record.pose, record.angle = (labels[k] for k in ('outfit', 'pose', 'angle'))
                    if record.outfit == AMBIGUOUS:
                        record.issues.append('multiple_outfits')
                except ValueError:
                    # Long natural-language captions may exceed the tag-parser limit.
                    pass
            except (OSError, ValueError, UnicodeError) as exc:
                record.caption_state = 'unreadable'
                record.issues.append('unreadable_caption')
                record.error = '; '.join(filter(None, (record.error, str(exc))))
            records.append(record)
            now = time.perf_counter()
            if progress and (now - last_progress >= .1 or index + 1 == len(paths)):
                progress(index + 1, len(paths), path.name)
                last_progress = now
        _cancelled(cancel)
        for group in captions.values():
            if len(group) > 1:
                for record in group:
                    record.issues.append('caption_collision')
        for digest, group in hashes.items():
            if len(group) > 1:
                for record in group:
                    record.duplicate_group = digest
                    record.issues.append('duplicate_image')
        if cache.warning:
            warnings.append('Optional metadata cache unavailable: ' + cache.warning)
        return DatasetReport(str(root), options, records, build_suggestions(records, options),
                             datetime.now(timezone.utc).isoformat(), time.perf_counter() - started,
                             cache.hits, warnings)
    finally:
        cache.close()
