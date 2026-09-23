"""Bounded asynchronous previews. No QPixmap is created outside the GUI thread."""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QStyle, QStyleOptionViewItem, QStyledItemDelegate, QTableView
from PIL import Image, ImageOps


def image_key(path, size):
    path = Path(path)
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, tuple(size))


class MediaLoader(QObject):
    ready = pyqtSignal(object)  # (key, QImage, oriented source dimensions, error)

    def __init__(self, parent=None, memory_limit=48 * 1024 * 1024):
        super().__init__(parent)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='lh-preview')
        self.memory_limit = memory_limit
        self.cache = OrderedDict()
        self.used = 0
        self.pending = set()
        self.closed = threading.Event()
        self.ready.connect(self._accept)

    def request(self, path, size=(96, 96)):
        if self.closed.is_set():
            return None
        try:
            key = image_key(path, size)
        except OSError as exc:
            return QImage(), (0, 0), str(exc)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if key not in self.pending and len(self.pending) < 40:
            self.pending.add(key)
            future = self.pool.submit(self._decode, key)
            future.add_done_callback(self._done)
        return None

    @staticmethod
    def _decode(key):
        path, _, _, _, size = key
        try:
            with Image.open(path) as source:
                if source.width * source.height > 120_000_000:
                    raise ValueError('Image too large for an automatic preview.')
                # JPEG draft reduces decode memory. EXIF orientation is honored.
                original = source.size
                orientation = source.getexif().get(274, 1)
                if orientation in (5, 6, 7, 8):
                    original = original[::-1]
                source.draft('RGB', size)
                image = ImageOps.exif_transpose(source)
                image.thumbnail(size, Image.Resampling.LANCZOS)
                image = image.convert('RGBA')
                raw = image.tobytes()
                qimage = QImage(raw, image.width, image.height, image.width * 4,
                                QImage.Format_RGBA8888).copy()
            if image_key(path, size) != key:
                raise OSError('Image changed while its preview was loading.')
            return key, qimage, original, ''
        except Exception as exc:
            return key, QImage(), (0, 0), str(exc)

    def _done(self, future):
        if self.closed.is_set() or future.cancelled():
            return
        try:
            self.ready.emit(future.result())
        except RuntimeError:
            pass  # owner already destroyed during application shutdown

    def _accept(self, payload):
        key, image, dimensions, error = payload
        self.pending.discard(key)
        if self.closed.is_set():
            return
        size = image.byteCount() if not image.isNull() else 0
        if size > self.memory_limit:
            return
        old = self.cache.pop(key, None)
        if old:
            self.used -= old[0].byteCount()
        self.cache[key] = (image, dimensions, error)
        self.used += size
        while self.cache and (self.used > self.memory_limit or len(self.cache) > 256):
            _, (old_image, _, _) = self.cache.popitem(last=False)
            self.used -= old_image.byteCount()

    def shutdown(self):
        if not self.closed.is_set():
            self.closed.set()
            self.pool.shutdown(wait=False, cancel_futures=True)


class ImageThumbnailDelegate(QStyledItemDelegate):
    """Show small image previews beside existing names; decode visible rows only."""

    def __init__(self, view, media, path_for_index, size=44):
        super().__init__(view)
        self.view, self.media = view, media
        self.path_for_index, self.size = path_for_index, size
        placeholder = QPixmap(size, size)
        placeholder.fill(Qt.transparent)
        self.placeholder = QIcon(placeholder)
        media.ready.connect(self._ready)

    def paint(self, painter, option, index):
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        styled.icon = self.placeholder
        styled.features |= QStyleOptionViewItem.HasDecoration
        styled.decorationPosition = QStyleOptionViewItem.Left
        styled.decorationSize = QSize(self.size, self.size)
        try:
            path = self.path_for_index(index)
            cached = self.media.request(path, (self.size, self.size)) if path else None
            if cached and not cached[0].isNull():
                styled.icon = QIcon(QPixmap.fromImage(cached[0]))
        except (IndexError, AttributeError, OSError, TypeError):
            pass
        style = styled.widget.style() if styled.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, styled, painter, styled.widget)

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        hint.setHeight(max(hint.height(), self.size + 10))
        return hint

    def _ready(self, payload):
        if payload[0][-1] == (self.size, self.size):
            self.view.viewport().update()


def attach_image_thumbnails(view, media, path_for_index, column=None, size=44):
    """Add previews to a Qt file list without changing its model or selection."""
    delegate = ImageThumbnailDelegate(view, media, path_for_index, size)
    if column is None:
        view.setItemDelegate(delegate)
    else:
        view.setItemDelegateForColumn(column, delegate)
    if isinstance(view, QTableView):
        view.verticalHeader().setDefaultSectionSize(size + 10)
    view._image_thumbnail_delegate = delegate
    return delegate
