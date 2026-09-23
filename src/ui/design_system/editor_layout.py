"""Rearrange the EXISTING editor controls. There is only one caption/draft store."""
from __future__ import annotations
from pathlib import Path
from PyQt5.QtCore import Qt, QObject, QTimer, QSize, QPoint, QEvent
from PyQt5.QtGui import QPixmap, QIcon, QKeySequence
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QListView,
                             QToolButton, QMenu, QShortcut, QSizePolicy, QFrame)
from src.ui import theme
from .widgets import button, label, line_icon
from .media import image_key


class EditorStudioLayout(QObject):
    def __init__(self, editor, media):
        super().__init__(editor)
        self.editor, self.media = editor, media
        self._selected = None
        self._image = None
        self._rows = {}
        self._icon_rows = set()
        self._legacy_items = []
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(80)
        self._thumb_timer.timeout.connect(self.request_visible_thumbnails)
        root = editor.layout()
        while root.count():
            self._legacy_items.append(root.takeAt(0))
        for item in self._legacy_items:
            if item.layout():
                for i in range(item.layout().count()):
                    child = item.layout().itemAt(i).widget()
                    if child is not None:
                        child.hide()
        editor._left_card.hide()
        editor._thumb_grid.hide()
        for widget in (editor.load_btn, editor.save_btn, editor.add_tag_btn,
                       editor.remove_tag_btn, editor.replace_tag_btn):
            widget.hide()
        root.setContentsMargins(12, 8, 12, 10)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        self.heading = label(role='heading')
        toolbar.addWidget(self.heading)
        toolbar.addStretch()
        self.previous = button('‹', lambda: self._move(-1))
        self.next = button('›', lambda: self._move(1))
        for btn in (self.previous, self.next):
            btn.setFixedWidth(32)
        self.position = label('—', role='muted')
        self.position.setWordWrap(False)
        toolbar.addWidget(self.previous)
        toolbar.addWidget(self.position)
        toolbar.addWidget(self.next)
        self.bulk = QToolButton()
        self.bulk.setPopupMode(QToolButton.InstantPopup)
        self.bulk.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.bulk.setIcon(line_icon('tools'))
        theme.bind_style(self.bulk, theme.btn_secondary)
        self.menu = QMenu(self.bulk)
        self.bulk.setMenu(self.menu)
        self.menu.aboutToShow.connect(self._prepare_menu)
        toolbar.addWidget(self.bulk)
        root.addLayout(toolbar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        editor._preview_card.setMinimumWidth(190)
        editor._editor_card.setMinimumWidth(290)
        self.splitter.addWidget(editor._preview_card)
        self.splitter.addWidget(editor._editor_card)
        self.splitter.setStretchFactor(0, 6)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setSizes([530, 420])
        editor._preview_card.show()
        editor._editor_card.show()
        editor.filename_lbl.setTextFormat(Qt.PlainText)
        root.addWidget(self.splitter, 1)
        self.details = label('', role='muted')
        editor._preview_card.layout().addWidget(self.details)
        editor.preview_lbl.installEventFilter(self)
        editor.preview_lbl.setMinimumSize(160, 220)
        editor.preview_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._caption_heading = label(role='heading')
        # Expose the existing, formerly hidden text store, not a second editor.
        editor._editor_card.layout().insertWidget(1, self._caption_heading)
        editor._editor_card.layout().insertWidget(2, editor.caption_edit)
        editor.caption_edit.setMinimumHeight(120)
        editor.caption_edit.setMaximumHeight(260)
        editor.caption_edit.show()
        theme.bind_style(editor.caption_edit, lambda: f'QTextEdit{{background:{theme.BG_SURFACE};border:1px solid {theme.BORDER};border-radius:7px;padding:10px;color:{theme.TEXT_PRIMARY};font-size:{theme.fs(12)};}} QTextEdit:focus{{border-color:{theme.get_accent()};}}')
        for btn in (editor._save_one_btn, editor._save_next_btn, editor._revert_btn):
            btn.setMinimumWidth(0)
        editor._chip_scroll.setMinimumHeight(90)
        editor._chip_scroll.setMaximumHeight(180)

        self.filmstrip = QFrame()
        self.filmstrip.setObjectName('studioFilmstrip')
        film = QVBoxLayout(self.filmstrip)
        film.setContentsMargins(8, 4, 8, 4)
        film.setSpacing(0)
        view = editor.image_list
        view.setViewMode(QListView.IconMode)
        view.setFlow(QListView.LeftToRight)
        view.setWrapping(False)
        view.setMovement(QListView.Static)
        view.setResizeMode(QListView.Adjust)
        view.setLayoutMode(QListView.Batched)
        view.setBatchSize(100)
        view.setUniformItemSizes(True)
        view.setIconSize(QSize(78, 70))
        view.setGridSize(QSize(98, 102))
        view.setFixedHeight(116)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        view.horizontalScrollBar().valueChanged.connect(lambda _: self._thumb_timer.start())
        view.installEventFilter(self)
        film.addWidget(view)
        view.show()
        theme.bind_style(self.filmstrip, lambda: f'QFrame#studioFilmstrip{{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:8px;}}')
        root.addWidget(self.filmstrip)
        root.addWidget(editor.status_lbl)
        editor.status_lbl.show()
        editor.status_lbl.setWordWrap(True)
        media.ready.connect(self._received)
        self._shortcuts = []
        for sequence, callback in [('Ctrl+S', editor._save_current),
                                   ('Ctrl+Shift+S', editor._save_all),
                                   ('Alt+Right', lambda: self._move(1)),
                                   ('Alt+Left', lambda: self._move(-1))]:
            shortcut = QShortcut(QKeySequence(sequence), editor)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        self.update_language(editor.lang)

    def _prepare_menu(self):
        self.menu.clear()
        e = self.editor
        for btn in (e.load_btn, e.save_btn, e.add_tag_btn, e.remove_tag_btn, e.replace_tag_btn):
            action = self.menu.addAction(btn.text())
            action.setEnabled(btn.isEnabled())
            action.triggered.connect(btn.click)

    def update_language(self, lang):
        tr = lang == 'tr'
        self.heading.setText('Görsel ve caption düzenleme' if tr else 'Image and caption editor')
        self._caption_heading.setText('Caption — dosyaya kaydedilecek metin' if tr else 'Caption — text saved to file')
        self.bulk.setText('Toplu işlemler' if tr else 'Batch actions')
        self.editor._save_one_btn.setText('Kaydet' if tr else 'Save')
        self.editor._save_next_btn.setText('Kaydet →' if tr else 'Save →')
        self.editor._revert_btn.setText('Geri al' if tr else 'Revert')
        self.editor._editor_title.setText('Etiketler' if tr else 'Tags')
        self.previous.setToolTip('Önceki görsel · Alt+Sol' if tr else 'Previous image · Alt+Left')
        self.next.setToolTip('Sonraki görsel · Alt+Sağ' if tr else 'Next image · Alt+Right')

    def _move(self, offset):
        view = self.editor.image_list
        target = view.currentRow() + offset
        if 0 <= target < view.count():
            view.setCurrentRow(target)
            view.scrollToItem(view.item(target))

    def after_folder_loaded(self):
        self._rows = {path: row for row, (path, _) in enumerate(self.editor._items)}
        self._icon_rows.clear()
        if not self.editor._items:
            self.clear()
        for row in range(self.editor.image_list.count()):
            self.editor.image_list.item(row).setSizeHint(QSize(98, 102))
        self._thumb_timer.start()

    def clear(self):
        self._selected = self._image = None
        self.details.clear()
        self.position.setText('—')

    def show_image(self, path):
        self._selected = path
        self._image = None
        self.editor.preview_lbl.clear()
        self.editor.preview_lbl.setText('Önizleme yükleniyor…' if self.editor.lang == 'tr' else 'Loading preview…')
        self.details.setText(str(Path(path).name))
        self.position.setText(f'{self.editor._current_idx + 1} / {len(self.editor._items)}')
        cached = self.media.request(path, (1200, 1200))
        if cached:
            self._show_preview(cached)
        self._thumb_timer.start()

    def _show_preview(self, cached):
        image, dimensions, error = cached
        self._image = image
        if image.isNull():
            self.editor.preview_lbl.setText('Önizleme okunamadı' if self.editor.lang == 'tr' else 'Preview unavailable')
            self.details.setText(error)
        else:
            self.details.setText(f'{dimensions[0]} × {dimensions[1]} px  ·  {Path(self._selected).name}')
            self._resize_preview()

    def _resize_preview(self):
        if self._image is not None and not self._image.isNull():
            target = self.editor.preview_lbl.size()
            self.editor.preview_lbl.setPixmap(QPixmap.fromImage(self._image).scaled(
                target, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def request_visible_thumbnails(self):
        view = self.editor.image_list
        if not self.editor.isVisible() or not view.count():
            return
        first = view.indexAt(QPoint(5, 32)).row()
        if first < 0:
            first = max(0, int(view.horizontalScrollBar().value() / 98) - 1)
        wanted = set(range(max(0, first - 1), min(view.count(), first + view.viewport().width() // 98 + 4)))
        for row in self._icon_rows - wanted:
            item = view.item(row)
            if item:
                item.setIcon(QIcon())
        self._icon_rows.intersection_update(wanted)
        for row in sorted(wanted):
            path = self.editor._items[row][0]
            cached = self.media.request(path)
            if cached and not cached[0].isNull():
                view.item(row).setIcon(QIcon(QPixmap.fromImage(cached[0])))
                self._icon_rows.add(row)

    def _received(self, payload):
        key, image, dimensions, error = payload
        if key[0] == self._selected and key[-1] == (1200, 1200):
            self._show_preview((image, dimensions, error))
        elif key[-1] == (96, 96) and key[0] in self._rows:
            self._thumb_timer.start()
        if self._selected and self._image is None:
            cached = self.media.request(self._selected, (1200, 1200))
            if cached:
                self._show_preview(cached)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.Show):
            if obj is self.editor.preview_lbl:
                self._resize_preview()
            else:
                self._thumb_timer.start()
        return False
