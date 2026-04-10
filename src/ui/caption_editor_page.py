"""
Caption Editor Page for LoRA-Harvester
Review and bulk-edit .txt captions alongside their source images.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QFileDialog, QListWidget,
    QListWidgetItem, QInputDialog, QSplitter,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon
from typing import Dict, List, Optional
from src.ui.translations import get_text
from src.ui import theme


_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class CaptionEditorPage(QWidget):
    """
    Page that lets users:
    - Load a folder of images
    - View each image alongside its .txt caption
    - Edit individual captions
    - Bulk add / remove / replace tags across all captions
    - Save all changes to disk
    """

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._folder: Optional[str] = None
        # [(image_path, caption_path)] — caption_path may not exist yet.
        self._items: List[tuple] = []
        # In-memory caption text keyed by caption_path.
        self._captions: Dict[str, str] = {}
        self._current_idx: int = -1
        self._dirty: bool = False
        self._init_ui()

    # ─── UI ──────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(12)
        root.setContentsMargins(20, 16, 20, 16)
        self.setLayout(root)

        # Title
        self.title_lbl = QLabel(get_text('caption_editor_title', self.lang))
        self.title_lbl.setFont(QFont('Arial', 20, QFont.Bold))
        self.title_lbl.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self.title_lbl)

        self.subtitle_lbl = QLabel(get_text('caption_editor_subtitle', self.lang))
        self.subtitle_lbl.setAlignment(Qt.AlignCenter)
        self.subtitle_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self.subtitle_lbl)

        # Toolbar
        toolbar = QHBoxLayout()
        self.load_btn = QPushButton(get_text('ce_load_folder', self.lang))
        self.load_btn.setStyleSheet(theme.btn_primary())
        self.load_btn.clicked.connect(self._browse_folder)
        toolbar.addWidget(self.load_btn)

        self.save_btn = QPushButton(get_text('ce_save_all', self.lang))
        self.save_btn.setStyleSheet(theme.btn_primary())
        self.save_btn.clicked.connect(self._save_all)
        self.save_btn.setEnabled(False)
        toolbar.addWidget(self.save_btn)

        toolbar.addSpacing(20)

        self.add_tag_btn = QPushButton(get_text('ce_add_tag', self.lang))
        self.add_tag_btn.setStyleSheet(theme.btn_secondary())
        self.add_tag_btn.clicked.connect(self._bulk_add_tag)
        self.add_tag_btn.setEnabled(False)
        toolbar.addWidget(self.add_tag_btn)

        self.remove_tag_btn = QPushButton(get_text('ce_remove_tag', self.lang))
        self.remove_tag_btn.setStyleSheet(theme.btn_secondary())
        self.remove_tag_btn.clicked.connect(self._bulk_remove_tag)
        self.remove_tag_btn.setEnabled(False)
        toolbar.addWidget(self.remove_tag_btn)

        self.replace_tag_btn = QPushButton(get_text('ce_replace_tag', self.lang))
        self.replace_tag_btn.setStyleSheet(theme.btn_secondary())
        self.replace_tag_btn.clicked.connect(self._bulk_replace_tag)
        self.replace_tag_btn.setEnabled(False)
        toolbar.addWidget(self.replace_tag_btn)

        toolbar.addStretch()
        root.addLayout(toolbar)

        # Splitter: left = thumbnail list, right = preview + editor
        splitter = QSplitter(Qt.Horizontal)

        # ── Left: image list ────────────────────────────────────────────
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(64, 64))
        self.image_list.setMinimumWidth(200)
        self.image_list.setMaximumWidth(320)
        self.image_list.setStyleSheet(
            f"background-color: {theme.BG_CARD}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 6px;"
        )
        self.image_list.currentRowChanged.connect(self._on_item_selected)
        splitter.addWidget(self.image_list)

        # ── Right: preview + caption editor ─────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 0, 0, 0)

        self.preview_lbl = QLabel()
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setMinimumHeight(300)
        self.preview_lbl.setStyleSheet(
            f"background-color: {theme.BG_DARK}; border-radius: 6px;"
        )
        right_lay.addWidget(self.preview_lbl, stretch=2)

        self.filename_lbl = QLabel("")
        self.filename_lbl.setStyleSheet(
            f"color: {theme.ORANGE_LIGHT}; font-weight: bold; margin-top: 6px;"
        )
        right_lay.addWidget(self.filename_lbl)

        self.caption_edit = QTextEdit()
        self.caption_edit.setMinimumHeight(100)
        self.caption_edit.setMaximumHeight(180)
        self.caption_edit.setStyleSheet(theme.log_area())
        self.caption_edit.textChanged.connect(self._on_caption_changed)
        right_lay.addWidget(self.caption_edit, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter, stretch=1)

        # Status bar
        self.status_lbl = QLabel(get_text('ce_no_images', self.lang))
        self.status_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self.status_lbl)

    # ─── Folder loading ──────────────────────────────────────────────────

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, get_text('ce_select_folder', self.lang))
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder: str):
        self._folder = folder
        self._items.clear()
        self._captions.clear()
        self.image_list.clear()
        self._current_idx = -1

        existing_captions = 0
        for f in sorted(Path(folder).rglob('*')):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS:
                cap_path = f.with_suffix('.txt')
                self._items.append((str(f), str(cap_path)))
                # Pre-load existing caption text
                if cap_path.exists():
                    try:
                        text = cap_path.read_text(encoding='utf-8')
                        self._captions[str(cap_path)] = text
                        existing_captions += 1
                    except Exception:
                        self._captions[str(cap_path)] = ""
                else:
                    self._captions[str(cap_path)] = ""

                # List item with thumbnail
                item = QListWidgetItem(f.name)
                try:
                    pix = QPixmap(str(f))
                    if not pix.isNull():
                        item.setIcon(QIcon(pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                except Exception:
                    pass
                self.image_list.addItem(item)

        has_items = len(self._items) > 0
        self.save_btn.setEnabled(has_items)
        self.add_tag_btn.setEnabled(has_items)
        self.remove_tag_btn.setEnabled(has_items)
        self.replace_tag_btn.setEnabled(has_items)

        self.status_lbl.setText(
            get_text('ce_loaded', self.lang).format(len(self._items), existing_captions)
        )
        if self._items:
            self.image_list.setCurrentRow(0)

    # ─── Item selection ──────────────────────────────────────────────────

    def _on_item_selected(self, row: int):
        # Save current before switching
        if self._current_idx >= 0:
            self._commit_current()

        if row < 0 or row >= len(self._items):
            self.preview_lbl.clear()
            self.caption_edit.clear()
            self.filename_lbl.setText("")
            self._current_idx = -1
            return

        self._current_idx = row
        img_path, cap_path = self._items[row]

        # Preview
        try:
            pix = QPixmap(img_path)
            if not pix.isNull():
                self.preview_lbl.setPixmap(
                    pix.scaled(
                        self.preview_lbl.width(), self.preview_lbl.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation,
                    )
                )
        except Exception:
            self.preview_lbl.setText("(preview error)")

        self.filename_lbl.setText(Path(img_path).name)

        # Load caption text without triggering textChanged
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(self._captions.get(cap_path, ""))
        self.caption_edit.blockSignals(False)

    def _commit_current(self):
        """Store the current editor text back into the in-memory dict."""
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self._captions[cap_path] = self.caption_edit.toPlainText()

    def _on_caption_changed(self):
        self._dirty = True

    # ─── Save ────────────────────────────────────────────────────────────

    def _save_all(self):
        self._commit_current()
        for cap_path, text in self._captions.items():
            try:
                Path(cap_path).write_text(text, encoding='utf-8')
            except Exception:
                pass
        self._dirty = False
        self.status_lbl.setText(get_text('ce_saved', self.lang))

    # ─── Bulk operations ─────────────────────────────────────────────────

    def _bulk_add_tag(self):
        tag, ok = QInputDialog.getText(
            self, get_text('ce_add_tag_title', self.lang),
            get_text('ce_add_tag_prompt', self.lang))
        if not ok or not tag.strip():
            return
        tag = tag.strip()
        self._commit_current()
        for cap_path in list(self._captions):
            text = self._captions[cap_path].strip()
            if text:
                self._captions[cap_path] = f"{tag}, {text}"
            else:
                self._captions[cap_path] = tag
        # Refresh visible editor
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(self._captions.get(cap_path, ""))
            self.caption_edit.blockSignals(False)
        self.status_lbl.setText(
            get_text('ce_add_tag_result', self.lang).format(tag, len(self._captions)))

    def _bulk_remove_tag(self):
        tag, ok = QInputDialog.getText(
            self, get_text('ce_remove_tag_title', self.lang),
            get_text('ce_remove_tag_prompt', self.lang))
        if not ok or not tag.strip():
            return
        tag = tag.strip()
        self._commit_current()
        removed = 0
        for cap_path in list(self._captions):
            parts = [t.strip() for t in self._captions[cap_path].split(',')]
            new_parts = [t for t in parts if t.lower() != tag.lower()]
            if len(new_parts) != len(parts):
                removed += 1
            self._captions[cap_path] = ', '.join(new_parts)
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(self._captions.get(cap_path, ""))
            self.caption_edit.blockSignals(False)
        self.status_lbl.setText(
            get_text('ce_remove_tag_result', self.lang).format(tag, removed))

    def _bulk_replace_tag(self):
        old, ok1 = QInputDialog.getText(
            self, get_text('ce_replace_tag_title', self.lang),
            get_text('ce_replace_find_prompt', self.lang))
        if not ok1 or not old.strip():
            return
        new, ok2 = QInputDialog.getText(
            self, get_text('ce_replace_tag_title', self.lang),
            get_text('ce_replace_with_prompt', self.lang).format(old.strip()))
        if not ok2:
            return
        old = old.strip()
        new = new.strip()
        self._commit_current()
        replaced = 0
        for cap_path in list(self._captions):
            parts = [t.strip() for t in self._captions[cap_path].split(',')]
            new_parts = [(new if t.lower() == old.lower() else t) for t in parts]
            if new_parts != parts:
                replaced += 1
            self._captions[cap_path] = ', '.join(p for p in new_parts if p)
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(self._captions.get(cap_path, ""))
            self.caption_edit.blockSignals(False)
        self.status_lbl.setText(
            get_text('ce_replace_tag_result', self.lang).format(old, new, replaced))

    # ─── Language ────────────────────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self.title_lbl.setText(get_text('caption_editor_title', lang))
        self.subtitle_lbl.setText(get_text('caption_editor_subtitle', lang))
        self.load_btn.setText(get_text('ce_load_folder', lang))
        self.save_btn.setText(get_text('ce_save_all', lang))
        self.add_tag_btn.setText(get_text('ce_add_tag', lang))
        self.remove_tag_btn.setText(get_text('ce_remove_tag', lang))
        self.replace_tag_btn.setText(get_text('ce_replace_tag', lang))
