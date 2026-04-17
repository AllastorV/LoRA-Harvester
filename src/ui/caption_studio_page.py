"""
Caption Studio Page — unified captioning + editing with Danbooru autocomplete.
Provides Generate (WD14 tagging) and Edit (with Danbooru autocomplete) tabs.
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox,
    QTextEdit, QPushButton, QProgressBar, QFileDialog, QFrame,
    QListWidget, QListWidgetItem, QInputDialog, QSplitter,
    QTabWidget, QCompleter, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QStringListModel
from PyQt5.QtGui import (
    QFont, QDragEnterEvent, QDropEvent, QPixmap, QIcon, QTextCursor,
)
from typing import Dict, List, Optional
from src.ui.translations import get_text
from src.ui import theme

_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


# ════════════════════════════════════════════════════════════════
#  Background tag loader (runs once, result shared across widgets)
# ════════════════════════════════════════════════════════════════

class _TagLoadThread(QThread):
    """Load Danbooru tags off the UI thread."""
    loaded = pyqtSignal(list)

    def run(self):
        try:
            from src.core.tag_autocomplete import load_danbooru_tags
            tags = load_danbooru_tags()
        except Exception:
            tags = []
        self.loaded.emit(tags)


_shared_tag_model: Optional[QStringListModel] = None
_shared_tag_load_started = False


# ════════════════════════════════════════════════════════════════
#  QTextEdit with Danbooru tag autocomplete popup
# ════════════════════════════════════════════════════════════════

class TagCompleterTextEdit(QTextEdit):
    """QTextEdit that shows a Danbooru-tag popup under the cursor.

    - Prefix = text between the last comma/newline and the cursor.
    - Selecting a suggestion inserts the full tag + ', '.
    - Tags load lazily on first focus via a background thread.
    """

    tags_loaded = pyqtSignal(int)
    tags_load_failed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._completer: Optional[QCompleter] = None
        self._loader: Optional[_TagLoadThread] = None
        self._setup_completer()

    def _setup_completer(self):
        global _shared_tag_model
        if _shared_tag_model is None:
            _shared_tag_model = QStringListModel([])

        c = QCompleter(_shared_tag_model, self)
        c.setWidget(self)
        c.setCompletionMode(QCompleter.PopupCompletion)
        c.setCaseSensitivity(Qt.CaseInsensitive)
        c.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
        c.activated[str].connect(self._insert_completion)
        self._completer = c

    # ── Lazy tag loading ────────────────────────────────────────

    def _ensure_tags_loaded(self):
        global _shared_tag_model, _shared_tag_load_started
        if _shared_tag_load_started:
            return
        if _shared_tag_model and _shared_tag_model.rowCount() > 0:
            return
        _shared_tag_load_started = True
        self._loader = _TagLoadThread()
        self._loader.loaded.connect(self._on_tags_loaded)
        self._loader.start()

    def _on_tags_loaded(self, tags: list):
        global _shared_tag_model
        if _shared_tag_model is not None:
            _shared_tag_model.setStringList(tags)
        if tags:
            self.tags_loaded.emit(len(tags))
        else:
            self.tags_load_failed.emit()
        if self._loader:
            self._loader.deleteLater()
            self._loader = None

    def focusInEvent(self, event):
        self._ensure_tags_loaded()
        super().focusInEvent(event)

    # ── Prefix extraction ───────────────────────────────────────

    def _text_under_cursor(self):
        """Return (word, start_pos) — the text between the last
        comma/newline and the cursor position."""
        cursor = self.textCursor()
        pos = cursor.position()
        text = self.toPlainText()
        start = pos
        while start > 0 and text[start - 1] not in ',\n':
            start -= 1
        while start < pos and text[start] == ' ':
            start += 1
        return text[start:pos], start

    # ── Completion insertion ────────────────────────────────────

    def _insert_completion(self, completion: str):
        word, start = self._text_under_cursor()
        tc = self.textCursor()
        tc.setPosition(start)
        tc.setPosition(tc.position() + len(word), QTextCursor.KeepAnchor)
        tc.insertText(completion + ', ')

    # ── Key handling ────────────────────────────────────────────

    def keyPressEvent(self, event):
        if self._completer and self._completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Tab,
                               Qt.Key_Backtab, Qt.Key_Escape):
                event.ignore()
                return

        super().keyPressEvent(event)

        if not self._completer:
            return
        if event.modifiers() in (Qt.ControlModifier, Qt.AltModifier, Qt.MetaModifier):
            if not event.text():
                return

        word, _ = self._text_under_cursor()
        if len(word) < 2:
            self._completer.popup().hide()
            return

        model = self._completer.model()
        if model is None or model.rowCount() == 0:
            return

        if word != self._completer.completionPrefix():
            self._completer.setCompletionPrefix(word)
            popup = self._completer.popup()
            popup.setCurrentIndex(
                self._completer.completionModel().index(0, 0))

        cr = self.cursorRect()
        cr.setWidth(
            self._completer.popup().sizeHintForColumn(0)
            + self._completer.popup().verticalScrollBar().sizeHint().width()
        )
        self._completer.complete(cr)


# ════════════════════════════════════════════════════════════════
#  Captioning background thread
# ════════════════════════════════════════════════════════════════

class CaptioningThread(QThread):
    """Background thread for captioning."""
    progress = pyqtSignal(int, int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, captioner, image_folder: str, settings: Dict):
        super().__init__()
        self.captioner = captioner
        self.image_folder = image_folder
        self.settings = settings
        self._running = True
        self._last_progress_pct = -1

    def run(self):
        try:
            if self.captioner.wd14 and self.captioner.enable_wd14:
                try:
                    self.log_message.emit("Loading WD14 model...")
                    self.captioner.wd14._load_model()
                    n_tags = len(self.captioner.wd14.tags) if self.captioner.wd14.tags else 0
                    self.log_message.emit(f"✅ WD14 model loaded ({n_tags} tags)")
                    if n_tags == 0:
                        self.log_message.emit("⚠️ WD14 tag list is empty — auto-tags will NOT be generated!")
                except Exception as e:
                    self.log_message.emit(f"❌ WD14 FAILED: {e}")
                    self.log_message.emit("⚠️ Auto-tagging disabled — captions will only contain trigger word!")
                    self.captioner.enable_wd14 = False
            elif self.captioner.enable_wd14:
                self.log_message.emit("⚠️ WD14 tagger not initialized — check model selection")

            if not self._running:
                return

            stats = self.captioner.caption_directory(
                self.image_folder,
                mode=self.settings.get('mode', 'tags_only'),
                overwrite=self.settings.get('overwrite', False),
                save_json=self.settings.get('save_json', False),
                progress_callback=self._progress_callback,
                recursive=self.settings.get('recursive', False),
            )
            if self._running:
                self.finished.emit(stats)
        except Exception as e:
            if self._running:
                self.error.emit(str(e))
        finally:
            try:
                if self.captioner and hasattr(self.captioner, 'cleanup'):
                    self.captioner.cleanup()
            except Exception:
                pass

    def _progress_callback(self, current: int, total: int, filename: str):
        if not self._running:
            return False
        pct = int((current / total) * 100) if total > 0 else 0
        if pct != self._last_progress_pct:
            self._last_progress_pct = pct
            self.progress.emit(current, total, filename)
        return True

    def stop(self):
        self._running = False
        self.requestInterruption()


# ════════════════════════════════════════════════════════════════
#  Generate Tab (WD14 tagging)
# ════════════════════════════════════════════════════════════════

class _GenerateTab(QWidget):
    """Auto-caption generation using WD14 tagger."""

    folder_changed = pyqtSignal(str)
    captioning_finished = pyqtSignal(str)

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self.selected_folder = None
        self.captioner = None
        self.captioning_thread = None
        self._init_ui()

    def _init_ui(self):
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 12, 20, 12)

        # ── Step 1: Folder ──────────────────────────────────────
        step1_frame = QFrame()
        step1_frame.setStyleSheet(theme.card_frame())
        step1_layout = QVBoxLayout(step1_frame)

        self.step1_title = QLabel(get_text('step1_select_folder', self.lang))
        self.step1_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step1_title.setStyleSheet(theme.label_section())
        step1_layout.addWidget(self.step1_title)

        folder_row = QHBoxLayout()
        self.drop_zone = QFrame()
        self.drop_zone.setMinimumHeight(65)
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())
        drop_layout = QHBoxLayout(self.drop_zone)
        drop_layout.setContentsMargins(15, 5, 15, 5)
        drop_layout.setAlignment(Qt.AlignCenter)

        self.drop_icon = QLabel("📂")
        self.drop_icon.setStyleSheet(theme.icon_transparent())
        self.drop_icon.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(self.drop_icon)

        self.folder_label = QLabel(get_text('drag_drop_folder', self.lang))
        self.folder_label.setStyleSheet(theme.label_transparent())
        self.folder_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        drop_layout.addWidget(self.folder_label, stretch=1)
        folder_row.addWidget(self.drop_zone, stretch=1)

        self.browse_btn = QPushButton(get_text('select_input_folder', self.lang))
        self.browse_btn.setStyleSheet(theme.btn_browse())
        self.browse_btn.clicked.connect(self.select_folder)
        folder_row.addWidget(self.browse_btn)
        step1_layout.addLayout(folder_row)

        opts_row = QHBoxLayout()
        self.recursive_cb = QCheckBox(get_text('recursive_search', self.lang))
        self.recursive_cb.setToolTip(get_text('recursive_tooltip', self.lang))
        self.overwrite_cb = QCheckBox(get_text('overwrite_existing', self.lang))
        self.overwrite_cb.setToolTip(get_text('overwrite_tooltip', self.lang))
        opts_row.addWidget(self.recursive_cb)
        opts_row.addWidget(self.overwrite_cb)
        opts_row.addStretch()
        step1_layout.addLayout(opts_row)

        self.image_count_label = QLabel("")
        self.image_count_label.setStyleSheet(theme.label_success())
        step1_layout.addWidget(self.image_count_label)
        layout.addWidget(step1_frame)

        # ── Step 2: Settings (WD14) ─────────────────────────────
        step2_frame = QFrame()
        step2_frame.setStyleSheet(theme.card_frame())
        step2_layout = QVBoxLayout(step2_frame)
        step2_layout.setSpacing(8)
        step2_layout.setContentsMargins(10, 8, 10, 8)

        self.step2_title = QLabel(get_text('step2_settings', self.lang))
        self.step2_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step2_title.setStyleSheet(theme.label_section())
        step2_layout.addWidget(self.step2_title)

        # Quality preset (auto-selects model + confidence + max_tags)
        preset_row = QHBoxLayout()
        self.preset_label = QLabel(get_text('preset_label', self.lang))
        self.preset_label.setStyleSheet(theme.label_frame())
        self.preset_info = QLabel("ℹ️")
        self.preset_info.setToolTip(get_text('preset_tooltip', self.lang))
        self.preset_info.setStyleSheet(theme.info_icon_frame())
        self.preset_info.setCursor(Qt.WhatsThisCursor)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem(get_text('preset_high_accuracy', self.lang), 'high_accuracy')
        self.preset_combo.addItem(get_text('preset_balanced', self.lang), 'balanced')
        self.preset_combo.addItem(get_text('preset_high_speed', self.lang), 'high_speed')
        self.preset_combo.addItem(get_text('preset_custom', self.lang), 'custom')
        self.preset_combo.setStyleSheet(theme.combo())
        self.preset_combo.setMinimumWidth(200)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_label)
        preset_row.addWidget(self.preset_info)
        preset_row.addWidget(self.preset_combo)
        preset_row.addStretch()
        step2_layout.addLayout(preset_row)

        # Trigger word (prepended)
        row1 = QHBoxLayout()
        self.trigger_label = QLabel(get_text('trigger_word', self.lang))
        self.trigger_label.setStyleSheet(theme.label_frame())
        self.trigger_info = QLabel("ℹ️")
        self.trigger_info.setToolTip(get_text('trigger_tooltip', self.lang))
        self.trigger_info.setStyleSheet(theme.info_icon_frame())
        self.trigger_info.setCursor(Qt.WhatsThisCursor)
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("sks person...")
        self.trigger_edit.setStyleSheet(theme.line_edit())
        self.trigger_edit.setMinimumWidth(150)
        row1.addWidget(self.trigger_label)
        row1.addWidget(self.trigger_info)
        row1.addWidget(self.trigger_edit)
        row1.addStretch()
        step2_layout.addLayout(row1)

        # Suffix tags (appended at end of caption)
        suffix_row = QHBoxLayout()
        self.suffix_label = QLabel(get_text('caption_suffix_label', self.lang))
        self.suffix_label.setStyleSheet(theme.label_frame())
        self.suffix_info = QLabel("ℹ️")
        self.suffix_info.setToolTip(get_text('caption_suffix_tooltip', self.lang))
        self.suffix_info.setStyleSheet(theme.info_icon_frame())
        self.suffix_info.setCursor(Qt.WhatsThisCursor)
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("masterpiece, best quality...")
        self.suffix_edit.setStyleSheet(theme.line_edit())
        self.suffix_edit.setMinimumWidth(150)
        suffix_row.addWidget(self.suffix_label)
        suffix_row.addWidget(self.suffix_info)
        suffix_row.addWidget(self.suffix_edit)
        suffix_row.addStretch()
        step2_layout.addLayout(suffix_row)

        # Max tags & Confidence
        row2 = QHBoxLayout()
        self.max_label = QLabel(get_text('max_tags', self.lang))
        self.max_label.setStyleSheet(theme.label_frame())
        self.max_info = QLabel("ℹ️")
        self.max_info.setToolTip(get_text('max_tags_tooltip', self.lang))
        self.max_info.setStyleSheet(theme.info_icon_frame())
        self.max_info.setCursor(Qt.WhatsThisCursor)
        self.max_tags_spin = QSpinBox()
        self.max_tags_spin.setRange(5, 100)
        self.max_tags_spin.setValue(25)
        self.max_tags_spin.setStyleSheet(theme.spinbox())

        self.conf_label = QLabel(get_text('min_confidence', self.lang))
        self.conf_label.setStyleSheet(theme.label_frame())
        self.conf_info = QLabel("ℹ️")
        self.conf_info.setToolTip(get_text('confidence_tooltip', self.lang))
        self.conf_info.setStyleSheet(theme.info_icon_frame())
        self.conf_info.setCursor(Qt.WhatsThisCursor)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.9)
        self.conf_spin.setValue(0.35)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setStyleSheet(theme.spinbox())

        row2.addWidget(self.max_label)
        row2.addWidget(self.max_info)
        row2.addWidget(self.max_tags_spin)
        row2.addSpacing(20)
        row2.addWidget(self.conf_label)
        row2.addWidget(self.conf_info)
        row2.addWidget(self.conf_spin)
        row2.addStretch()
        step2_layout.addLayout(row2)

        # Negative tags
        neg_row = QHBoxLayout()
        self.neg_label = QLabel(get_text('negative_tags', self.lang))
        self.neg_label.setStyleSheet(theme.label_frame())
        self.neg_info = QLabel("ℹ️")
        self.neg_info.setToolTip(get_text('negative_tooltip', self.lang))
        self.neg_info.setStyleSheet(theme.info_icon_frame_compact())
        self.neg_info.setCursor(Qt.WhatsThisCursor)
        neg_row.addWidget(self.neg_label)
        neg_row.addWidget(self.neg_info)
        neg_row.addStretch()
        step2_layout.addLayout(neg_row)

        self.neg_edit = QLineEdit()
        self.neg_edit.setText(
            "watermark, signature, text, username, artist_name, "
            "twitter_username, patreon_username, dated"
        )
        self.neg_edit.setStyleSheet(theme.line_edit())
        step2_layout.addWidget(self.neg_edit)

        # WD14 model selection (hidden unless preset = Custom)
        model_row = QHBoxLayout()
        self.wd14_cb = QCheckBox(get_text('use_wd14', self.lang))
        self.wd14_cb.setChecked(True)
        self.wd14_cb.setStyleSheet(theme.checkbox_frame())
        self.wd14_info = QLabel("ℹ️")
        self.wd14_info.setToolTip(get_text('wd14_tooltip', self.lang))
        self.wd14_info.setStyleSheet(theme.info_icon_frame())
        self.wd14_info.setCursor(Qt.WhatsThisCursor)
        self.wd14_combo = QComboBox()
        self.wd14_combo.addItems([
            'SmilingWolf/wd-swinv2-tagger-v3',
            'SmilingWolf/wd-convnext-tagger-v3',
            'SmilingWolf/wd-vit-tagger-v3',
            'SmilingWolf/wd-v1-4-moat-tagger-v2',
            'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        ])
        self.wd14_combo.setStyleSheet(theme.combo())
        self.wd14_combo.setMinimumWidth(280)
        self.wd14_combo.setToolTip(get_text('wd14_model_tooltip', self.lang))
        self._model_row_widget = QFrame()
        _mrl = QHBoxLayout(self._model_row_widget)
        _mrl.setContentsMargins(0, 0, 0, 0)
        _mrl.addWidget(self.wd14_cb)
        _mrl.addWidget(self.wd14_info)
        _mrl.addWidget(self.wd14_combo)
        _mrl.addStretch()
        model_row.addWidget(self._model_row_widget)
        step2_layout.addLayout(model_row)

        # Checkboxes
        cb_row = QHBoxLayout()
        self.keep_char_cb = QCheckBox(get_text('keep_character_tags', self.lang))
        self.keep_char_cb.setChecked(True)
        self.keep_char_cb.setToolTip(get_text('keep_char_tooltip', self.lang))
        self.json_cb = QCheckBox(get_text('save_json', self.lang))
        self.json_cb.setToolTip(get_text('json_tooltip', self.lang))
        cb_row.addWidget(self.keep_char_cb)
        cb_row.addWidget(self.json_cb)
        cb_row.addStretch()
        step2_layout.addLayout(cb_row)
        layout.addWidget(step2_frame)

        # ── Step 3: Start / Stop / Progress ─────────────────────
        step3_frame = QFrame()
        step3_frame.setStyleSheet(theme.card_frame())
        step3_layout = QVBoxLayout(step3_frame)

        self.step3_title = QLabel(get_text('step3_start', self.lang))
        self.step3_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step3_title.setStyleSheet(theme.label_section())
        step3_layout.addWidget(self.step3_title)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton(get_text('start_captioning', self.lang))
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(theme.btn_primary())
        self.start_btn.clicked.connect(self.start_captioning)
        self.stop_btn = QPushButton(get_text('stop_btn', self.lang))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.stop_btn.clicked.connect(self.stop_captioning)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        step3_layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(theme.progress_bar())
        step3_layout.addWidget(self.progress_bar)
        layout.addWidget(step3_frame)

        # ── Log ─────────────────────────────────────────────────
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(105)
        self.log_text.setMaximumHeight(225)
        self.log_text.setStyleSheet(theme.log_area())
        layout.addWidget(self.log_text)

        layout.addStretch()
        self.setLayout(layout)

        # Apply default preset (Balanced) so model/confidence/max_tags are synced
        self.preset_combo.setCurrentIndex(1)
        self._on_preset_changed(1)

    # ── Helpers ─────────────────────────────────────────────────

    def log(self, message: str):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                if os.path.isdir(urls[0].toLocalFile()):
                    event.acceptProposedAction()
                    self.drop_zone.setStyleSheet(theme.drop_zone_frame_active())
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    self._set_folder(path)
                    return
        event.ignore()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self._set_folder(folder)

    def _set_folder(self, folder: str):
        self.selected_folder = folder
        display = folder if len(folder) <= 50 else "..." + folder[-47:]
        self.folder_label.setText(display)
        self.folder_label.setStyleSheet(theme.label_success())
        self.drop_icon.setText("✅")
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_success())
        self.drop_zone.setToolTip(folder)

        count = self._count_images(folder)
        self.image_count_label.setText(
            get_text('images_found', self.lang).format(count))
        self.start_btn.setEnabled(count > 0)
        self.log(f"📁 {get_text('folder_selected', self.lang)}: {folder}")
        self.log(f"🖼️ {count} images found")
        self.folder_changed.emit(folder)

    def _count_images(self, folder: str) -> int:
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        count = 0
        p = Path(folder)
        glob = p.rglob if self.recursive_cb.isChecked() else p.glob
        for ext in extensions:
            count += len(list(glob(f'*{ext}')))
        return count

    # ── Preset handling ─────────────────────────────────────────

    # Preset → (wd14_model, min_confidence, max_tags)
    _PRESETS = {
        'high_accuracy': ('SmilingWolf/wd-swinv2-tagger-v3', 0.30, 40),
        'balanced':      ('SmilingWolf/wd-convnext-tagger-v3', 0.35, 30),
        'high_speed':    ('SmilingWolf/wd-vit-tagger-v3', 0.40, 20),
    }

    def _on_preset_changed(self, index: int):
        """Apply preset values to model/confidence/max_tags.
        'custom' unlocks the manual model combo."""
        key = self.preset_combo.itemData(index)
        if key == 'custom':
            self._model_row_widget.setVisible(True)
            return
        preset = self._PRESETS.get(key)
        if not preset:
            return
        model, conf, mtags = preset
        idx = self.wd14_combo.findText(model)
        if idx >= 0:
            self.wd14_combo.setCurrentIndex(idx)
        self.conf_spin.setValue(conf)
        self.max_tags_spin.setValue(mtags)
        self._model_row_widget.setVisible(False)

    def get_settings(self) -> Dict:
        return {
            'mode': 'tags_only',
            'trigger_word': self.trigger_edit.text().strip(),
            'caption_suffix': self.suffix_edit.text().strip(),
            'max_tags': self.max_tags_spin.value(),
            'min_confidence': self.conf_spin.value(),
            'negative_tags': [t.strip() for t in self.neg_edit.text().split(',') if t.strip()],
            'keep_character_tags': self.keep_char_cb.isChecked(),
            'save_json': self.json_cb.isChecked(),
            'overwrite': self.overwrite_cb.isChecked(),
            'recursive': self.recursive_cb.isChecked(),
            'use_wd14': self.wd14_cb.isChecked(),
            'wd14_model': self.wd14_combo.currentText(),
        }

    # ── Captioning control ──────────────────────────────────────

    def start_captioning(self):
        if not self.selected_folder:
            return
        self.log(f"\n{'=' * 40}")
        self.log("🚀 " + get_text('log_captioning', self.lang))
        settings = self.get_settings()
        if settings['trigger_word']:
            self.log(f"   Trigger: {settings['trigger_word']}")
        try:
            from src.core.advanced_captioner import AdvancedCaptioner, TagSettings
            tag_settings = TagSettings(
                trigger_word=settings['trigger_word'] or "",
                caption_suffix=settings.get('caption_suffix', '') or "",
                max_tags=settings['max_tags'],
                min_confidence=settings['min_confidence'],
                negative_tags=settings['negative_tags'],
                keep_character_tags=settings['keep_character_tags'],
            )
            self.captioner = AdvancedCaptioner(
                tag_settings=tag_settings,
                wd14_model=settings['wd14_model'],
                enable_wd14=settings['use_wd14'],
            )
            self.log("✅ Captioner initialized")
            if settings.get('caption_suffix'):
                self.log(f"   Suffix: {settings['caption_suffix']}")
        except Exception as e:
            self.log(f"❌ Error: {e}")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)

        self.captioning_thread = CaptioningThread(
            self.captioner, self.selected_folder, settings)
        self.captioning_thread.progress.connect(self._on_progress)
        self.captioning_thread.log_message.connect(self.log)
        self.captioning_thread.finished.connect(self._on_finished)
        self.captioning_thread.error.connect(self._on_error)
        self.captioning_thread.finished.connect(self._safe_delete_thread)
        self.captioning_thread.start()

    def stop_captioning(self):
        if self.captioning_thread and self.captioning_thread.isRunning():
            self.captioning_thread.stop()
            self.log("Stopping...")
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.browse_btn.setEnabled(True)

    def _on_progress(self, current: int, total: int, filename: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{current}/{total} - {filename}")

    def _on_finished(self, stats: Dict):
        self.progress_bar.setValue(100)
        self.log(f"\n{'=' * 40}")
        self.log(f"✅ Complete! Captioned: {stats.get('captioned', 0)} images")
        self.log(f"   Skipped: {stats.get('skipped', 0)}")
        self.log(f"   Errors: {stats.get('errors', 0)}")
        zero_tags = stats.get('zero_tags', 0)
        if zero_tags > 0:
            self.log(f"   ⚠️ Zero auto-tags: {zero_tags} images")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        if self.selected_folder:
            self.captioning_finished.emit(self.selected_folder)

    def _on_error(self, error: str):
        self.log(f"❌ Error: {error}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

    def _safe_delete_thread(self):
        """Single cleanup path — called via finished signal after _on_finished."""
        if self.captioning_thread:
            self.captioning_thread.wait(3000)
            self.captioning_thread.deleteLater()
            self.captioning_thread = None
        if self.captioner:
            try:
                self.captioner.cleanup()
            except Exception:
                pass
            self.captioner = None

    # ── Language / Theme ────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self.step1_title.setText(get_text('step1_select_folder', lang))
        self.step2_title.setText(get_text('step2_settings', lang))
        self.step3_title.setText(get_text('step3_start', lang))
        self.browse_btn.setText(get_text('select_input_folder', lang))
        self.start_btn.setText(get_text('start_captioning', lang))
        self.stop_btn.setText(get_text('stop_btn', lang))
        if not self.selected_folder:
            self.folder_label.setText(get_text('drag_drop_folder', lang))
        self.preset_label.setText(get_text('preset_label', lang))
        self.preset_info.setToolTip(get_text('preset_tooltip', lang))
        # Preserve selection index while refreshing item labels
        cur_idx = self.preset_combo.currentIndex()
        self.preset_combo.blockSignals(True)
        self.preset_combo.setItemText(0, get_text('preset_high_accuracy', lang))
        self.preset_combo.setItemText(1, get_text('preset_balanced', lang))
        self.preset_combo.setItemText(2, get_text('preset_high_speed', lang))
        self.preset_combo.setItemText(3, get_text('preset_custom', lang))
        self.preset_combo.setCurrentIndex(cur_idx)
        self.preset_combo.blockSignals(False)
        self.trigger_label.setText(get_text('trigger_word', lang))
        self.suffix_label.setText(get_text('caption_suffix_label', lang))
        self.suffix_info.setToolTip(get_text('caption_suffix_tooltip', lang))
        self.max_label.setText(get_text('max_tags', lang))
        self.conf_label.setText(get_text('min_confidence', lang))
        self.neg_label.setText(get_text('negative_tags', lang))
        self.recursive_cb.setText(get_text('recursive_search', lang))
        self.overwrite_cb.setText(get_text('overwrite_existing', lang))
        self.keep_char_cb.setText(get_text('keep_character_tags', lang))
        self.json_cb.setText(get_text('save_json', lang))
        self.wd14_cb.setText(get_text('use_wd14', lang))
        self.trigger_info.setToolTip(get_text('trigger_tooltip', lang))
        self.max_info.setToolTip(get_text('max_tags_tooltip', lang))
        self.conf_info.setToolTip(get_text('confidence_tooltip', lang))
        self.neg_info.setToolTip(get_text('negative_tooltip', lang))
        self.wd14_info.setToolTip(get_text('wd14_tooltip', lang))
        self.wd14_combo.setToolTip(get_text('wd14_model_tooltip', lang))
        self.keep_char_cb.setToolTip(get_text('keep_char_tooltip', lang))
        self.json_cb.setToolTip(get_text('json_tooltip', lang))
        self.recursive_cb.setToolTip(get_text('recursive_tooltip', lang))
        self.overwrite_cb.setToolTip(get_text('overwrite_tooltip', lang))

    def refresh_styles(self):
        for frame in self.findChildren(QFrame):
            if frame is self.drop_zone:
                continue
            ss = frame.styleSheet()
            if "border-radius: 10px" in ss:
                frame.setStyleSheet(theme.card_frame())
        self.step1_title.setStyleSheet(theme.label_section())
        if self.selected_folder:
            self.drop_zone.setStyleSheet(theme.drop_zone_frame_success())
            self.folder_label.setStyleSheet(theme.label_success())
        else:
            self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())
            self.folder_label.setStyleSheet(theme.label_transparent())
        self.drop_icon.setStyleSheet(theme.icon_transparent())
        self.browse_btn.setStyleSheet(theme.btn_browse())
        self.image_count_label.setStyleSheet(theme.label_success())
        self.step2_title.setStyleSheet(theme.label_section())
        self.preset_label.setStyleSheet(theme.label_frame())
        self.preset_info.setStyleSheet(theme.info_icon_frame())
        self.preset_combo.setStyleSheet(theme.combo())
        self.trigger_label.setStyleSheet(theme.label_frame())
        self.trigger_info.setStyleSheet(theme.info_icon_frame())
        self.trigger_edit.setStyleSheet(theme.line_edit())
        self.suffix_label.setStyleSheet(theme.label_frame())
        self.suffix_info.setStyleSheet(theme.info_icon_frame())
        self.suffix_edit.setStyleSheet(theme.line_edit())
        self.max_label.setStyleSheet(theme.label_frame())
        self.max_info.setStyleSheet(theme.info_icon_frame())
        self.max_tags_spin.setStyleSheet(theme.spinbox())
        self.conf_label.setStyleSheet(theme.label_frame())
        self.conf_info.setStyleSheet(theme.info_icon_frame())
        self.conf_spin.setStyleSheet(theme.spinbox())
        self.neg_label.setStyleSheet(theme.label_frame())
        self.neg_info.setStyleSheet(theme.info_icon_frame_compact())
        self.neg_edit.setStyleSheet(theme.line_edit())
        self._model_row_widget.setStyleSheet(
            f"background-color: transparent;")
        self.wd14_cb.setStyleSheet(theme.checkbox_frame())
        self.wd14_info.setStyleSheet(theme.info_icon_frame())
        self.wd14_combo.setStyleSheet(theme.combo())
        self.step3_title.setStyleSheet(theme.label_section())
        self.start_btn.setStyleSheet(theme.btn_primary())
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.progress_bar.setStyleSheet(theme.progress_bar())
        self.log_text.setStyleSheet(theme.log_area())


# ════════════════════════════════════════════════════════════════
#  Edit Tab (ported from CaptionEditorPage, with TagCompleterTextEdit)
# ════════════════════════════════════════════════════════════════

class _EditTab(QWidget):
    """Browse images, review/edit captions, bulk tag operations."""

    folder_changed = pyqtSignal(str)

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._folder: Optional[str] = None
        self._items: List[tuple] = []
        self._captions: Dict[str, str] = {}
        self._current_idx: int = -1
        self._dirty: bool = False
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(12)
        root.setContentsMargins(20, 12, 20, 12)
        self.setLayout(root)

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

        # TagCompleterTextEdit instead of plain QTextEdit
        self.caption_edit = TagCompleterTextEdit()
        self.caption_edit.setMinimumHeight(100)
        self.caption_edit.setMaximumHeight(180)
        self.caption_edit.setStyleSheet(theme.log_area())
        self.caption_edit.textChanged.connect(self._on_caption_changed)
        right_lay.addWidget(self.caption_edit, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)

        self.status_lbl = QLabel(get_text('ce_no_images', self.lang))
        self.status_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self.status_lbl)

    # ── Folder loading ──────────────────────────────────────────

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, get_text('ce_select_folder', self.lang))
        if folder:
            self._load_folder(folder)

    def reload_folder(self, folder: str):
        """Public entry point — reload without dialog."""
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
                if cap_path.exists():
                    try:
                        self._captions[str(cap_path)] = cap_path.read_text(encoding='utf-8')
                        existing_captions += 1
                    except Exception:
                        self._captions[str(cap_path)] = ""
                else:
                    self._captions[str(cap_path)] = ""
                item = QListWidgetItem(f.name)
                try:
                    pix = QPixmap(str(f))
                    if not pix.isNull():
                        item.setIcon(QIcon(pix.scaled(
                            64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                except Exception:
                    pass
                self.image_list.addItem(item)

        has = len(self._items) > 0
        self.save_btn.setEnabled(has)
        self.add_tag_btn.setEnabled(has)
        self.remove_tag_btn.setEnabled(has)
        self.replace_tag_btn.setEnabled(has)
        self.status_lbl.setText(
            get_text('ce_loaded', self.lang).format(len(self._items), existing_captions))
        if self._items:
            self.image_list.setCurrentRow(0)
        self.folder_changed.emit(folder)

    # ── Item selection ──────────────────────────────────────────

    def _on_item_selected(self, row: int):
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
        try:
            pix = QPixmap(img_path)
            if not pix.isNull():
                self.preview_lbl.setPixmap(pix.scaled(
                    self.preview_lbl.width(), self.preview_lbl.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            self.preview_lbl.setText("(preview error)")
        self.filename_lbl.setText(Path(img_path).name)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(self._captions.get(cap_path, ""))
        self.caption_edit.blockSignals(False)

    def _commit_current(self):
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self._captions[cap_path] = self.caption_edit.toPlainText()

    def _on_caption_changed(self):
        self._dirty = True

    # ── Save ────────────────────────────────────────────────────

    def _save_all(self):
        self._commit_current()
        for cap_path, text in self._captions.items():
            try:
                Path(cap_path).write_text(text, encoding='utf-8')
            except Exception:
                pass
        self._dirty = False
        self.status_lbl.setText(get_text('ce_saved', self.lang))

    # ── Bulk operations ─────────────────────────────────────────

    def _refresh_editor(self):
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(self._captions.get(cap_path, ""))
            self.caption_edit.blockSignals(False)

    def _bulk_add_tag(self):
        tag, ok = QInputDialog.getText(
            self, get_text('ce_add_tag_title', self.lang),
            get_text('ce_add_tag_prompt', self.lang))
        if not ok or not tag.strip():
            return
        tag = tag.strip()
        self._commit_current()
        for cp in list(self._captions):
            text = self._captions[cp].strip()
            self._captions[cp] = f"{tag}, {text}" if text else tag
        self._refresh_editor()
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
        for cp in list(self._captions):
            parts = [t.strip() for t in self._captions[cp].split(',')]
            new_parts = [t for t in parts if t.lower() != tag.lower()]
            if len(new_parts) != len(parts):
                removed += 1
            self._captions[cp] = ', '.join(new_parts)
        self._refresh_editor()
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
        old, new = old.strip(), new.strip()
        self._commit_current()
        replaced = 0
        for cp in list(self._captions):
            parts = [t.strip() for t in self._captions[cp].split(',')]
            new_parts = [(new if t.lower() == old.lower() else t) for t in parts]
            if new_parts != parts:
                replaced += 1
            self._captions[cp] = ', '.join(p for p in new_parts if p)
        self._refresh_editor()
        self.status_lbl.setText(
            get_text('ce_replace_tag_result', self.lang).format(old, new, replaced))

    # ── Language / Theme ────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self.load_btn.setText(get_text('ce_load_folder', lang))
        self.save_btn.setText(get_text('ce_save_all', lang))
        self.add_tag_btn.setText(get_text('ce_add_tag', lang))
        self.remove_tag_btn.setText(get_text('ce_remove_tag', lang))
        self.replace_tag_btn.setText(get_text('ce_replace_tag', lang))

    def refresh_styles(self):
        self.load_btn.setStyleSheet(theme.btn_primary())
        self.save_btn.setStyleSheet(theme.btn_primary())
        self.add_tag_btn.setStyleSheet(theme.btn_secondary())
        self.remove_tag_btn.setStyleSheet(theme.btn_secondary())
        self.replace_tag_btn.setStyleSheet(theme.btn_secondary())
        self.image_list.setStyleSheet(
            f"background-color: {theme.BG_CARD}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 6px;")
        self.preview_lbl.setStyleSheet(
            f"background-color: {theme.BG_DARK}; border-radius: 6px;")
        self.filename_lbl.setStyleSheet(
            f"color: {theme.ORANGE_LIGHT}; font-weight: bold; margin-top: 6px;")
        self.caption_edit.setStyleSheet(theme.log_area())
        self.status_lbl.setStyleSheet(theme.label_muted())


# ════════════════════════════════════════════════════════════════
#  CaptionStudioPage — top-level page with two tabs
# ════════════════════════════════════════════════════════════════

class CaptionStudioPage(QWidget):
    """Unified captioning + editing page with Danbooru autocomplete."""

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(8)
        root.setContentsMargins(16, 12, 16, 12)
        self.setLayout(root)

        self.title_lbl = QLabel(get_text('caption_studio_title', self.lang))
        self.title_lbl.setFont(QFont('Arial', 20, QFont.Bold))
        self.title_lbl.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self.title_lbl)

        self.subtitle_lbl = QLabel(get_text('caption_studio_subtitle', self.lang))
        self.subtitle_lbl.setStyleSheet(theme.label_muted())
        self.subtitle_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self.subtitle_lbl)

        # Tab widget
        self.tabs = QTabWidget()
        self.generate_tab = _GenerateTab(self.lang, self)
        self.edit_tab = _EditTab(self.lang, self)

        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setWidget(self.generate_tab)
        gen_scroll.setFrameShape(QFrame.NoFrame)

        self.tabs.addTab(gen_scroll,
                         get_text('caption_studio_tab_generate', self.lang))
        self.tabs.addTab(self.edit_tab,
                         get_text('caption_studio_tab_edit', self.lang))
        self.tabs.setStyleSheet(theme.tab_widget())
        root.addWidget(self.tabs, stretch=1)

        self.generate_tab.captioning_finished.connect(self._on_generation_done)

    def _on_generation_done(self, folder: str):
        if folder:
            self.edit_tab.reload_folder(folder)

    def update_language(self, lang: str):
        self.lang = lang
        self.title_lbl.setText(get_text('caption_studio_title', lang))
        self.subtitle_lbl.setText(get_text('caption_studio_subtitle', lang))
        self.tabs.setTabText(0, get_text('caption_studio_tab_generate', lang))
        self.tabs.setTabText(1, get_text('caption_studio_tab_edit', lang))
        self.generate_tab.update_language(lang)
        self.edit_tab.update_language(lang)

    def refresh_styles(self):
        self.title_lbl.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self.subtitle_lbl.setStyleSheet(theme.label_muted())
        self.tabs.setStyleSheet(theme.tab_widget())
        self.generate_tab.refresh_styles()
        self.edit_tab.refresh_styles()
