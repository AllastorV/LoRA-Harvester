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
    QTabWidget, QCompleter, QScrollArea, QSizePolicy,
    QGridLayout, QButtonGroup, QRadioButton, QSlider, QMessageBox,
)
from PyQt5 import sip
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QStringListModel
from PyQt5.QtGui import (
    QFont, QDragEnterEvent, QDropEvent, QPixmap, QIcon, QTextCursor,
)
from typing import Dict, List, Optional
from src.ui.translations import get_text
from src.ui import theme
from src.ui.clothing_tagger_widget import ClothingTaggerWidget
from src.core.caption_sync import (path_key, read_caption_snapshot, save_edited_caption)
from src.ui.animations import ToggleSwitch, Chip, SearchCombo, ThumbnailGrid

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


def _get_shared_tag_model() -> QStringListModel:
    """Return the shared tag model, recreating it if Qt already destroyed it.

    PyQt deletes parentless QObjects together with their QApplication, so the
    module-level wrapper can outlive the C++ model when a new application is
    created in the same process (e.g. consecutive offscreen Qt tests).
    """
    global _shared_tag_model
    if _shared_tag_model is None or sip.isdeleted(_shared_tag_model):
        _shared_tag_model = QStringListModel([])
    return _shared_tag_model


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
        # The shared dictionary is loaded on first focus, not while every
        # hidden editor is constructed during application startup.

    def _setup_completer(self):
        c = QCompleter(_get_shared_tag_model(), self)
        # For QTextEdit the popup must be parented to the viewport
        c.setWidget(self)
        c.setCompletionMode(QCompleter.PopupCompletion)
        c.setCaseSensitivity(Qt.CaseInsensitive)
        c.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
        c.setMaxVisibleItems(10)
        c.activated[str].connect(self._insert_completion)

        # Style the popup to match the dark theme
        popup = c.popup()
        theme.bind_style(popup, lambda: f"QListView {{"
            f"  background-color: {theme.BG_ELEVATED};"
            f"  color: {theme.TEXT_PRIMARY};"
            f"  border: 1px solid {theme.get_accent()};"
            f"  border-radius: 4px;"
            f"  font-size: {theme.fs(11)};"
            f"  padding: 2px;"
            f"}}"
            f"QListView::item:selected {{"
            f"  background-color: {theme.get_accent()};"
            f"  color: #ffffff;"
            f"}}"
            f"QListView::item:hover {{"
            f"  background-color: {theme.BG_HOVER};"
            f"}}")
        self._completer = c

    # ── Tag loading ─────────────────────────────────────────────

    def _ensure_tags_loaded(self):
        global _shared_tag_load_started
        if _shared_tag_load_started:
            return
        if _get_shared_tag_model().rowCount() > 0:
            return
        _shared_tag_load_started = True
        self._loader = _TagLoadThread()
        self._loader.loaded.connect(self._on_tags_loaded)
        self._loader.start()

    def _on_tags_loaded(self, tags: list):
        _get_shared_tag_model().setStringList(tags)
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
        """Return (normalized_word, start_pos).

        Extracts text between the last comma/newline and the cursor,
        then normalizes underscores → spaces so typing 'blue_sky'
        matches the 'blue sky' suggestion in the space-form tag list.
        """
        cursor = self.textCursor()
        pos = cursor.position()
        text = self.toPlainText()
        start = pos
        while start > 0 and text[start - 1] not in ',\n':
            start -= 1
        while start < pos and text[start] == ' ':
            start += 1
        raw_word = text[start:pos]
        # Normalize _ → space for matching against the space-form tag list
        normalized = raw_word.replace('_', ' ')
        return normalized, start

    # ── Completion insertion ────────────────────────────────────

    def _insert_completion(self, completion: str):
        word, start = self._text_under_cursor()
        tc = self.textCursor()
        tc.setPosition(start)
        tc.setPosition(start + len(word), QTextCursor.KeepAnchor)
        tc.insertText(completion + ', ')
        self.setTextCursor(tc)

    # ── Key handling ────────────────────────────────────────────

    def keyPressEvent(self, event):
        popup = self._completer.popup() if self._completer else None

        # Let the completer handle navigation keys when popup is visible
        if popup and popup.isVisible():
            if event.key() in (Qt.Key_Up, Qt.Key_Down):
                popup.keyPressEvent(event)
                return
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Tab):
                idx = popup.currentIndex()
                if idx.isValid():
                    self._completer.activated[str].emit(
                        self._completer.completionModel().data(idx))
                popup.hide()
                return
            if event.key() == Qt.Key_Escape:
                popup.hide()
                return

        super().keyPressEvent(event)

        if not self._completer:
            return
        # Ignore pure modifier presses
        if not event.text():
            return
        if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return

        word, _ = self._text_under_cursor()
        if len(word) < 1:
            self._completer.popup().hide()
            return

        if self._completer.model() is None or self._completer.model().rowCount() == 0:
            return

        if word != self._completer.completionPrefix():
            self._completer.setCompletionPrefix(word)
            self._completer.popup().setCurrentIndex(
                self._completer.completionModel().index(0, 0))

        # Map cursor rect from viewport to widget coordinates for correct popup position
        cr = self.cursorRect()
        cr.moveTopLeft(self.viewport().mapTo(self, cr.topLeft()))
        cr.setWidth(
            min(320, self._completer.popup().sizeHintForColumn(0)
                + self._completer.popup().verticalScrollBar().sizeHint().width() + 8)
        )
        self._completer.complete(cr)


# ════════════════════════════════════════════════════════════════
#  Captioning background thread
# ════════════════════════════════════════════════════════════════

class CaptioningThread(QThread):
    """Background thread for captioning (WD14 / Florence-2 / combined)."""
    progress = pyqtSignal(int, int, str)
    log_message = pyqtSignal(str)
    captioning_finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, captioner, image_folder: str, settings: Dict,
                 florence2=None):
        super().__init__()
        self.captioner = captioner
        self.florence2 = florence2
        self.image_folder = image_folder
        self.settings = settings
        self._running = True
        self._last_progress_pct = -1

    def run(self):
        mode = self.settings.get('mode', 'tags_only')
        use_wd14 = self.settings.get('use_wd14', True)
        use_f2 = self.settings.get('use_florence2', False)
        stats, error = None, None
        try:
            if use_wd14 and self.captioner:
                if self.captioner.wd14 and self.captioner.enable_wd14:
                    try:
                        self.log_message.emit("Loading WD14 model...")
                        self.captioner.wd14._load_model()
                        n_tags = len(self.captioner.wd14.tags) if self.captioner.wd14.tags else 0
                        self.log_message.emit(f"WD14 model loaded ({n_tags} tags)")
                        if n_tags == 0:
                            self.log_message.emit("WD14 tag list is empty; no automatic tags will be generated.")
                    except Exception as exc:
                        self.log_message.emit(f"WD14 failed: {exc}")
                        self.captioner.enable_wd14 = False
                elif self.captioner.enable_wd14:
                    self.log_message.emit("WD14 tagger is not initialized.")
            if use_f2 and self.florence2:
                try:
                    self.log_message.emit("Loading Florence-2 model...")
                    self.florence2._load_model()
                except Exception as exc:
                    self.log_message.emit(f"Florence-2 failed: {exc}")
                    if mode == 'florence2':
                        raise RuntimeError(f"Florence-2 load failed: {exc}") from exc
                    self.florence2 = None
            if not self._running:
                return
            if use_wd14 and not use_f2 and mode == 'tags_only':
                stats = self.captioner.caption_directory(
                    self.image_folder, mode='tags_only',
                    overwrite=self.settings.get('overwrite', False),
                    save_json=self.settings.get('save_json', False),
                    progress_callback=self._progress_callback,
                    recursive=self.settings.get('recursive', False))
            else:
                stats = self._run_with_florence2()
            self._release_caption_models()
            if self._running:
                try:
                    from src.core.clothing_service import optional_pipeline_pass
                    from src.core.clothing_io import find_images
                    clothing = optional_pipeline_pass(
                        find_images(Path(self.image_folder), self.settings.get('recursive', False)),
                        'caption', log=self.log_message.emit, stop=lambda: not self._running)
                    if clothing is not None:
                        stats['clothing'] = clothing
                except Exception as exc:
                    self.log_message.emit(f"Clothing post-pass failed; existing captions preserved: {exc}")
                    stats['clothing_error'] = str(exc)
        except Exception as exc:
            error = str(exc)
        finally:
            self._release_caption_models()
        if self._running:
            if error is not None:
                self.error.emit(error)
            elif stats is not None:
                self.captioning_finished.emit(stats)

    def _release_caption_models(self):
        for model in (self.captioner, self.florence2):
            if model is not None and hasattr(model, 'cleanup'):
                try:
                    model.cleanup()
                except Exception:
                    pass

    def _run_with_florence2(self) -> Dict:
        """Process images with Florence-2 (or combined WD14+Florence-2)."""
        import cv2
        import numpy as np
        from pathlib import Path as _Path

        def _imread_unicode(path):
            try:
                data = np.fromfile(str(path), dtype=np.uint8)
                if data.size == 0:
                    return None
                return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            except Exception:
                return None

        mode = self.settings.get('mode', 'tags_only')
        use_wd14 = self.settings.get('use_wd14', True)
        use_f2   = self.settings.get('use_florence2', False)
        f2_task  = self.settings.get('florence2_task', '<DETAILED_CAPTION>')
        overwrite = self.settings.get('overwrite', False)
        save_json = self.settings.get('save_json', False)
        recursive = self.settings.get('recursive', False)

        trigger = (self.settings.get('trigger_word') or '').strip()
        suffix  = (self.settings.get('caption_suffix') or '').strip()
        sep = ', '

        dir_path = _Path(self.image_folder)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        all_images = []
        seen_images = set()
        glob_fn = dir_path.rglob if recursive else dir_path.glob
        for ext in extensions:
            for img_path in list(glob_fn(ext)) + list(glob_fn(ext.upper())):
                key = str(img_path.resolve()).casefold()
                if key not in seen_images:
                    seen_images.add(key)
                    all_images.append(img_path)

        stats = {'total': len(all_images), 'captioned': 0,
                 'skipped': 0, 'errors': 0, 'zero_tags': 0}

        images_to_process = []
        for img_path in all_images:
            cap_path = img_path.with_suffix('.txt')
            if cap_path.exists() and not overwrite:
                stats['skipped'] += 1
            else:
                images_to_process.append(img_path)

        total = len(images_to_process)
        for i, img_path in enumerate(images_to_process):
            if not self._running:
                break
            if not self._progress_callback(i, total, img_path.name):
                break

            try:
                image = _imread_unicode(img_path)
                if image is None:
                    self.log_message.emit(f"⚠️ Cannot read image: {img_path.name}")
                    stats['errors'] += 1
                    continue

                parts = []
                nlp_caption = ''
                wd14_line = ''

                # Models run based on independent checkbox flags, not mode
                if self.florence2 and use_f2:
                    try:
                        nlp_caption = self.florence2.generate(image, task=f2_task) or ''
                    except Exception as e:
                        self.log_message.emit(f"⚠️ Florence-2 error on {img_path.name}: {e}")

                if use_wd14 and self.captioner and self.captioner.enable_wd14:
                    try:
                        raw_tags = self.captioner.wd14.predict(
                            image, self.settings.get('min_confidence', 0.35))
                        processed = self.captioner.process_tags(raw_tags)
                        wd14_line = sep.join(processed) if processed else ''
                        if not processed and not nlp_caption:
                            stats['zero_tags'] += 1
                    except Exception as e:
                        self.log_message.emit(f"⚠️ WD14 error on {img_path.name}: {e}")

                # Output FORMAT determined by mode pill selection
                if mode == 'tag_first':
                    # Tags on line 1, NLP caption on line 2 (uses whichever models are on)
                    body_parts = [p for p in (wd14_line, nlp_caption) if p]
                    body = '\n'.join(body_parts)
                elif mode == 'combined':
                    # NLP first, then tags — joined by separator on same line
                    body_parts = [p for p in (nlp_caption, wd14_line) if p]
                    body = sep.join(body_parts)
                else:
                    # tags_only / florence2 — whichever source is enabled
                    body_parts = [p for p in (wd14_line, nlp_caption) if p]
                    body = sep.join(body_parts)

                # Assemble: trigger + body + suffix
                caption = body
                if trigger:
                    caption = f"{trigger}{sep}{caption}" if caption else trigger
                if suffix:
                    caption = f"{caption}{sep}{suffix}" if caption else suffix

                cap_path = img_path.with_suffix('.txt')
                cap_path.write_text(caption.strip(), encoding='utf-8')

                if save_json:
                    import json
                    json_path = img_path.with_suffix('.json')
                    json_path.write_text(json.dumps(
                        {'caption': caption.strip(), 'mode': mode},
                        indent=2, ensure_ascii=False), encoding='utf-8')

                stats['captioned'] += 1
            except Exception as e:
                self.log_message.emit(f"❌ Error on {img_path.name}: {e}")
                stats['errors'] += 1

        return stats

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
        self.florence2 = None
        self.captioning_thread = None
        self._init_ui()

    def _init_ui(self):  # noqa: C901
        self.setAcceptDrops(True)
        self._install_thread = None

        # ── Hidden compat store (keeps widgets alive for backend/refresh) ──
        self._adv = QFrame()
        self._adv.hide()
        _adv_lay = QVBoxLayout(self._adv)

        # Legacy step titles (referenced in refresh_styles / update_language)
        self.step1_title = QLabel(get_text('step1_select_folder', self.lang))
        self.step2_title = QLabel()
        self.step3_title = QLabel(get_text('step3_start', self.lang))

        # Drop zone widgets (drag/drop still works via dragEnterEvent/dropEvent)
        self.drop_zone = QFrame()
        self.drop_zone.setMinimumHeight(65)
        theme.bind_style(self.drop_zone, theme.drop_zone_frame_default)
        _dz_lay = QHBoxLayout(self.drop_zone)
        _dz_lay.setContentsMargins(15, 5, 15, 5)
        self.drop_icon = QLabel("▸")
        theme.bind_style(self.drop_icon, theme.icon_transparent)
        _dz_lay.addWidget(self.drop_icon)
        self.folder_label = QLabel(get_text('drag_drop_folder', self.lang))
        theme.bind_style(self.folder_label, theme.label_transparent)
        self.folder_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        _dz_lay.addWidget(self.folder_label, stretch=1)

        self.browse_btn = QPushButton(get_text('select_input_folder', self.lang))
        theme.bind_style(self.browse_btn, theme.btn_browse)
        self.browse_btn.clicked.connect(self.select_folder)

        self.recursive_cb = QCheckBox(get_text('recursive_search', self.lang))
        self.recursive_cb.setToolTip(get_text('recursive_tooltip', self.lang))
        self.overwrite_cb = QCheckBox(get_text('overwrite_existing', self.lang))
        self.overwrite_cb.setToolTip(get_text('overwrite_tooltip', self.lang))
        self.overwrite_cb.setChecked(True)

        self.image_count_label = QLabel("")
        theme.bind_style(self.image_count_label, theme.label_success)

        # Hidden combos / spinboxes referenced by backend
        self.model_combo = QComboBox()
        self.model_combo.addItem(get_text('model_wd14', self.lang), 'wd14')
        self.model_combo.addItem(get_text('model_florence2', self.lang), 'florence2')
        self.model_combo.addItem(get_text('model_both', self.lang), 'both')
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)

        self.variant_combo = QComboBox()
        self.variant_combo.currentIndexChanged.connect(self._on_variant_changed)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(get_text('caption_mode_tags', self.lang), 'tags_only')
        self.mode_combo.addItem(get_text('caption_mode_tag_first', self.lang), 'tag_first')
        self.mode_combo.addItem(get_text('caption_mode_combined', self.lang), 'combined')
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText(get_text('trigger_placeholder', self.lang))
        theme.bind_style(self.trigger_edit, theme.line_edit)

        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText(get_text('suffix_placeholder', self.lang))
        theme.bind_style(self.suffix_edit, theme.line_edit)

        self.max_tags_spin = QSpinBox()
        self.max_tags_spin.setRange(5, 100); self.max_tags_spin.setValue(25)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.10, 0.90); self.conf_spin.setValue(0.35)
        self.conf_spin.setSingleStep(0.05); self.conf_spin.setDecimals(2)

        self.mode_label = QLabel(); self.mode_info = QLabel()
        self.f2_label = QLabel(get_text('florence2_model_label', self.lang))
        self.f2_combo = QComboBox()
        self.f2_combo.addItem(get_text('f2_base', self.lang), 'florence-2-base')
        self.f2_combo.addItem(get_text('f2_large', self.lang), 'florence-2-large')
        self.f2_task_combo = QComboBox()
        self.f2_task_combo.addItem(get_text('florence2_task_detailed', self.lang), '<DETAILED_CAPTION>')
        self.f2_task_combo.addItem(get_text('florence2_task_more', self.lang), '<MORE_DETAILED_CAPTION>')
        self.f2_task_combo.addItem(get_text('florence2_task_short', self.lang), '<CAPTION>')
        self._f2_settings_widget = QFrame()
        _f2l = QHBoxLayout(self._f2_settings_widget)
        _f2l.setContentsMargins(0, 0, 0, 0)
        _f2l.addWidget(self.f2_label); _f2l.addWidget(self.f2_combo); _f2l.addWidget(self.f2_task_combo)

        self.preset_label = QLabel(); self.preset_info = QLabel()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem('', 'high_accuracy')
        self.preset_combo.addItem('', 'balanced')
        self.preset_combo.addItem('', 'high_speed')
        self.preset_combo.addItem('', 'custom')
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self.max_label = QLabel(); self.max_info = QLabel()
        self.conf_label = QLabel(); self.conf_info = QLabel()
        self.neg_label = QLabel(); self.neg_info = QLabel()
        self.neg_edit = QLineEdit()
        self.neg_edit.setText(
            "watermark, signature, text, username, artist_name, "
            "twitter_username, patreon_username, dated"
        )

        self.wd14_cb = QCheckBox(); self.wd14_cb.setChecked(True)
        self.wd14_info = QLabel()
        self.wd14_combo = QComboBox()
        self.wd14_combo.addItems([
            'SmilingWolf/wd-swinv2-tagger-v3',
            'SmilingWolf/wd-convnext-tagger-v3',
            'SmilingWolf/wd-vit-tagger-v3',
            'SmilingWolf/wd-v1-4-moat-tagger-v2',
            'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        ])
        self._model_row_widget = QFrame()
        _mrl = QHBoxLayout(self._model_row_widget)
        _mrl.setContentsMargins(0, 0, 0, 0)
        _mrl.addWidget(self.wd14_cb); _mrl.addWidget(self.wd14_combo)

        self.keep_char_cb = QCheckBox(); self.keep_char_cb.setChecked(True)
        self.json_cb = QCheckBox()
        self.trigger_label = QLabel(); self.trigger_info = QLabel()
        self.suffix_label = QLabel(); self.suffix_info = QLabel()
        self.install_btn = QPushButton()
        self._model_lbl = QLabel(get_text('gen_model_lbl', self.lang))
        self._variant_lbl = QLabel(get_text('gen_variant_lbl', self.lang))
        self._mode_lbl = QLabel(get_text('gen_mode_lbl', self.lang))
        self._trigger_lbl = QLabel(get_text('gen_trigger_lbl', self.lang))
        self._suffix_lbl = QLabel(get_text('gen_lastwords_lbl', self.lang))
        self._maxtags_lbl = QLabel(get_text('gen_maxtags_lbl', self.lang))
        self._conf_lbl = QLabel(get_text('gen_conf_lbl', self.lang))

        for _w in [
            self.step1_title, self.step2_title, self.step3_title,
            self.drop_zone, self.browse_btn, self.recursive_cb, self.overwrite_cb,
            self.image_count_label, self.model_combo, self.variant_combo, self.mode_combo,
            self.max_tags_spin, self.conf_spin, self.mode_label, self.mode_info,
            self._f2_settings_widget, self.preset_label, self.preset_info, self.preset_combo,
            self.max_label, self.max_info, self.conf_label, self.conf_info,
            self.neg_label, self.neg_info, self.neg_edit, self._model_row_widget,
            self.keep_char_cb, self.json_cb, self.trigger_label, self.trigger_info,
            self.suffix_label, self.suffix_info, self.install_btn,
            self._model_lbl, self._variant_lbl, self._mode_lbl,
            self._trigger_lbl, self._suffix_lbl, self._maxtags_lbl, self._conf_lbl,
        ]:
            _adv_lay.addWidget(_w)

        self._populate_variants('wd14')
        self.preset_combo.setCurrentIndex(1)
        self._on_preset_changed(1)

        # ── Style helpers ────────────────────────────────────────
        def _sl_style():
            return (
                f"QSlider::groove:horizontal {{height:4px;background:{theme.BORDER};border-radius:2px;}}"
                f"QSlider::handle:horizontal {{width:14px;height:14px;background:{theme.ORANGE};"
                f"border-radius:7px;margin:-5px 0;}}"
                f"QSlider::sub-page:horizontal {{background:{theme.ORANGE};border-radius:2px;}}"
            )

        def _mono_lbl(text):
            lbl = QLabel(text)
            theme.bind_style(lbl, lambda: f"color:{theme.TEXT_MUTED};font-size:{theme.fs(10)};font-family:{theme.FONT_MONO};"
                f"background:transparent;border:none;letter-spacing:0.05em;")
            return lbl

        def _hint_lbl(text):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            theme.bind_style(lbl, lambda: f"color:{theme.TEXT_MUTED};font-size:{theme.fs(9)};background:transparent;border:none;")
            return lbl

        def _card(title, icon_char=None):
            f = QFrame()
            f.setProperty("lhCard", True)
            theme.bind_style(f, lambda: f"QFrame {{background:{theme.BG_CARD};border:1px solid {theme.BORDER_LIGHT};border-radius:10px;}}")
            lay = QVBoxLayout(f)
            lay.setContentsMargins(20, 16, 20, 16)
            lay.setSpacing(16)
            hdr = QHBoxLayout(); hdr.setSpacing(8)
            if icon_char:
                ic = QLabel(icon_char)
                theme.bind_style(ic, lambda: f"color:{theme.ORANGE};background:transparent;border:none;font-size:16px;")
                hdr.addWidget(ic)
            t = QLabel(title)
            theme.bind_style(t, lambda: f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(14)};font-weight:600;"
                f"background:transparent;border:none;letter-spacing:-0.01em;")
            hdr.addWidget(t); hdr.addStretch()
            lay.addLayout(hdr)
            return f

        # ══════════════════════════════════════════════════════════
        # ROOT LAYOUT
        # ══════════════════════════════════════════════════════════
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(24, 20, 24, 12)
        root_layout.setSpacing(16)
        self.setLayout(root_layout)

        # Page header
        pg_title = QLabel(get_text('batch_caption_title', self.lang))
        theme.bind_style(pg_title, lambda: f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(22)};font-weight:600;"
            f"letter-spacing:-0.015em;background:transparent;border:none;")
        self._gen_page_title = pg_title
        pg_title.setToolTip(get_text('batch_caption_desc', self.lang))
        hdr_left = QVBoxLayout(); hdr_left.setSpacing(4)
        hdr_left.addWidget(pg_title)
        status_badge = QFrame()
        theme.bind_style(status_badge, lambda: f"QFrame {{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:12px;}}")
        sb_lay = QHBoxLayout(status_badge); sb_lay.setContentsMargins(10, 4, 14, 4); sb_lay.setSpacing(6)
        dot = QLabel("●"); dot.setStyleSheet("color:#22c55e;background:transparent;border:none;font-size:8px;")
        sb_txt = QLabel(get_text('harvester_idle', self.lang))
        theme.bind_style(sb_txt, lambda: f"color:{theme.TEXT_SECONDARY};font-size:{theme.fs(10)};font-family:{theme.FONT_MONO};"
            f"background:transparent;border:none;")
        sb_lay.addWidget(dot); sb_lay.addWidget(sb_txt)
        hdr_row = QHBoxLayout()
        hdr_row.addLayout(hdr_left); hdr_row.addStretch(); hdr_row.addWidget(status_badge)
        root_layout.addLayout(hdr_row)

        # ── BENTO ROW ────────────────────────────────────────────
        bento = QHBoxLayout(); bento.setSpacing(16)

        # ── LEFT COLUMN (stretch=8) ──────────────────────────────
        left_col = QVBoxLayout(); left_col.setSpacing(16)

        # -- Model Selection card ---------------------------------
        model_card = _card(get_text('model_selection_title', self.lang), "⚙")
        mc_lay = model_card.layout()

        def _model_subcard(title, badge_text, badge_color, desc_text, sl_attr, sl_lbl_attr, sl_default, cb_attr):
            sc = QFrame()
            theme.bind_style(sc, lambda: f"QFrame {{background:rgba(0,0,0,0.3);border:1px solid {theme.BORDER};border-radius:6px;}}")
            scl = QVBoxLayout(sc); scl.setContentsMargins(16, 16, 16, 16); scl.setSpacing(10)
            top_row = QHBoxLayout()
            cb = QCheckBox(title); cb.setChecked(True)
            theme.bind_style(cb, lambda: f"QCheckBox {{color:{theme.TEXT_PRIMARY};font-size:{theme.fs(12)};font-weight:500;"
                f"background:transparent;border:none;}}"
                f"QCheckBox::indicator {{width:14px;height:14px;border:1px solid {theme.BORDER};"
                f"border-radius:2px;background:{theme.BG_SURFACE};}}"
                f"QCheckBox::indicator:checked {{background:{theme.ORANGE};border-color:{theme.ORANGE};}}")
            setattr(self, cb_attr, cb)
            top_row.addWidget(cb); top_row.addStretch()
            badge = Chip(badge_text, accent=badge_color)
            top_row.addWidget(badge)
            scl.addLayout(top_row)
            cb.setToolTip(desc_text)
            conf_row = QHBoxLayout()
            conf_row.addWidget(_mono_lbl("Confidence Threshold")); conf_row.addStretch()
            val_lbl = QLabel(f"{sl_default/100:.2f}")
            theme.bind_style(val_lbl, lambda: f"color:{theme.ORANGE};font-size:{theme.fs(11)};font-family:{theme.FONT_MONO};"
                f"background:transparent;border:none;")
            setattr(self, sl_lbl_attr, val_lbl)
            conf_row.addWidget(val_lbl)
            scl.addLayout(conf_row)
            sl = QSlider(Qt.Horizontal); sl.setMinimum(10); sl.setMaximum(90); sl.setValue(sl_default)
            theme.bind_style(sl, lambda _sl_style=_sl_style: _sl_style())
            setattr(self, sl_attr, sl)
            sl.valueChanged.connect(lambda v, lbl=val_lbl: lbl.setText(f"{v/100:.2f}"))
            scl.addWidget(sl)
            return sc

        wd14_sc = _model_subcard(
            "WD14 Tagger V2", "BOORU TAGS", "#60a5fa",
            "Best for extracting comma-separated Danbooru style tags.",
            "_wd14_sl", "_wd14_sl_lbl", 35, "_wd14_vis_cb",
        )
        f2_sc = _model_subcard(
            "Florence-2 Large", "NATURAL LANG", "#c084fc",
            "Generates descriptive natural language sentences for complex scenes.",
            "_f2_sl", "_f2_sl_lbl", 70, "_f2_vis_cb",
        )
        self._f2_vis_cb.setChecked(False)
        self._wd14_sl.valueChanged.connect(lambda v: self.conf_spin.setValue(round(v / 100.0, 2)))
        self._wd14_vis_cb.toggled.connect(self.wd14_cb.setChecked)

        models_hbox = QHBoxLayout(); models_hbox.setSpacing(12)
        models_hbox.addWidget(wd14_sc); models_hbox.addWidget(f2_sc)
        mc_lay.addLayout(models_hbox)

        # Model variant search-combo (visible, synced to hidden wd14_combo)
        mc_lay.addWidget(_mono_lbl("Model Variant"))
        self._wd14_search_combo = SearchCombo()
        self._wd14_search_combo.addItems([
            'SmilingWolf/wd-swinv2-tagger-v3',
            'SmilingWolf/wd-convnext-tagger-v3',
            'SmilingWolf/wd-vit-tagger-v3',
            'SmilingWolf/wd-v1-4-moat-tagger-v2',
            'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        ])
        self._wd14_search_combo.apply_theme()
        self._wd14_search_combo.currentTextChanged.connect(
            lambda t: self.wd14_combo.setCurrentText(t)
        )
        mc_lay.addWidget(self._wd14_search_combo)

        def _update_variant_combo():
            use_wd14 = self._wd14_vis_cb.isChecked()
            use_f2 = self._f2_vis_cb.isChecked()
            self._wd14_search_combo.blockSignals(True)
            self._wd14_search_combo.clear()
            if use_wd14 and not use_f2:
                # WD14 only
                self._wd14_search_combo.addItems([
                    'SmilingWolf/wd-swinv2-tagger-v3',
                    'SmilingWolf/wd-convnext-tagger-v3',
                    'SmilingWolf/wd-vit-tagger-v3',
                    'SmilingWolf/wd-v1-4-moat-tagger-v2',
                    'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
                ])
                self._wd14_search_combo.setPlaceholderText(get_text('wd14_variant_ph', self.lang))
            elif use_f2 and not use_wd14:
                # Florence-2 only
                self._wd14_search_combo.addItems([
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-large',
                ])
                self._wd14_search_combo.setPlaceholderText(get_text('f2_variant_ph', self.lang))
            else:
                # Both or neither
                self._wd14_search_combo.addItems([
                    'SmilingWolf/wd-swinv2-tagger-v3',
                    'SmilingWolf/wd-convnext-tagger-v3',
                    'SmilingWolf/wd-vit-tagger-v3',
                ])
                self._wd14_search_combo.setPlaceholderText(get_text('wd14_variant_ph', self.lang))
            self._wd14_search_combo.blockSignals(False)

        self._wd14_vis_cb.toggled.connect(lambda _: _update_variant_combo())
        self._f2_vis_cb.toggled.connect(lambda _: _update_variant_combo())

        left_col.addWidget(model_card)

        # -- Global Formatting card -------------------------------
        fmt_card = _card(get_text('global_formatting_title', self.lang), "🎛")
        fmt_lay = fmt_card.layout()
        grid = QGridLayout(); grid.setSpacing(16); grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1)

        def _field_col(label_text, widget, hint=None):
            col = QVBoxLayout(); col.setSpacing(4)
            col.addWidget(_mono_lbl(label_text))
            col.addWidget(widget)
            if hint:
                widget.setToolTip(hint)
            return col

        self._preset_vis = QComboBox()
        self._preset_vis.addItem(get_text('qpreset_high', self.lang),   {'thresh': 25, 'max_tags': 35})
        self._preset_vis.addItem(get_text('qpreset_normal', self.lang), {'thresh': 35, 'max_tags': 25})
        self._preset_vis.addItem(get_text('qpreset_speed', self.lang),  {'thresh': 45, 'max_tags': 15})
        self._preset_vis.addItem(get_text('qpreset_custom', self.lang), None)
        self._preset_vis.setCurrentIndex(1)
        theme.bind_style(self._preset_vis, theme.combo)

        def _apply_quality_preset(idx):
            data = self._preset_vis.itemData(idx)
            if data:
                thresh = data['thresh']
                max_tags = data['max_tags']
                self._wd14_sl.setValue(thresh)
                if hasattr(self, '_f2_sl'):
                    self._f2_sl.setValue(thresh)
                if hasattr(self, '_max_tags_vis'):
                    self._max_tags_vis.setValue(max_tags)
                self.max_tags_spin.setValue(max_tags)
                self.conf_spin.setValue(round(thresh / 100.0, 2))

        self._preset_vis.currentIndexChanged.connect(_apply_quality_preset)
        _apply_quality_preset(1)   # apply Normal on init
        grid.addLayout(_field_col(get_text('preset_label', self.lang), self._preset_vis), 0, 0)

        theme.bind_style(self.trigger_edit, theme.line_edit)
        grid.addLayout(
            _field_col(get_text('trigger_word', self.lang), self.trigger_edit,
                       get_text('trigger_hint', self.lang)),
            0, 1,
        )

        self._last_words_edit = QLineEdit()
        self._last_words_edit.setPlaceholderText(get_text('last_words_placeholder', self.lang))
        theme.bind_style(self._last_words_edit, theme.line_edit)
        grid.addLayout(
            _field_col(get_text('last_words_label', self.lang), self._last_words_edit,
                       get_text('last_words_hint', self.lang)),
            0, 2,
        )

        self._neg_prompt_edit = QLineEdit()
        self._neg_prompt_edit.setPlaceholderText(get_text('neg_prompt_placeholder', self.lang))
        theme.bind_style(self._neg_prompt_edit, theme.line_edit)
        self._neg_prompt_edit.setText(self.neg_edit.text())  # sync from hidden
        self._neg_prompt_edit.textChanged.connect(self.neg_edit.setText)
        self._neg_prompt_field_lbl = _mono_lbl(get_text('negative_prompt_label', self.lang))
        _neg_col = QVBoxLayout(); _neg_col.setSpacing(4)
        _neg_col.addWidget(self._neg_prompt_field_lbl)
        _neg_col.addWidget(self._neg_prompt_edit)
        grid.addLayout(_neg_col, 1, 0, 1, 2)

        self._max_tags_vis = QSpinBox()
        self._max_tags_vis.setRange(5, 150)
        self._max_tags_vis.setValue(self.max_tags_spin.value())
        theme.bind_style(self._max_tags_vis, theme.spinbox_compact)
        self._max_tags_vis.valueChanged.connect(self.max_tags_spin.setValue)
        self._max_tags_field_lbl = _mono_lbl(get_text('max_tags_field', self.lang))
        _mt_col = QVBoxLayout(); _mt_col.setSpacing(4)
        _mt_col.addWidget(self._max_tags_field_lbl)
        _mt_col.addWidget(self._max_tags_vis)
        grid.addLayout(_mt_col, 1, 2)

        # Tag Mode segmented radio bar
        tm_col = QVBoxLayout(); tm_col.setSpacing(6)
        tm_col.addWidget(_mono_lbl(get_text('tag_mode_label', self.lang)))
        seg_bar = QFrame()
        theme.bind_style(seg_bar, lambda: f"QFrame {{background:{theme.BG_SURFACE};border:1px solid {theme.BORDER};border-radius:6px;}}")
        seg_lay = QHBoxLayout(seg_bar); seg_lay.setContentsMargins(3, 3, 3, 3); seg_lay.setSpacing(2)
        self._tag_mode_grp = QButtonGroup(self); self._tag_mode_grp.setExclusive(True)

        def _rb_style(checked):
            if checked:
                return (
                    f"QRadioButton {{background:{theme.ORANGE};color:#ffffff;font-weight:700;"
                    f"border:none;border-radius:4px;padding:6px 4px;font-size:{theme.fs(12)};"
                    f"text-align:center;}}"
                    f"QRadioButton::indicator {{width:0;height:0;}}"
                )
            return (
                f"QRadioButton {{background:transparent;color:{theme.TEXT_PRIMARY};border:none;"
                f"border-radius:4px;padding:6px 4px;font-size:{theme.fs(12)};text-align:center;}}"
                f"QRadioButton::indicator {{width:0;height:0;}}"
            )

        _tag_mode_labels = [
            get_text('tag_mode_tag_only', self.lang),
            get_text('tag_mode_tag_first', self.lang),
            get_text('tag_mode_combined', self.lang),
        ]
        for i, lbl in enumerate(_tag_mode_labels):
            rb = QRadioButton(lbl)
            rb.setFixedWidth(139)
            rb.setFixedHeight(34)
            rb.setChecked(i == 1)
            theme.bind_style(rb, lambda _rb_style=_rb_style, i=i: _rb_style(i == 1))
            self._tag_mode_grp.addButton(rb, i)
            seg_lay.addWidget(rb)
            setattr(self, f"_tag_rb_{i}", rb)

        def _on_tag_mode(btn):
            for b in [self._tag_rb_0, self._tag_rb_1, self._tag_rb_2]:
                theme.bind_style(b, lambda _rb_style=_rb_style, b=b: _rb_style(b.isChecked()))
            self.mode_combo.setCurrentIndex(self._tag_mode_grp.checkedId())

        self._tag_mode_grp.buttonClicked.connect(_on_tag_mode)
        _seg_row = QHBoxLayout(); _seg_row.setContentsMargins(0, 0, 0, 0); _seg_row.setSpacing(0)
        _seg_row.addWidget(seg_bar); _seg_row.addStretch()
        tm_col.addLayout(_seg_row)
        grid.addLayout(tm_col, 2, 0, 1, 2)

        fmt_lay.addLayout(grid)
        left_col.addWidget(fmt_card)
        left_col.addStretch()

        left_w = QWidget(); left_w.setStyleSheet("background:transparent;")
        left_w.setLayout(left_col)
        bento.addWidget(left_w, stretch=8)

        # ── RIGHT COLUMN (stretch=4) ─────────────────────────────
        right_col = QVBoxLayout(); right_col.setSpacing(16)

        # -- Batch Rules card ------------------------------------
        rules_card = _card(get_text('batch_rules_title', self.lang), "")
        rules_lay = rules_card.layout()
        write_grp = QButtonGroup(self); write_grp.setExclusive(True)
        self._overwrite_rb = QRadioButton(); self._overwrite_rb.setChecked(True)
        self._overwrite_rb.toggled.connect(self.overwrite_cb.setChecked)
        self._append_rb = QRadioButton()
        write_grp.addButton(self._overwrite_rb, 0); write_grp.addButton(self._append_rb, 1)

        _rb_indicator = (
            f"QRadioButton {{background:transparent;border:none;}}"
            f"QRadioButton::indicator {{width:14px;height:14px;border:1px solid {theme.BORDER};"
            f"border-radius:7px;background:{theme.BG_SURFACE};}}"
            f"QRadioButton::indicator:checked {{background:{theme.ORANGE};border-color:{theme.ORANGE};}}"
        )

        def _rule_row(rb, title, desc_text):
            row = QHBoxLayout(); row.setSpacing(10); row.setAlignment(Qt.AlignTop)
            theme.bind_style(rb, lambda: f'QRadioButton {{background:transparent;border:none;}}QRadioButton::indicator {{width:14px;height:14px;border:1px solid {theme.BORDER};border-radius:7px;background:{theme.BG_SURFACE};}}QRadioButton::indicator:checked {{background:{theme.ORANGE};border-color:{theme.ORANGE};}}')
            row.addWidget(rb)
            txt = QVBoxLayout(); txt.setSpacing(2)
            t = QLabel(title)
            theme.bind_style(t, lambda: f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(12)};font-weight:500;"
                f"background:transparent;border:none;")
            rb.setToolTip(desc_text)
            t.setToolTip(desc_text)
            txt.addWidget(t)
            row.addLayout(txt)
            return row

        rules_lay.addLayout(_rule_row(
            self._overwrite_rb,
            get_text('batch_overwrite', self.lang),
            get_text('batch_overwrite_desc', self.lang),
        ))
        rules_lay.addLayout(_rule_row(
            self._append_rb,
            get_text('batch_append', self.lang),
            get_text('batch_append_desc', self.lang),
        ))
        div_line = QFrame(); div_line.setFrameShape(QFrame.HLine)
        theme.bind_style(div_line, lambda: f"background:{theme.BORDER};border:none;"); div_line.setFixedHeight(1)
        rules_lay.addWidget(div_line)

        self._auto_clean_cb = ToggleSwitch(checked=True)
        ac_row = QHBoxLayout(); ac_row.setSpacing(10); ac_row.setAlignment(Qt.AlignTop)
        ac_row.addWidget(self._auto_clean_cb)
        ac_txt = QVBoxLayout(); ac_txt.setSpacing(2)
        act = QLabel(get_text('auto_clean_tags', self.lang))
        theme.bind_style(act, lambda: f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(12)};font-weight:500;"
            f"background:transparent;border:none;")
        act.setToolTip(get_text('auto_clean_desc', self.lang))
        ac_txt.addWidget(act)
        ac_row.addLayout(ac_txt)
        rules_lay.addLayout(ac_row)
        rules_lay.addStretch()
        right_col.addWidget(rules_card, stretch=1)

        # -- Action card (orange top border) ----------------------
        action_card = QFrame()
        action_card.setProperty("lhActionCard", True)
        theme.bind_style(action_card, lambda: f"QFrame {{background:{theme.BG_CARD};border:1px solid {theme.BORDER_LIGHT};"
            f"border-top:2px solid {theme.ORANGE};border-radius:10px;}}")
        ac_lay = QVBoxLayout(action_card); ac_lay.setContentsMargins(20, 16, 20, 16); ac_lay.setSpacing(10)

        # ── Drop zone for dataset folder ──────────────────────────
        self._gen_drop_zone = QFrame()
        self._gen_drop_zone.setMinimumHeight(72)
        self._gen_drop_zone.setCursor(Qt.PointingHandCursor)
        theme.bind_style(self._gen_drop_zone, lambda: f"QFrame {{background:{theme.BG_SURFACE};border:2px dashed {theme.BORDER_ACCENT};"
            f"border-radius:8px;}}"
            f"QFrame:hover {{border-color:{theme.ORANGE};background:{theme.BG_HOVER};}}")
        _dz_lay = QVBoxLayout(self._gen_drop_zone)
        _dz_lay.setAlignment(Qt.AlignCenter); _dz_lay.setSpacing(4)
        self._gen_dz_icon = QLabel("▸")
        self._gen_dz_icon.setAlignment(Qt.AlignCenter)
        theme.bind_style(self._gen_dz_icon, lambda: f"color:{theme.TEXT_MUTED};font-size:22px;background:transparent;border:none;")
        self._gen_dz_title = QLabel(get_text('gen_drop_dataset', self.lang))
        self._gen_dz_title.setAlignment(Qt.AlignCenter)
        theme.bind_style(self._gen_dz_title, lambda: f"color:{theme.TEXT_SECONDARY};font-size:{theme.fs(12)};font-weight:600;"
            f"background:transparent;border:none;")
        _dz_lay.addWidget(self._gen_dz_icon); _dz_lay.addWidget(self._gen_dz_title)
        self._gen_drop_zone.mousePressEvent = lambda e: self.select_folder()
        ac_lay.addWidget(self._gen_drop_zone)

        # Browse button
        self._gen_browse_btn = QPushButton(get_text('gen_browse_folder', self.lang))
        self._gen_browse_btn.setFixedHeight(32)
        self._gen_browse_btn.setCursor(Qt.PointingHandCursor)
        theme.bind_style(self._gen_browse_btn, theme.btn_browse)
        self._gen_browse_btn.clicked.connect(self.select_folder)
        ac_lay.addWidget(self._gen_browse_btn)

        # Separator
        _sep = QFrame(); _sep.setFrameShape(QFrame.HLine)
        theme.bind_style(_sep, lambda: f"background:{theme.BORDER};border:none;"); _sep.setFixedHeight(1)
        ac_lay.addWidget(_sep)

        def _info_row(key, val_widget):
            r = QHBoxLayout()
            k = QLabel(key)
            theme.bind_style(k, lambda: f"color:{theme.TEXT_MUTED};font-size:{theme.fs(10)};font-family:{theme.FONT_MONO};"
                f"background:transparent;border:none;")
            r.addWidget(k); r.addStretch(); r.addWidget(val_widget)
            return r

        self._target_path_lbl = QLabel("—")
        theme.bind_style(self._target_path_lbl, lambda: f"color:{theme.TEXT_SECONDARY};font-size:{theme.fs(10)};font-family:{theme.FONT_MONO};"
            f"background:transparent;border:none;")
        ac_lay.addLayout(_info_row(get_text('target_dataset_label', self.lang), self._target_path_lbl))

        self._img_count_lbl = QLabel("—")
        theme.bind_style(self._img_count_lbl, lambda: f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(12)};font-weight:600;"
            f"font-family:{theme.FONT_MONO};background:transparent;border:none;")
        ac_lay.addLayout(_info_row(get_text('images_found_count', self.lang), self._img_count_lbl))

        self.start_btn = QPushButton(get_text('run_batch_captioning', self.lang))
        self.start_btn.setEnabled(False)
        self.start_btn.setFixedHeight(48)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        theme.bind_style(self.start_btn, lambda: f"QPushButton {{background:{theme.ORANGE};color:#1a1a1a;font-size:{theme.fs(13)};"
            f"font-weight:700;border-radius:6px;border:none;}}"
            f"QPushButton:hover {{background:#d97720;}}"
            f"QPushButton:disabled {{background:{theme.BG_SURFACE};color:{theme.TEXT_MUTED};}}")
        self.start_btn.clicked.connect(self.start_captioning)
        ac_lay.addWidget(self.start_btn)

        self.stop_btn = QPushButton(get_text('stop_btn', self.lang))
        self.stop_btn.setEnabled(False)
        theme.bind_style(self.stop_btn, theme.btn_danger)
        self.stop_btn.clicked.connect(self.stop_captioning)
        self.stop_btn.hide()
        ac_lay.addWidget(self.stop_btn)

        self.progress_bar = QProgressBar()
        theme.bind_style(self.progress_bar, theme.progress_bar)
        self.progress_bar.setFixedHeight(4)
        ac_lay.addWidget(self.progress_bar)

        right_col.addWidget(action_card)
        right_w = QWidget(); right_w.setStyleSheet("background:transparent;")
        right_w.setLayout(right_col)
        bento.addWidget(right_w, stretch=4)

        root_layout.addLayout(bento, stretch=1)

        # ── Log ──────────────────────────────────────────────────
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(140)
        theme.bind_style(self.log_text, theme.log_area)
        root_layout.addWidget(self.log_text)

        # Sync mode_combo to "Tag First" (index 1) as default
        self.mode_combo.setCurrentIndex(1)

        # Tooltips
        self._wd14_sl.setToolTip(get_text('gen_tt_wd14_sl', self.lang))
        self._f2_sl.setToolTip(get_text('gen_tt_f2_sl', self.lang))
        self.trigger_edit.setToolTip(get_text('gen_tt_trigger', self.lang))
        self.suffix_edit.setToolTip(get_text('gen_tt_suffix', self.lang))
        self._preset_vis.setToolTip(get_text('gen_tt_preset', self.lang))
        self._last_words_edit.setToolTip(get_text('gen_tt_last_words', self.lang))
        self.start_btn.setToolTip(get_text('gen_tt_start', self.lang))
        self.stop_btn.setToolTip(get_text('gen_tt_stop', self.lang))
        self._overwrite_rb.setToolTip(get_text('gen_tt_overwrite', self.lang))
        self._append_rb.setToolTip(get_text('gen_tt_append', self.lang))
        self._auto_clean_cb.setToolTip(get_text('gen_tt_auto_clean', self.lang))
        self._target_path_lbl.setToolTip(get_text('gen_tt_target_path', self.lang))
        self.conf_spin.setToolTip(get_text('gen_tt_conf_spin', self.lang))
        self.max_tags_spin.setToolTip(get_text('gen_tt_max_tags', self.lang))
        self.neg_edit.setToolTip(get_text('gen_tt_neg', self.lang))
        self.wd14_combo.setToolTip(get_text('gen_tt_wd14_combo', self.lang))
        self.f2_combo.setToolTip(get_text('gen_tt_f2_combo', self.lang))
        self.f2_task_combo.setToolTip(get_text('gen_tt_f2_task', self.lang))

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
                    # highlight both drop zones
                    theme.bind_style(self.drop_zone, theme.drop_zone_frame_active)
                    if hasattr(self, '_gen_drop_zone'):
                        theme.bind_style(self._gen_drop_zone, lambda: f"QFrame {{background:{theme.ORANGE_SUBTLE};"
                            f"border:2px dashed {theme.ORANGE};border-radius:8px;}}")
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        theme.bind_style(self.drop_zone, theme.drop_zone_frame_default)
        if hasattr(self, '_gen_drop_zone') and not self.selected_folder:
            theme.bind_style(self._gen_drop_zone, lambda: f"QFrame {{background:{theme.BG_SURFACE};border:2px dashed {theme.BORDER_ACCENT};"
                f"border-radius:8px;}}"
                f"QFrame:hover {{border-color:{theme.ORANGE};background:{theme.BG_HOVER};}}")

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
        theme.bind_style(self.folder_label, theme.label_success)
        self.drop_icon.setText("✅")
        theme.bind_style(self.drop_zone, theme.drop_zone_frame_success)
        self.drop_zone.setToolTip(folder)

        count = self._count_images(folder)
        self.image_count_label.setText(
            get_text('images_found', self.lang).format(count))
        self.start_btn.setEnabled(count > 0)

        # Update action card display
        short = folder if len(folder) <= 36 else "..." + folder[-33:]
        if hasattr(self, '_target_path_lbl'):
            self._target_path_lbl.setText(short)
        if hasattr(self, '_img_count_lbl'):
            self._img_count_lbl.setText(f"{count:,}")

        # Update visible drop zone to show success state
        if hasattr(self, '_gen_drop_zone'):
            name = Path(folder).name
            self._gen_dz_icon.setText("✅")
            theme.bind_style(self._gen_dz_icon, lambda: f"color:{theme.GREEN};font-size:22px;background:transparent;border:none;")
            self._gen_dz_title.setText(name)
            theme.bind_style(self._gen_dz_title, lambda: f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(12)};font-weight:600;"
                f"background:transparent;border:none;")
            theme.bind_style(self._gen_drop_zone, lambda: f"QFrame {{background:{theme.BG_SURFACE};border:2px solid {theme.GREEN};"
                f"border-radius:8px;}}"
                f"QFrame:hover {{border-color:{theme.ORANGE};background:{theme.BG_HOVER};}}")
            self._gen_drop_zone.setToolTip(folder)

        self.log(f"📁 {get_text('folder_selected', self.lang)}: {folder}")
        self.log(f"🖼️ {count} images found")
        self.folder_changed.emit(folder)

    def _count_images(self, folder: str) -> int:
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        seen = set()
        p = Path(folder)
        glob = p.rglob if self.recursive_cb.isChecked() else p.glob
        for pattern in ('*',):
            for img_path in glob(pattern):
                if img_path.is_file() and img_path.suffix.lower() in extensions:
                    seen.add(str(img_path.resolve()).casefold())
        return len(seen)

    # ── Mode handling ──────────────────────────────────────────

    # ── Variant tables ───────────────────────────────────────────
    # Each entry: (display_label, data_key, wd14_repo, f2_repo, conf, max_tags)
    _VARIANTS = {
        'wd14': [
            ("🏆 High Quality",  'hq',     'SmilingWolf/wd-swinv2-tagger-v3',    None, 0.28, 30),
            ("⭐ Recommended",   'rec',    'SmilingWolf/wd-convnext-tagger-v3',  None, 0.35, 25),
            ("⚡ Fast",          'fast',   'SmilingWolf/wd-vit-tagger-v3',       None, 0.40, 20),
            ("🔧 Custom",        'custom', 'SmilingWolf/wd-swinv2-tagger-v3',    None, None, None),
        ],
        'florence2': [
            ("⭐ Recommended",   'rec',    None, 'microsoft/Florence-2-base',  None, None),
            ("🏆 High Quality",  'hq',     None, 'microsoft/Florence-2-large', None, None),
        ],
        'both': [
            ("⭐ Recommended",   'rec',    'SmilingWolf/wd-convnext-tagger-v3', 'microsoft/Florence-2-base',  0.35, 25),
            ("🏆 High Quality",  'hq',     'SmilingWolf/wd-swinv2-tagger-v3',   'microsoft/Florence-2-large', 0.28, 30),
            ("⚡ Fast",          'fast',   'SmilingWolf/wd-vit-tagger-v3',      'microsoft/Florence-2-base',  0.40, 20),
        ],
    }

    def _populate_variants(self, model_key: str):
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        for label, data_key, *_ in self._VARIANTS.get(model_key, []):
            self.variant_combo.addItem(label, data_key)
        self.variant_combo.blockSignals(False)
        self._on_variant_changed(0)

    def _on_model_combo_changed(self, index: int):
        model_key = self.model_combo.itemData(index) or 'wd14'
        self._populate_variants(model_key)

    def _on_variant_changed(self, index: int):
        model_key = self.model_combo.currentData() or 'wd14'
        variants   = self._VARIANTS.get(model_key, [])
        if index < 0 or index >= len(variants):
            return
        _, data_key, wd14_repo, f2_repo, conf, max_tags = variants[index]
        is_custom = (data_key == 'custom')

        # Sync hidden model combos
        if wd14_repo and hasattr(self, 'wd14_combo'):
            idx = self.wd14_combo.findText(wd14_repo)
            if idx >= 0:
                self.wd14_combo.setCurrentIndex(idx)
        if f2_repo and hasattr(self, 'f2_combo'):
            idx = self.f2_combo.findData(
                'florence-2-base' if 'base' in f2_repo.lower() else 'florence-2-large')
            if idx >= 0:
                self.f2_combo.setCurrentIndex(idx)

        # Preset sets default values; spinboxes always stay editable
        if hasattr(self, 'conf_spin') and hasattr(self, 'max_tags_spin'):
            if not is_custom:
                if conf is not None:
                    self.conf_spin.setValue(conf)
                if max_tags is not None:
                    self.max_tags_spin.setValue(max_tags)

    def _on_mode_changed(self, index: int):
        """No-op: mode is read at captioning time via get_settings()."""
        pass

    # ── Preset handling ─────────────────────────────────────────

    # Preset → (wd14_model, min_confidence)
    # max_tags is always user-controlled — presets never override it.
    _PRESETS = {
        'high_accuracy': ('SmilingWolf/wd-swinv2-tagger-v3', 0.30),
        'balanced':      ('SmilingWolf/wd-convnext-tagger-v3', 0.35),
        'high_speed':    ('SmilingWolf/wd-vit-tagger-v3', 0.40),
    }

    def _on_preset_changed(self, index: int):
        """Apply preset values to model/confidence.
        'custom' unlocks the manual model combo.
        max_tags is never touched — the user controls it freely."""
        key = self.preset_combo.itemData(index)
        if key == 'custom':
            self._model_row_widget.setVisible(True)
            return
        preset = self._PRESETS.get(key)
        if not preset:
            return
        model, conf = preset
        idx = self.wd14_combo.findText(model)
        if idx >= 0:
            self.wd14_combo.setCurrentIndex(idx)
        self.conf_spin.setValue(conf)
        self._model_row_widget.setVisible(False)

    def get_settings(self) -> Dict:
        use_wd14      = getattr(self, '_wd14_vis_cb', self.wd14_cb).isChecked()
        use_florence2 = getattr(self, '_f2_vis_cb', None) is not None and self._f2_vis_cb.isChecked()
        grp = getattr(self, '_tag_mode_grp', None)
        mode_map = {0: 'tags_only', 1: 'tag_first', 2: 'combined'}
        mode = mode_map.get(grp.checkedId() if grp else -1, 'tags_only')
        conf = round(getattr(self, '_wd14_sl', self.conf_spin).value() / 100.0, 2) \
            if hasattr(self, '_wd14_sl') else self.conf_spin.value()
        return {
            'mode':           mode,
            'trigger_word':   self.trigger_edit.text().strip(),
            'caption_suffix': getattr(self, '_last_words_edit', self.suffix_edit).text().strip(),
            'max_tags':       getattr(self, '_max_tags_vis', self.max_tags_spin).value(),
            'min_confidence': conf,
            'negative_tags':  [t.strip() for t in
                               getattr(self, '_neg_prompt_edit', self.neg_edit).text().split(',')
                               if t.strip()],
            'keep_character_tags': self.keep_char_cb.isChecked(),
            'save_json':      self.json_cb.isChecked(),
            'overwrite':      getattr(self, '_overwrite_rb', self.overwrite_cb).isChecked(),
            'recursive':      self.recursive_cb.isChecked(),
            'use_wd14':       use_wd14,
            'use_florence2':  use_florence2,
            'wd14_model':     self.wd14_combo.currentText(),
            'florence2_model': self.f2_combo.currentData() or 'florence-2-base',
            'florence2_task': self.f2_task_combo.currentData() or '<DETAILED_CAPTION>',
        }

    # ── Model installer ─────────────────────────────────────────

    def _start_model_install(self):
        if self._install_thread and self._install_thread.isRunning():
            return
        from src.core.model_installer import ModelInstallThread
        self.install_btn.setEnabled(False)
        self.install_btn.setText("⬇  Installing…")
        self.progress_bar.setValue(0)
        self.log("")
        self.log("═" * 40)
        self.log("📦 Installing default models…")
        self._install_thread = ModelInstallThread(include_florence2=False, parent=self)
        self._install_thread.log_message.connect(self.log)
        self._install_thread.progress.connect(self.progress_bar.setValue)
        self._install_thread.finished_ok.connect(self._on_install_done)
        self._install_thread.start()

    def _on_install_done(self, ok: bool, summary: str):
        self.log(summary)
        self.log("═" * 40)
        self.progress_bar.setValue(100 if ok else 0)
        self.install_btn.setText("⬇  Install Models")
        self.install_btn.setEnabled(True)

    # ── Captioning control ──────────────────────────────────────

    def _ensure_models_downloaded(self) -> bool:
        """Return True if required model files exist; auto-download if not."""
        from src.core.model_paths import WD14_DIR, FLORENCE2_DIR
        settings = self.get_settings()

        need_wd14 = settings['use_wd14']
        need_f2   = settings['use_florence2']
        wd14_repo = settings.get('wd14_model', 'SmilingWolf/wd-swinv2-tagger-v3')
        f2_repo   = settings.get('florence2_model', 'florence-2-base')

        missing = []
        if need_wd14 and not any(WD14_DIR.rglob("model.onnx")):
            missing.append(('wd14', wd14_repo))
        if need_f2 and not any(FLORENCE2_DIR.rglob("config.json")):
            missing.append(('florence2', f2_repo))

        if not missing:
            return True

        if self._install_thread and self._install_thread.isRunning():
            return False

        labels = ', '.join(r for _, r in missing)
        self.log(f"📦 Downloading missing models: {labels}…")
        self.start_btn.setEnabled(False)
        from src.core.model_installer import ModelInstallThread
        self._install_thread = ModelInstallThread(
            include_florence2=need_f2,
            wd14_repo=wd14_repo,
            parent=self,
        )
        self._install_thread.log_message.connect(self.log)
        self._install_thread.progress.connect(self.progress_bar.setValue)
        self._install_thread.finished_ok.connect(self._on_auto_install_done)
        self._install_thread.start()
        return False

    def _on_auto_install_done(self, ok: bool, summary: str):
        self.log(summary)
        if ok:
            self.log("▶ Resuming captioning…")
            self.start_captioning()
        else:
            self.log("⚠️ Download had errors — check log and retry.")
            self.start_btn.setEnabled(True)

    def start_captioning(self):
        if getattr(self.window(), '_close_pending', False):
            return
        if self.captioning_thread is not None:
            self.log('The previous caption job is still finishing. Wait for cleanup before restarting.')
            return
        studio = getattr(self.window(), 'caption_studio_page', None)
        if studio is not None and studio.clothing_tab.is_busy():
            self.log("Finish or stop the clothing job before captioning.")
            return
        if not self.selected_folder:
            return
        if not self._ensure_models_downloaded():
            return
        self.log(f"\n{'=' * 40}")
        self.log("🚀 " + get_text('log_captioning', self.lang))
        settings = self.get_settings()
        mode = settings['mode']
        if settings['trigger_word']:
            self.log(f"   Trigger: {settings['trigger_word']}")
        if settings.get('caption_suffix'):
            self.log(f"   Suffix: {settings['caption_suffix']}")
        self.log(f"   Mode: {mode}")

        use_wd14      = settings['use_wd14']
        use_florence2 = settings['use_florence2']

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
                enable_wd14=use_wd14,
            )
            self.log("✅ WD14 Captioner initialized" if use_wd14 else "ℹ️ WD14 disabled for this mode")
        except Exception as e:
            self.log(f"❌ Error: {e}")
            return

        self.florence2 = None
        if use_florence2:
            try:
                from src.core.florence2_captioner import Florence2Captioner
                self.florence2 = Florence2Captioner(
                    model_type=settings.get('florence2_model', 'florence-2-base'))
                self.log("✅ Florence-2 initialized")
            except Exception as e:
                self.log(f"❌ Florence-2 error: {e}")
                if mode == 'florence2':
                    return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)

        self.captioning_thread = CaptioningThread(
            self.captioner, self.selected_folder, settings,
            florence2=self.florence2)
        self.captioning_thread.progress.connect(self._on_progress)
        self.captioning_thread.log_message.connect(self.log)
        self.captioning_thread.captioning_finished.connect(self._on_finished)
        self.captioning_thread.error.connect(self._on_error)
        self.captioning_thread.finished.connect(self._safe_delete_thread)
        observer = getattr(self, '_ai_before_worker_start', None)
        if observer is not None:
            observer(self.captioning_thread)
        self.captioning_thread.start()

    def stop_captioning(self):
        if self.captioning_thread and self.captioning_thread.isRunning():
            self.captioning_thread.stop()
            self.log("Stopping...")
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.browse_btn.setEnabled(False)

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
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        if self.captioner:
            try:
                self.captioner.cleanup()
            except Exception:
                pass
            self.captioner = None
        if hasattr(self, 'florence2') and self.florence2:
            try:
                self.florence2.cleanup()
            except Exception:
                pass
            self.florence2 = None

    # ── Language / Theme ────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang

        def _s(widget, text):
            if widget and hasattr(widget, 'setText'):
                widget.setText(text)

        def _tt(widget, text):
            if widget and hasattr(widget, 'setToolTip'):
                widget.setToolTip(text)

        # Titles that exist
        _s(getattr(self, 'step1_title', None), get_text('step1_select_folder', lang))
        _s(getattr(self, 'step3_title', None), get_text('step3_start', lang))

        def _combo_items(combo, keys):
            if not combo:
                return
            idx = combo.currentIndex()
            combo.blockSignals(True)
            for i, key in enumerate(keys):
                if i < combo.count():
                    # preserve userData — setItemText only changes the label
                    combo.setItemText(i, get_text(key, lang))
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        # Hidden advanced labels (kept for backend, refreshed for completeness)
        _s(getattr(self, '_model_lbl',   None), get_text('gen_model_lbl', lang))
        _s(getattr(self, '_variant_lbl', None), get_text('gen_variant_lbl', lang))
        _s(getattr(self, '_mode_lbl',    None), get_text('gen_mode_lbl', lang))
        _s(getattr(self, '_trigger_lbl', None), get_text('gen_trigger_lbl', lang))
        _s(getattr(self, '_suffix_lbl',  None), get_text('gen_lastwords_lbl', lang))
        _s(getattr(self, '_maxtags_lbl', None), get_text('gen_maxtags_lbl', lang))
        _s(getattr(self, '_conf_lbl',    None), get_text('gen_conf_lbl', lang))

        # Visible formatting-card field labels
        _s(getattr(self, '_neg_prompt_field_lbl', None), get_text('negative_prompt_label', lang))
        _s(getattr(self, '_max_tags_field_lbl',   None), get_text('max_tags_field', lang))

        # Combo item texts (userData preserved)
        _combo_items(getattr(self, 'mode_combo', None),
                     ['caption_mode_tags', 'caption_mode_tag_first', 'caption_mode_combined'])
        _combo_items(getattr(self, 'model_combo', None),
                     ['model_wd14', 'model_florence2', 'model_both'])
        _combo_items(getattr(self, 'f2_combo', None),
                     ['f2_base', 'f2_large'])
        _combo_items(getattr(self, '_preset_vis', None),
                     ['qpreset_high', 'qpreset_normal', 'qpreset_speed', 'qpreset_custom'])

        # Buttons & misc
        _s(getattr(self, 'browse_btn',  None), get_text('select_input_folder', lang))
        _s(getattr(self, '_gen_browse_btn', None), get_text('gen_browse_folder', lang))
        _s(getattr(self, '_gen_dz_title',   None), get_text('gen_drop_dataset', lang))
        _s(getattr(self, 'start_btn',   None), get_text('start_captioning', lang))
        _s(getattr(self, 'stop_btn',    None), get_text('stop_btn', lang))
        if not self.selected_folder:
            _s(getattr(self, 'folder_label', None), get_text('drag_drop_folder', lang))

        # Checkboxes
        _s(getattr(self, 'recursive_cb',   None), get_text('recursive_search', lang))
        _s(getattr(self, 'overwrite_cb',   None), get_text('overwrite_existing', lang))

        # Placeholders
        _trig = getattr(self, 'trigger_edit', None)
        if _trig:
            _trig.setPlaceholderText(get_text('trigger_placeholder', lang))
        _suf = getattr(self, 'suffix_edit', None)
        if _suf:
            _suf.setPlaceholderText(get_text('suffix_placeholder', lang))
        _lw = getattr(self, '_last_words_edit', None)
        if _lw:
            _lw.setPlaceholderText(get_text('last_words_placeholder', lang))
        _np = getattr(self, '_neg_prompt_edit', None)
        if _np:
            _np.setPlaceholderText(get_text('neg_prompt_placeholder', lang))

        # Tooltips
        _tt(getattr(self, '_gen_page_title', None), get_text('batch_caption_desc', lang))
        _tt(getattr(self, '_wd14_vis_cb', None), get_text('gen_tt_wd14_sl', lang))
        _tt(getattr(self, '_f2_vis_cb', None), get_text('gen_tt_f2_sl', lang))
        _tt(getattr(self, 'recursive_cb',  None), get_text('recursive_tooltip', lang))
        _tt(getattr(self, 'overwrite_cb',  None), get_text('overwrite_tooltip', lang))
        _tt(getattr(self, '_wd14_sl',          None), get_text('gen_tt_wd14_sl', lang))
        _tt(getattr(self, '_f2_sl',            None), get_text('gen_tt_f2_sl', lang))
        _tt(getattr(self, 'trigger_edit',      None), get_text('gen_tt_trigger', lang))
        _tt(getattr(self, 'suffix_edit',       None), get_text('gen_tt_suffix', lang))
        _tt(getattr(self, '_preset_vis',       None), get_text('gen_tt_preset', lang))
        _tt(getattr(self, '_last_words_edit',  None), get_text('gen_tt_last_words', lang))
        _tt(getattr(self, 'start_btn',         None), get_text('gen_tt_start', lang))
        _tt(getattr(self, 'stop_btn',          None), get_text('gen_tt_stop', lang))
        _tt(getattr(self, '_overwrite_rb',     None), get_text('gen_tt_overwrite', lang))
        _tt(getattr(self, '_append_rb',        None), get_text('gen_tt_append', lang))
        _tt(getattr(self, '_auto_clean_cb',    None), get_text('gen_tt_auto_clean', lang))
        _tt(getattr(self, '_target_path_lbl',  None), get_text('gen_tt_target_path', lang))
        _tt(getattr(self, 'conf_spin',         None), get_text('gen_tt_conf_spin', lang))
        _tt(getattr(self, 'max_tags_spin',     None), get_text('gen_tt_max_tags', lang))
        _tt(getattr(self, 'neg_edit',          None), get_text('gen_tt_neg', lang))
        _tt(getattr(self, 'wd14_combo',        None), get_text('gen_tt_wd14_combo', lang))
        _tt(getattr(self, 'f2_combo',          None), get_text('gen_tt_f2_combo', lang))
        _tt(getattr(self, 'f2_task_combo',     None), get_text('gen_tt_f2_task', lang))

        # Florence-2 task combo (hidden but kept for backend)
        f2t = getattr(self, 'f2_task_combo', None)
        if f2t:
            idx = f2t.currentIndex()
            f2t.blockSignals(True)
            for i, key in enumerate(['florence2_task_detailed',
                                      'florence2_task_more', 'florence2_task_short']):
                if i < f2t.count():
                    f2t.setItemText(i, get_text(key, lang))
            f2t.setCurrentIndex(idx)
            f2t.blockSignals(False)
        self.f2_task_combo.blockSignals(False)

    def refresh_styles(self):
        """Refresh existing controls, including dynamically added children."""
        return theme.refresh_styles(self)


# ════════════════════════════════════════════════════════════════
#  Edit Tab (ported from CaptionEditorPage, with TagCompleterTextEdit)
# ════════════════════════════════════════════════════════════════

class _EditTab(QWidget):
    """Browse images, review/edit captions, bulk tag operations."""

    folder_changed = pyqtSignal(str)
    captions_saved = pyqtSignal(object)  # Absolute .txt paths after a successful save/reload.

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._folder: Optional[str] = None
        self._items: List[tuple] = []
        self._captions: Dict[str, str] = {}
        self._current_idx: int = -1
        self._dirty: bool = False
        self._snapshots = {}
        self._read_errors = {}
        self._dirty_paths = set()
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(10)
        root.setContentsMargins(20, 12, 20, 12)
        self.setLayout(root)

        # ── Toolbar ──────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.load_btn = QPushButton(get_text('ce_load_folder', self.lang))
        theme.bind_style(self.load_btn, theme.btn_primary)
        self.load_btn.setMinimumWidth(140)
        self.load_btn.clicked.connect(self._browse_folder)
        toolbar.addWidget(self.load_btn)

        self.save_btn = QPushButton(get_text('ce_save_all', self.lang))
        theme.bind_style(self.save_btn, theme.btn_secondary)
        self.save_btn.setMinimumWidth(130)
        self.save_btn.clicked.connect(self._save_all)
        self.save_btn.setEnabled(False)
        toolbar.addWidget(self.save_btn)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        theme.bind_style(sep, lambda: f"color: {theme.BORDER};")
        toolbar.addWidget(sep)

        self.add_tag_btn = QPushButton(get_text('ce_add_tag', self.lang))
        theme.bind_style(self.add_tag_btn, theme.btn_secondary)
        self.add_tag_btn.setMinimumWidth(130)
        self.add_tag_btn.clicked.connect(self._bulk_add_tag)
        self.add_tag_btn.setEnabled(False)
        toolbar.addWidget(self.add_tag_btn)

        self.remove_tag_btn = QPushButton(get_text('ce_remove_tag', self.lang))
        theme.bind_style(self.remove_tag_btn, theme.btn_secondary)
        self.remove_tag_btn.setMinimumWidth(130)
        self.remove_tag_btn.clicked.connect(self._bulk_remove_tag)
        self.remove_tag_btn.setEnabled(False)
        toolbar.addWidget(self.remove_tag_btn)

        self.replace_tag_btn = QPushButton(get_text('ce_replace_tag', self.lang))
        theme.bind_style(self.replace_tag_btn, theme.btn_secondary)
        self.replace_tag_btn.setMinimumWidth(120)
        self.replace_tag_btn.clicked.connect(self._bulk_replace_tag)
        self.replace_tag_btn.setEnabled(False)
        toolbar.addWidget(self.replace_tag_btn)

        toolbar.addStretch()

        self.status_lbl = QLabel(get_text('ce_no_images', self.lang))
        theme.bind_style(self.status_lbl, theme.label_muted)
        toolbar.addWidget(self.status_lbl)

        root.addLayout(toolbar)

        # ── 3-panel bento ────────────────────────────────────────
        bento = QHBoxLayout()
        bento.setSpacing(12)

        # Left panel — file list card (fixed 220px)
        self._left_card = QFrame()
        self._left_card.setFixedWidth(220)
        theme.bind_style(self._left_card, lambda: f"QFrame {{ background: {theme.BG_CARD}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 10px; }}")
        left_card_lay = QVBoxLayout(self._left_card)
        left_card_lay.setContentsMargins(0, 0, 0, 0)
        left_card_lay.setSpacing(0)

        # File list header with grid/list toggle
        list_hdr_row = QHBoxLayout()
        list_hdr_row.setContentsMargins(8, 0, 4, 0)
        list_hdr_row.setSpacing(4)
        list_hdr = QLabel(get_text('edit_images_hdr', self.lang))
        self._list_hdr = list_hdr
        list_hdr.setFixedHeight(36)
        theme.bind_style(list_hdr, lambda: f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(11)}; font-weight: 600; "
            f"letter-spacing: 0.05em; background: transparent; border: none;")
        list_hdr_row.addWidget(list_hdr, stretch=1)
        self._edit_list_btn = QPushButton("≡")
        self._edit_list_btn.setFixedSize(24, 24)
        self._edit_list_btn.setCheckable(True); self._edit_list_btn.setChecked(True)
        theme.bind_style(self._edit_list_btn, lambda: f"QPushButton {{ background: {theme.ORANGE_SUBTLE}; color: {theme.ORANGE};"
            f" border: none; border-radius: 4px; font-size: 14px; }}"
            f"QPushButton:!checked {{ background: transparent; color: {theme.TEXT_MUTED}; }}")
        self._edit_grid_btn = QPushButton("⊞")
        self._edit_grid_btn.setFixedSize(24, 24)
        self._edit_grid_btn.setCheckable(True)
        theme.bind_style(self._edit_grid_btn, lambda: f"QPushButton {{ background: transparent; color: {theme.TEXT_MUTED};"
            f" border: none; border-radius: 4px; font-size: 14px; }}"
            f"QPushButton:checked {{ background: {theme.ORANGE_SUBTLE}; color: {theme.ORANGE}; }}")
        list_hdr_row.addWidget(self._edit_list_btn)
        list_hdr_row.addWidget(self._edit_grid_btn)
        list_hdr_wrap = QWidget()
        theme.bind_style(list_hdr_wrap, lambda: f"border-bottom: 1px solid {theme.BORDER}; background: transparent;")
        list_hdr_wrap.setLayout(list_hdr_row)
        list_hdr_wrap.setFixedHeight(37)
        left_card_lay.addWidget(list_hdr_wrap)

        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(48, 48))
        self.image_list.setSpacing(2)
        theme.bind_style(self.image_list, lambda: f"QListWidget {{ background: transparent; border: none; "
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(11)}; }}"
            f"QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}"
            f"QListWidget::item:selected {{ background: {theme.ORANGE}22; "
            f"color: {theme.ORANGE_LIGHT}; }}"
            f"QListWidget::item:hover:!selected {{ background: {theme.BG_SURFACE}; }}")
        self.image_list.currentRowChanged.connect(self._on_item_selected)
        left_card_lay.addWidget(self.image_list)

        self._thumb_grid = ThumbnailGrid(thumb_size=62, columns=3)
        self._thumb_grid.item_clicked.connect(self._on_thumb_clicked)
        self._thumb_grid.hide()
        left_card_lay.addWidget(self._thumb_grid)

        def _toggle_edit_view(to_grid: bool):
            self._edit_grid_btn.setChecked(to_grid)
            self._edit_list_btn.setChecked(not to_grid)
            if to_grid:
                self.image_list.hide()
                self._thumb_grid.show()
                # Populate grid with current paths
                paths = [img_path for img_path, _ in self._items]
                self._thumb_grid.set_paths(paths)
            else:
                self._thumb_grid.hide()
                self.image_list.show()

        self._edit_list_btn.clicked.connect(lambda: _toggle_edit_view(False))
        self._edit_grid_btn.clicked.connect(lambda: _toggle_edit_view(True))

        bento.addWidget(self._left_card)

        # Right area — preview card (top) + editor card (bottom)
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # Preview card
        self._preview_card = QFrame()
        theme.bind_style(self._preview_card, lambda: f"QFrame {{ background: {theme.BG_CARD}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 10px; }}")
        preview_lay = QVBoxLayout(self._preview_card)
        preview_lay.setContentsMargins(12, 10, 12, 10)
        preview_lay.setSpacing(6)

        self.filename_lbl = QLabel("")
        theme.bind_style(self.filename_lbl, lambda: f"color: {theme.TEXT_PRIMARY}; font-weight: 600; "
            f"font-size: {theme.fs(11)}; background: transparent; border: none;")
        preview_lay.addWidget(self.filename_lbl)

        self.preview_lbl = QLabel()
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setMinimumHeight(240)
        self.preview_lbl.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        theme.bind_style(self.preview_lbl, lambda: f"background: {theme.BG_DEEPEST}; border-radius: 6px; border: none;")
        preview_lay.addWidget(self.preview_lbl, stretch=1)

        right_col.addWidget(self._preview_card, stretch=55)

        # Editor card
        self._editor_card = QFrame()
        theme.bind_style(self._editor_card, lambda: f"QFrame {{ background: {theme.BG_CARD}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 10px; }}")
        editor_lay = QVBoxLayout(self._editor_card)
        editor_lay.setContentsMargins(12, 10, 12, 10)
        editor_lay.setSpacing(8)

        editor_hdr = QHBoxLayout()
        editor_title = QLabel(get_text('edit_tags_caption', self.lang))
        self._editor_title = editor_title
        theme.bind_style(editor_title, lambda: f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(10)}; "
            f"font-weight: 600; background: transparent; border: none;")
        editor_hdr.addWidget(editor_title)
        editor_hdr.addStretch()

        self._save_one_btn = QPushButton(get_text('edit_save', self.lang))
        self._save_one_btn.setMinimumWidth(70)
        theme.bind_style(self._save_one_btn, theme.btn_primary)
        self._save_one_btn.clicked.connect(self._save_current)
        self._save_one_btn.setEnabled(False)
        editor_hdr.addWidget(self._save_one_btn)

        self._save_next_btn = QPushButton(get_text('edit_save_next', self.lang))
        self._save_next_btn.setMinimumWidth(150)
        theme.bind_style(self._save_next_btn, theme.btn_secondary)
        self._save_next_btn.clicked.connect(self._save_and_next)
        self._save_next_btn.setEnabled(False)
        editor_hdr.addWidget(self._save_next_btn)

        self._revert_btn = QPushButton(get_text('edit_revert', self.lang))
        self._revert_btn.setMinimumWidth(70)
        theme.bind_style(self._revert_btn, lambda: f"QPushButton {{ background: transparent; border: 1px solid {theme.BORDER}; "
            f"color: {theme.TEXT_MUTED}; border-radius: 5px; padding: 6px 12px; "
            f"font-size: {theme.fs(11)}; }}"
            f"QPushButton:hover {{ border-color: {theme.BORDER_LIGHT}; color: {theme.TEXT_SECONDARY}; }}")
        self._revert_btn.clicked.connect(self._revert_current)
        self._revert_btn.setEnabled(False)
        editor_hdr.addWidget(self._revert_btn)

        editor_lay.addLayout(editor_hdr)

        # Tag chip toolbar
        chip_toolbar = QHBoxLayout()
        chip_toolbar.setSpacing(6)

        self._chip_search = QLineEdit()
        self._chip_search.setPlaceholderText(get_text('edit_search_tags_ph', self.lang))
        self._chip_search.setFixedHeight(26)
        theme.bind_style(self._chip_search, lambda: f"QLineEdit {{ background: {theme.BG_SURFACE}; border: 1px solid {theme.BORDER};"
            f" border-radius: 5px; padding: 2px 8px; color: {theme.TEXT_PRIMARY};"
            f" font-size: {theme.fs(11)}; }}"
            f"QLineEdit:focus {{ border-color: {theme.ORANGE}; }}")
        self._chip_search.textChanged.connect(self._filter_chips)
        chip_toolbar.addWidget(self._chip_search, stretch=1)

        self._clear_all_btn = QPushButton(get_text('edit_clear_all', self.lang))
        self._clear_all_btn.setFixedHeight(26)
        theme.bind_style(self._clear_all_btn, lambda: f"QPushButton {{ background: transparent; border: 1px solid {theme.BORDER};"
            f" color: {theme.TEXT_MUTED}; border-radius: 5px; padding: 0 10px;"
            f" font-size: {theme.fs(10)}; }}"
            f"QPushButton:hover {{ border-color: {theme.RED}; color: {theme.RED}; }}")
        self._clear_all_btn.setToolTip(get_text('edit_tt_clear_all', self.lang))
        self._clear_all_btn.clicked.connect(self._clear_all_tags)
        chip_toolbar.addWidget(self._clear_all_btn)

        editor_lay.addLayout(chip_toolbar)

        # Chip flow area
        self._chip_scroll = QScrollArea()
        self._chip_scroll.setWidgetResizable(True)
        self._chip_scroll.setFrameShape(QFrame.NoFrame)
        self._chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._chip_scroll.setMinimumHeight(80)
        self._chip_scroll.setMaximumHeight(240)
        self._chip_scroll.setStyleSheet("background: transparent; border: none;")

        self._chip_container = QWidget()
        self._chip_container.setStyleSheet("background: transparent;")
        self._chip_flow = QVBoxLayout(self._chip_container)
        self._chip_flow.setContentsMargins(2, 4, 2, 4)
        self._chip_flow.setSpacing(4)
        self._chip_scroll.setWidget(self._chip_container)
        editor_lay.addWidget(self._chip_scroll, stretch=1)

        # Single-line tag-add input with Danbooru autocomplete
        self._tag_add_input = QLineEdit()
        self._tag_add_input.setFixedHeight(34)
        self._tag_add_input.setPlaceholderText(get_text('edit_add_tag_ph', self.lang))
        theme.bind_style(self._tag_add_input, lambda: f"QLineEdit {{background:{theme.BG_SURFACE};border:1px solid {theme.BORDER};"
            f"border-radius:6px;color:{theme.TEXT_PRIMARY};font-size:{theme.fs(12)};padding:4px 8px;}}"
            f"QLineEdit:focus {{border-color:{theme.ORANGE};}}")
        self._tag_add_input.returnPressed.connect(self._add_typed_tag)
        editor_lay.addWidget(self._tag_add_input)

        # Hidden backing store — keeps the comma-separated text, synced from chips
        # Must be created before _attach_tag_add_completer so _shared_tag_model is initialized
        self.caption_edit = TagCompleterTextEdit()
        self.caption_edit.hide()
        self.caption_edit.textChanged.connect(self._on_caption_changed)

        # Attach autocomplete to _tag_add_input now that _shared_tag_model exists
        self._attach_tag_add_completer()

        right_col.addWidget(self._editor_card, stretch=45)

        bento.addLayout(right_col, stretch=1)
        root.addLayout(bento, stretch=1)

        # Tooltips
        self.load_btn.setToolTip(get_text('edit_tt_load', self.lang))
        self.save_btn.setToolTip(get_text('edit_tt_save_all', self.lang))
        self.add_tag_btn.setToolTip(get_text('edit_tt_add_tag', self.lang))
        self.remove_tag_btn.setToolTip(get_text('edit_tt_remove_tag', self.lang))
        self.replace_tag_btn.setToolTip(get_text('edit_tt_replace_tag', self.lang))
        self.caption_edit.setToolTip(get_text('edit_tt_caption', self.lang))
        self._save_one_btn.setToolTip(get_text('edit_tt_save_one', self.lang))
        self._save_next_btn.setToolTip(get_text('edit_tt_save_next', self.lang))
        self._revert_btn.setToolTip(get_text('edit_tt_revert', self.lang))
        self.image_list.setToolTip(get_text('edit_tt_image_list', self.lang))

    # ── Folder loading ──────────────────────────────────────────

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, get_text('ce_select_folder', self.lang))
        if folder:
            self._load_folder(folder)

    def reload_folder(self, folder: str):
        """Explicit folder/generation reload; drafts require confirmation."""
        self._load_folder(folder)

    def _load_folder(self, folder: str):
        if self.has_unsaved_edits():
            message = ('Kaydedilmemiş caption değişiklikleri var. Bunlar atılıp klasör açılsın mı?'
                       if self.lang == 'tr' else 'Discard unsaved caption edits and open the folder?')
            if QMessageBox.question(self, 'Edit', message, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) != QMessageBox.Yes:
                return
        root = Path(folder).resolve()
        if not root.is_dir():
            self.status_lbl.setText('Image folder does not exist.')
            return
        self._folder = str(root)
        self._current_idx = -1  # Clear before QListWidget emits any selection changes.
        self.image_list.blockSignals(True)
        self._items.clear()
        self._captions.clear()
        self._snapshots.clear()
        self._read_errors.clear()
        self._dirty_paths.clear()
        self._dirty = False
        self.image_list.clear()
        existing_captions = 0
        try:
            for f in sorted(root.rglob('*')):
                if (not f.is_file() or f.is_symlink() or f.suffix.lower() not in _IMAGE_EXTENSIONS
                        or any(part.startswith('.lh-') for part in f.relative_to(root).parts)):
                    continue
                cap_path = f.with_suffix('.txt')
                cp = str(cap_path)
                self._items.append((str(f), cp))
                try:
                    snapshot = read_caption_snapshot(f)
                    self._snapshots[cp] = snapshot
                    self._captions[cp] = snapshot.text
                    existing_captions += int(snapshot.raw is not None)
                except (OSError, ValueError, RuntimeError) as exc:
                    self._read_errors[cp] = str(exc)
                    self._captions[cp] = ''
                item = QListWidgetItem(f.name)
                # Studio loads only visible thumbnails off the GUI thread.
                if not hasattr(self, '_studio_layout'):
                    try:
                        pix = QPixmap(str(f))
                        if not pix.isNull():
                            item.setIcon(QIcon(pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                    except Exception:
                        pass
                self.image_list.addItem(item)
        finally:
            self.image_list.blockSignals(False)
        self.caption_edit.blockSignals(True)
        self.caption_edit.clear()
        self.caption_edit.blockSignals(False)
        self.preview_lbl.clear()
        self.filename_lbl.setText('')
        if hasattr(self, '_chip_scroll'):
            self._build_chips('')
        has = bool(self._items)
        for button in (self.save_btn, self.add_tag_btn, self.remove_tag_btn, self.replace_tag_btn,
                       self._save_one_btn, self._save_next_btn, self._revert_btn):
            button.setEnabled(has)
        self.status_lbl.setText(get_text('ce_loaded', self.lang).format(len(self._items), existing_captions))
        if self._items:
            self.image_list.setCurrentRow(0)
        if hasattr(self, '_thumb_grid') and self._thumb_grid.isVisible():
            self._thumb_grid.set_paths([p for p, _ in self._items])
        if self._read_errors:
            self.status_lbl.setText(f'{len(self._read_errors)} caption read error(s); unreadable files are protected.')
        if hasattr(self, '_studio_layout'):
            self._studio_layout.after_folder_loaded()
        self.folder_changed.emit(self._folder)

    def _commit_current(self):
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self._captions[cap_path] = self.caption_edit.toPlainText()
            self._mark_dirty(cap_path)

    def _mark_dirty(self, cap_path):
        baseline = self._snapshots.get(cap_path)
        clean_text = baseline.text if baseline is not None else ''
        if self._captions.get(cap_path, '') != clean_text:
            self._dirty_paths.add(cap_path)
        else:
            self._dirty_paths.discard(cap_path)
        self._dirty = bool(self._dirty_paths)

    def has_unsaved_edits(self):
        self._commit_current()
        for cp in self._captions:
            self._mark_dirty(cp)
        return self._dirty

    def _report_save_errors(self, errors):
        if not errors:
            return
        title = 'Caption kaydedilemedi' if self.lang == 'tr' else 'Caption could not be saved'
        message = ('Başarısız dosyaların taslakları korundu.\n' if self.lang == 'tr'
                   else 'Drafts for the failed files were kept.\n')
        self.status_lbl.setText(f'{len(errors)} caption error(s)')
        QMessageBox.warning(self, title, message + '\n'.join(errors[:5]))

    def refresh_saved_captions(self, paths=None):
        """Read saved changes without reloading the folder or overwriting any draft."""
        self._commit_current()
        wanted = None if paths is None else {path_key(p) for p in paths}
        conflicts, updated = [], []
        for image, cp in self._items:
            if wanted is not None and path_key(cp) not in wanted:
                continue
            if cp in self._dirty_paths:
                conflicts.append(cp)
                continue
            try:
                snapshot = read_caption_snapshot(image)
            except (OSError, ValueError, RuntimeError) as exc:
                self._read_errors[cp] = str(exc)
                conflicts.append(cp)
                continue
            if self._captions.get(cp) != snapshot.text:
                updated.append(cp)
            self._snapshots[cp] = snapshot
            self._captions[cp] = snapshot.text
            self._read_errors.pop(cp, None)
            self._mark_dirty(cp)
        if 0 <= self._current_idx < len(self._items):
            if self._items[self._current_idx][1] in updated:
                self._refresh_editor()
        if conflicts:
            self.status_lbl.setText('Taslak/okuma çakışması: mevcut düzenlemeler korundu.' if self.lang == 'tr'
                                    else 'Draft/read conflict: existing edits were preserved.')
        return updated

    def _attach_tag_add_completer(self):
        """Attach Danbooru completer to _tag_add_input (call after caption_edit is created)."""
        _c = QCompleter(_get_shared_tag_model(), self._tag_add_input)
        _c.setFilterMode(Qt.MatchStartsWith)
        _c.setCaseSensitivity(Qt.CaseInsensitive)
        _c.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
        _c.setCompletionMode(QCompleter.PopupCompletion)
        _c.setMaxVisibleItems(10)
        self._tag_add_completer = _c
        theme.bind_style(_c.popup(), lambda self=self: self._tag_popup_style())
        _c.activated[str].connect(self._on_tag_add_completed)
        self._tag_add_input.setCompleter(_c)
        # Disconnect Qt's auto full-text connections — we manage prefix per-word ourselves
        try:
            self._tag_add_input.textEdited.disconnect(_c.setCompletionPrefix)
        except TypeError:
            pass
        try:
            self._tag_add_input.textEdited.disconnect(_c.complete)
        except TypeError:
            pass
        self._tag_add_input.textEdited.connect(self._on_tag_add_edited)

    def _tag_popup_style(self) -> str:
        return (
            f"QListView {{background:{theme.BG_ELEVATED};color:{theme.TEXT_PRIMARY};"
            f"border:1px solid {theme.get_accent()};border-radius:4px;font-size:{theme.fs(11)};padding:2px;}}"
            f"QListView::item:selected {{background:{theme.get_accent()};color:#ffffff;}}"
            f"QListView::item:hover {{background:{theme.BG_HOVER};}}"
        )

    def _on_tag_add_edited(self, text: str):
        """Drive autocomplete prefix from the word after the last comma."""
        c = self._tag_add_input.completer()
        if not c:
            return
        parts = text.split(',')
        prefix = parts[-1].strip()
        c.setCompletionPrefix(prefix)
        if prefix:
            c.complete()

    def _on_tag_add_completed(self, completion: str):
        """Insert completed tag, add it to caption and clear the input."""
        text = self._tag_add_input.text()
        parts = text.split(',')
        parts[-1] = completion
        new_text = ', '.join(p.strip() for p in parts if p.strip())
        self._tag_add_input.blockSignals(True)
        self._tag_add_input.setText(new_text)
        self._tag_add_input.blockSignals(False)
        self._add_typed_tag()

    def _add_typed_tag(self):
        """Add tag(s) from the QLineEdit input field (supports comma-separated)."""
        text = self._tag_add_input.text().strip()
        if not text:
            return
        new_tags = [t.strip() for t in text.split(',') if t.strip()]
        current = self.caption_edit.toPlainText().strip()
        existing = [t.strip() for t in current.split(',') if t.strip()]
        merged = existing + [t for t in new_tags if t not in existing]
        self._set_caption_text(', '.join(merged))
        self._tag_add_input.clear()

    def _on_caption_changed(self):
        self._commit_current()
        if hasattr(self, '_chip_scroll') and not getattr(self, '_syncing_input', False):
            self._build_chips(self.caption_edit.toPlainText())

    # ── Tag chip helpers ────────────────────────────────────────

    def _build_chips(self, text: str):
        """Parse caption text into chip widgets."""
        # Clear existing chips
        while self._chip_flow.count():
            item = self._chip_flow.takeAt(0)
            if item.layout():
                while item.layout().count():
                    w = item.layout().takeAt(0).widget()
                    if w:
                        w.deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        raw = text.strip()
        if not raw:
            placeholder = QLabel(get_text('edit_no_tags', self.lang))
            theme.bind_style(placeholder, lambda: f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; background: transparent; border: none;")
            self._chip_flow.addWidget(placeholder)
            self._chip_flow.addStretch()
            return

        tags = [t.strip() for t in raw.split(',') if t.strip()]

        row_lay = QHBoxLayout()
        row_lay.setSpacing(4)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_count = 0

        filter_text = self._chip_search.text().lower().strip() if hasattr(self, '_chip_search') else ''

        for tag in tags:
            if filter_text and filter_text not in tag.lower():
                continue

            chip = self._make_chip(tag)
            row_lay.addWidget(chip)
            row_count += 1

            if row_count >= 4:
                row_lay.addStretch()
                self._chip_flow.addLayout(row_lay)
                row_lay = QHBoxLayout()
                row_lay.setSpacing(4)
                row_lay.setContentsMargins(0, 0, 0, 0)
                row_count = 0

        if row_count > 0:
            row_lay.addStretch()
            self._chip_flow.addLayout(row_lay)

        self._chip_flow.addStretch()

    def _make_chip(self, tag: str) -> QFrame:
        chip = QFrame()
        theme.bind_style(chip, lambda: f"QFrame {{ background: {theme.BG_SURFACE}; border: 1px solid {theme.BORDER};"
            f" border-radius: 12px; }}"
            f"QFrame:hover {{ border-color: {theme.ORANGE}44; }}")
        lay = QHBoxLayout(chip)
        lay.setContentsMargins(8, 3, 4, 3)
        lay.setSpacing(4)

        lbl = QLabel(tag)
        theme.bind_style(lbl, lambda: f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(11)}; background: transparent; border: none;")
        lay.addWidget(lbl)

        rm_btn = QPushButton("×")
        rm_btn.setFixedSize(16, 16)
        theme.bind_style(rm_btn, lambda: f"QPushButton {{ background: transparent; border: none; color: {theme.TEXT_MUTED};"
            f" font-size: 14px; font-weight: 700; padding: 0; }}"
            f"QPushButton:hover {{ color: {theme.RED}; }}")
        rm_btn.setToolTip(f"Remove '{tag}'")
        rm_btn.clicked.connect(lambda checked=False, t=tag: self._remove_chip_tag(t))
        lay.addWidget(rm_btn)

        return chip

    def _remove_chip_tag(self, tag: str):
        """Remove a single tag from the caption."""
        current = self.caption_edit.toPlainText()
        tags = [t.strip() for t in current.split(',')]
        tags = [t for t in tags if t and t != tag]
        new_text = ', '.join(tags)
        self._set_caption_text(new_text)

    def _clear_all_tags(self):
        """Remove all tags from caption."""
        self._set_caption_text("")

    def _set_caption_text(self, text: str):
        """Update all caption storage locations atomically."""
        self._syncing_input = True
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(text)
        self.caption_edit.blockSignals(False)
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            self._captions[cap_path] = text
        if 0 <= self._current_idx < len(self._items):
            self._mark_dirty(self._items[self._current_idx][1])
        self._build_chips(text)
        self._syncing_input = False

    def _filter_chips(self, _):
        """Rebuild chip display with current filter."""
        self._build_chips(self.caption_edit.toPlainText())

    def _on_thumb_clicked(self, path: str):
        """Select image by path when thumbnail is clicked in grid view."""
        for i, (img_path, _) in enumerate(self._items):
            if img_path == path:
                self.image_list.setCurrentRow(i)
                break

    # ── Item selection + per-item save ─────────────────────────

    def _on_item_selected(self, row: int):
        if self._current_idx >= 0:
            self._commit_current()
        if row < 0 or row >= len(self._items):
            self._current_idx = -1
            self.preview_lbl.clear()
            blocked = self.caption_edit.blockSignals(True)
            self.caption_edit.clear()
            self.caption_edit.blockSignals(blocked)
            self.filename_lbl.setText("")
            if hasattr(self, '_studio_layout'):
                self._studio_layout.clear()
            return
        self._current_idx = row
        img_path, cap_path = self._items[row]
        if hasattr(self, '_studio_layout'):
            self._studio_layout.show_image(img_path)
        else:
            try:
                pix = QPixmap(img_path)
                if not pix.isNull():
                    self.preview_lbl.setPixmap(pix.scaled(
                        self.preview_lbl.width(), self.preview_lbl.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception:
                self.preview_lbl.setText("(preview error)")
        self.filename_lbl.setText(Path(img_path).name)
        cap_text = self._captions.get(cap_path, "")
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(cap_text)
        self.caption_edit.blockSignals(False)
        if hasattr(self, '_chip_scroll'):
            self._build_chips(cap_text)

    # ── Save ────────────────────────────────────────────────────

    def _save_current(self):
        """Publish only successful saves; failure keeps the draft and selection."""
        if not 0 <= self._current_idx < len(self._items):
            return False
        self._commit_current()
        _, cp = self._items[self._current_idx]
        try:
            if cp not in self._snapshots or cp in self._read_errors:
                raise RuntimeError(self._read_errors.get(cp, 'Caption baseline is unavailable. Reload/revert first.'))
            self._snapshots[cp] = save_edited_caption(self._snapshots[cp], self._captions[cp])
        except Exception as exc:
            self._report_save_errors([f'{cp}: {exc}'])
            return False
        self._mark_dirty(cp)
        self.status_lbl.setText(f'Saved: {Path(cp).name}')
        self.captions_saved.emit([str(Path(cp).absolute())])
        return True

    def _save_and_next(self):
        """Never advance on a failed save."""
        if not self._save_current():
            return
        nxt = self._current_idx + 1
        if nxt < len(self._items):
            self.image_list.setCurrentRow(nxt)

    def _revert_current(self):
        """Discard only this draft, not drafts belonging to other images."""
        if 0 <= self._current_idx < len(self._items):
            image, cp = self._items[self._current_idx]
            try:
                snapshot = read_caption_snapshot(image)
            except Exception as exc:
                self._report_save_errors([f'{cp}: {exc}'])
                return False
            self._snapshots[cp] = snapshot
            self._read_errors.pop(cp, None)
            self._captions[cp] = snapshot.text
            self._refresh_editor()
            # Also refresh the opposite tab if an external writer changed the disk.
            self.captions_saved.emit([str(Path(cp).absolute())])
            return True
        return False

    def _save_all(self):
        self._commit_current()
        saved, errors = [], []
        for cp, text in list(self._captions.items()):
            try:
                if cp not in self._snapshots or cp in self._read_errors:
                    raise RuntimeError(self._read_errors.get(cp, 'Caption baseline is unavailable. Reload/revert first.'))
                self._snapshots[cp] = save_edited_caption(self._snapshots[cp], text)
                self._mark_dirty(cp)
                saved.append(str(Path(cp).absolute()))
            except Exception as exc:
                errors.append(f'{cp}: {exc}')
        if saved:
            self.captions_saved.emit(saved)
        if errors:
            self._report_save_errors(errors)
        else:
            self.status_lbl.setText(get_text('ce_saved', self.lang))
        return not errors

    # ── Bulk operations ─────────────────────────────────────────

    def _refresh_editor(self):
        for cp in self._captions:
            self._mark_dirty(cp)
        if 0 <= self._current_idx < len(self._items):
            _, cap_path = self._items[self._current_idx]
            text = self._captions.get(cap_path, '')
            if self.caption_edit.toPlainText() != text:
                cursor = self.caption_edit.textCursor()
                position, anchor = cursor.position(), cursor.anchor()
                scroll = self.caption_edit.verticalScrollBar().value()
                old_blocked = self.caption_edit.blockSignals(True)
                try:
                    self.caption_edit.setPlainText(text)
                    cursor = self.caption_edit.textCursor()
                    limit = self.caption_edit.document().characterCount() - 1
                    cursor.setPosition(min(anchor, limit))
                    cursor.setPosition(min(position, limit), QTextCursor.KeepAnchor)
                    self.caption_edit.setTextCursor(cursor)
                    self.caption_edit.verticalScrollBar().setValue(scroll)
                finally:
                    self.caption_edit.blockSignals(old_blocked)
            if hasattr(self, '_chip_scroll'):
                self._build_chips(text)

    def _bulk_add_tag(self):
        raw, ok = QInputDialog.getText(
            self, get_text('ce_add_tag_title', self.lang),
            get_text('ce_add_tag_prompt', self.lang))
        if not ok or not raw.strip():
            return
        new_tags = [t.strip() for t in raw.split(',') if t.strip()]
        self._commit_current()
        for cp in list(self._captions):
            existing = [t.strip() for t in self._captions[cp].split(',') if t.strip()]
            merged = [t for t in new_tags if t not in existing] + existing
            self._captions[cp] = ', '.join(merged)
        self._refresh_editor()
        label = ', '.join(new_tags)
        self.status_lbl.setText(
            get_text('ce_add_tag_result', self.lang).format(label, len(self._captions)))

    def _bulk_remove_tag(self):
        raw, ok = QInputDialog.getText(
            self, get_text('ce_remove_tag_title', self.lang),
            get_text('ce_remove_tag_prompt', self.lang))
        if not ok or not raw.strip():
            return
        remove_set = {t.strip().lower() for t in raw.split(',') if t.strip()}
        self._commit_current()
        removed = 0
        for cp in list(self._captions):
            parts = [t.strip() for t in self._captions[cp].split(',')]
            new_parts = [t for t in parts if t.lower() not in remove_set]
            if len(new_parts) != len(parts):
                removed += 1
            self._captions[cp] = ', '.join(p for p in new_parts if p)
        self._refresh_editor()
        label = ', '.join(t for t in raw.split(',') if t.strip())
        self.status_lbl.setText(
            get_text('ce_remove_tag_result', self.lang).format(label, removed))

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

        def _s(widget, text):
            if widget and hasattr(widget, 'setText'):
                widget.setText(text)

        def _tt(widget, text):
            if widget and hasattr(widget, 'setToolTip'):
                widget.setToolTip(text)

        def _ph(widget, text):
            if widget and hasattr(widget, 'setPlaceholderText'):
                widget.setPlaceholderText(text)

        # Toolbar buttons (built with get_text but never refreshed)
        _s(getattr(self, 'load_btn',        None), get_text('ce_load_folder', lang))
        _s(getattr(self, 'save_btn',        None), get_text('ce_save_all', lang))
        _s(getattr(self, 'add_tag_btn',     None), get_text('ce_add_tag', lang))
        _s(getattr(self, 'remove_tag_btn',  None), get_text('ce_remove_tag', lang))
        _s(getattr(self, 'replace_tag_btn', None), get_text('ce_replace_tag', lang))

        # Editor labels / buttons
        _s(getattr(self, '_list_hdr',       None), get_text('edit_images_hdr', lang))
        _s(getattr(self, '_editor_title',   None), get_text('edit_tags_caption', lang))
        _s(getattr(self, '_save_one_btn',   None), get_text('edit_save', lang))
        _s(getattr(self, '_save_next_btn',  None), get_text('edit_save_next', lang))
        _s(getattr(self, '_revert_btn',     None), get_text('edit_revert', lang))
        _s(getattr(self, '_clear_all_btn',  None), get_text('edit_clear_all', lang))

        # Status label — preserve loaded-count message when a folder is open
        if getattr(self, 'status_lbl', None):
            if self._items:
                existing = sum(1 for v in self._captions.values() if v.strip())
                self.status_lbl.setText(
                    get_text('ce_loaded', lang).format(len(self._items), existing))
            else:
                self.status_lbl.setText(get_text('ce_no_images', lang))

        # Placeholders
        _ph(getattr(self, '_chip_search',   None), get_text('edit_search_tags_ph', lang))
        _ph(getattr(self, '_tag_add_input', None), get_text('edit_add_tag_ph', lang))

        # Tooltips
        _tt(getattr(self, 'load_btn',        None), get_text('edit_tt_load', lang))
        _tt(getattr(self, 'save_btn',        None), get_text('edit_tt_save_all', lang))
        _tt(getattr(self, 'add_tag_btn',     None), get_text('edit_tt_add_tag', lang))
        _tt(getattr(self, 'remove_tag_btn',  None), get_text('edit_tt_remove_tag', lang))
        _tt(getattr(self, 'replace_tag_btn', None), get_text('edit_tt_replace_tag', lang))
        _tt(getattr(self, 'caption_edit',    None), get_text('edit_tt_caption', lang))
        _tt(getattr(self, '_save_one_btn',   None), get_text('edit_tt_save_one', lang))
        _tt(getattr(self, '_save_next_btn',  None), get_text('edit_tt_save_next', lang))
        _tt(getattr(self, '_revert_btn',     None), get_text('edit_tt_revert', lang))
        _tt(getattr(self, 'image_list',      None), get_text('edit_tt_image_list', lang))
        _tt(getattr(self, '_clear_all_btn',  None), get_text('edit_tt_clear_all', lang))

        # Rebuild chips so the "no tags" placeholder follows the language
        if hasattr(self, 'caption_edit') and hasattr(self, '_chip_scroll'):
            self._build_chips(self.caption_edit.toPlainText())
        if hasattr(self, '_studio_layout'):
            self._studio_layout.update_language(lang)

    def refresh_styles(self):
        """Refresh existing controls, including dynamically added children."""
        return theme.refresh_styles(self)


# ════════════════════════════════════════════════════════════════
#  _QualityTab — caption quality audit
# ════════════════════════════════════════════════════════════════

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QMessageBox

class _QualityTab(QWidget):
    """
    Caption quality audit: scan for missing/empty captions, missing
    trigger words, low tag count, duplicate tags. Bulk-fix actions.
    """

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._folder: Optional[str] = None
        self._issues: list = []
        self._init_ui()

    # ── UI ────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        _t = lambda k: get_text(k, self.lang)

        # ── Toolbar ──
        tb = QHBoxLayout(); tb.setSpacing(8)

        self._trigger_edit = QLineEdit()
        self._trigger_edit.setPlaceholderText(_t('quality_trigger_ph'))
        theme.bind_style(self._trigger_edit, theme.line_edit_compact)
        self._trigger_edit.setFixedWidth(180)
        self._trigger_lbl = QLabel(_t('quality_trigger'))
        tb.addWidget(self._trigger_lbl); tb.addWidget(self._trigger_edit)

        self._min_tags_lbl = min_lbl = QLabel(_t('quality_min_tags'))
        theme.bind_style(min_lbl, theme.label_default)
        self._min_tags_spin = QSpinBox()
        self._min_tags_spin.setRange(0, 50); self._min_tags_spin.setValue(3)
        theme.bind_style(self._min_tags_spin, theme.spinbox_compact)
        tb.addWidget(min_lbl); tb.addWidget(self._min_tags_spin)

        tb.addStretch()

        self._run_btn = QPushButton(_t('quality_run_audit'))
        theme.bind_style(self._run_btn, theme.btn_primary)
        self._run_btn.clicked.connect(self._run_audit)
        tb.addWidget(self._run_btn)

        self._add_trigger_btn = QPushButton(_t('quality_add_trigger'))
        theme.bind_style(self._add_trigger_btn, theme.btn_secondary)
        self._add_trigger_btn.clicked.connect(self._bulk_add_trigger)
        self._add_trigger_btn.setEnabled(False)
        tb.addWidget(self._add_trigger_btn)

        self._delete_btn = QPushButton(_t('quality_delete_sel'))
        theme.bind_style(self._delete_btn, theme.btn_danger)
        self._delete_btn.clicked.connect(self._delete_selected)
        self._delete_btn.setEnabled(False)
        tb.addWidget(self._delete_btn)

        root.addLayout(tb)

        # ── Stat cards ──
        stats_row = QHBoxLayout(); stats_row.setSpacing(8)
        self._cards = {}
        for key, label in [('total', _t('quality_stat_total')), ('missing', _t('quality_stat_missing')),
                            ('empty', _t('quality_stat_empty')), ('low_tags', _t('quality_stat_lowtags')),
                            ('no_trigger', _t('quality_stat_notrigger')), ('dup_tags', _t('quality_stat_duptags'))]:
            card = QFrame()
            card.setFixedSize(100, 52)
            theme.bind_style(card, lambda: f"QFrame{{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:8px;}}")
            cl = QVBoxLayout(card); cl.setContentsMargins(6, 4, 6, 4)
            val = QLabel("0"); val.setAlignment(Qt.AlignCenter)
            theme.bind_style(val, lambda: f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(15)};font-weight:700;border:none;")
            lbl = QLabel(label); lbl.setAlignment(Qt.AlignCenter)
            theme.bind_style(lbl, lambda: f"color:{theme.TEXT_SECONDARY};font-size:{theme.fs(9)};border:none;")
            cl.addWidget(val); cl.addWidget(lbl)
            card._val = val
            card._lbl = lbl
            card._stat_key = key
            self._cards[key] = card
            stats_row.addWidget(card)
        stats_row.addStretch()
        root.addLayout(stats_row)

        # ── Table ──
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels([
            _t('quality_col_image'), _t('quality_col_issues'),
            _t('quality_col_tags'), _t('quality_col_caption')])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(2, 55)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        theme.bind_style(self._table, lambda: f"""
            QTableWidget{{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:8px;}}
            QHeaderView::section{{background:{theme.BG_CARD};color:{theme.TEXT_SECONDARY};
                border:none;padding:6px 8px;font-size:{theme.fs(11)};font-weight:600;}}
            QTableWidget::item:selected{{background:{theme.ORANGE};color:#fff;}}
        """)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._table, 1)

        # ── Status ──
        self._status = QLabel(_t('quality_status_initial'))
        theme.bind_style(self._status, lambda: f"color:{theme.TEXT_SECONDARY};font-size:{theme.fs(11)};padding:2px 0;")
        root.addWidget(self._status)

    # ── Audit ─────────────────────────────────────────────────

    def reload_folder(self, folder: str):
        self._folder = folder
        self._status.setText(get_text('quality_folder_set', self.lang).format(Path(folder).name))

    def _run_audit(self):
        if not self._folder or not Path(self._folder).exists():
            self._status.setText(get_text('quality_status_nofolder', self.lang))
            return

        from src.core.dataset_scanner import scan_dataset, validate_captions
        pairs = scan_dataset(self._folder)
        trigger = self._trigger_edit.text().strip()
        min_tags = self._min_tags_spin.value()
        self._issues = validate_captions(pairs, trigger_word=trigger, min_tags=min_tags)

        self._populate_table()
        self._update_stats(len(pairs))
        self._add_trigger_btn.setEnabled(bool(self._issues) and bool(trigger))
        self._status.setText(
            get_text('quality_audit_complete', self.lang).format(len(self._issues), len(pairs)))

    def _populate_table(self):
        from src.core.dataset_scanner import (
            ISSUE_MISSING_CAPTION, ISSUE_EMPTY_CAPTION, ISSUE_NO_TRIGGER,
            ISSUE_LOW_TAG_COUNT, ISSUE_DUPLICATE_TAGS
        )
        _ICONS = {
            ISSUE_MISSING_CAPTION: get_text('quality_issue_missing', self.lang),
            ISSUE_EMPTY_CAPTION:   get_text('quality_issue_empty', self.lang),
            ISSUE_NO_TRIGGER:      get_text('quality_issue_no_trigger', self.lang),
            ISSUE_LOW_TAG_COUNT:   get_text('quality_issue_low_tags', self.lang),
            ISSUE_DUPLICATE_TAGS:  get_text('quality_issue_dup_tags', self.lang),
        }
        self._table.setRowCount(0)
        for issue in self._issues:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(issue.image.name))
            self._table.setItem(row, 1, QTableWidgetItem(
                ', '.join(_ICONS.get(i, i) for i in issue.issues)))
            self._table.setItem(row, 2, QTableWidgetItem(str(issue.tag_count)))
            preview = issue.caption_text[:80] + ('…' if len(issue.caption_text) > 80 else '')
            self._table.setItem(row, 3, QTableWidgetItem(preview))
            # Store issue ref for bulk ops
            self._table.item(row, 0).setData(Qt.UserRole, issue)

    def _update_stats(self, total: int):
        from src.core.dataset_scanner import (
            ISSUE_MISSING_CAPTION, ISSUE_EMPTY_CAPTION, ISSUE_NO_TRIGGER,
            ISSUE_LOW_TAG_COUNT, ISSUE_DUPLICATE_TAGS
        )
        def _count(key): return sum(1 for i in self._issues if key in i.issues)
        self._cards['total']._val.setText(str(total))
        self._cards['missing']._val.setText(str(_count(ISSUE_MISSING_CAPTION)))
        self._cards['empty']._val.setText(str(_count(ISSUE_EMPTY_CAPTION)))
        self._cards['low_tags']._val.setText(str(_count(ISSUE_LOW_TAG_COUNT)))
        self._cards['no_trigger']._val.setText(str(_count(ISSUE_NO_TRIGGER)))
        self._cards['dup_tags']._val.setText(str(_count(ISSUE_DUPLICATE_TAGS)))

    # ── Bulk actions ──────────────────────────────────────────

    def _bulk_add_trigger(self):
        trigger = self._trigger_edit.text().strip()
        if not trigger:
            return
        from src.core.dataset_scanner import ISSUE_NO_TRIGGER, ISSUE_EMPTY_CAPTION
        fixed = 0
        errors = []
        for issue in self._issues:
            if ISSUE_NO_TRIGGER not in issue.issues and ISSUE_EMPTY_CAPTION not in issue.issues:
                continue
            caption_path = issue.caption
            try:
                if caption_path is None or not caption_path.exists():
                    # Create a new .txt
                    caption_path = issue.image.with_suffix('.txt')
                    caption_path.write_text(trigger, encoding='utf-8')
                else:
                    current = caption_path.read_text(encoding='utf-8').strip()
                    new_text = f"{trigger}, {current}" if current else trigger
                    caption_path.write_text(new_text, encoding='utf-8')
                fixed += 1
            except Exception as e:
                errors.append(f"{issue.image.name}: {e}")

        msg = get_text('quality_added_trigger', self.lang).format(fixed)
        if errors:
            msg += "\n" + get_text('quality_added_errors', self.lang).format(
                len(errors), '; '.join(errors[:3]))
        self._status.setText(msg)
        self._run_audit()

    def _delete_selected(self):
        selected = self._table.selectedItems()
        rows = sorted({self._table.row(i) for i in selected}, reverse=True)
        if not rows:
            return
        reply = QMessageBox.question(
            self, get_text('review_confirm_delete_title', self.lang),
            get_text('quality_confirm_delete_msg', self.lang).format(len(rows)),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for row in rows:
            issue = self._table.item(row, 0).data(Qt.UserRole)
            try:
                issue.image.unlink(missing_ok=True)
                if issue.caption and issue.caption.exists():
                    issue.caption.unlink(missing_ok=True)
            except Exception:
                pass
            self._table.removeRow(row)
        self._status.setText(get_text('quality_deleted', self.lang).format(len(rows)))

    def _on_selection_changed(self):
        has_sel = bool(self._table.selectedItems())
        self._delete_btn.setEnabled(has_sel)

    # ── Theme ─────────────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        _t = lambda k: get_text(k, lang)
        self._trigger_lbl.setText(_t('quality_trigger'))
        self._trigger_edit.setPlaceholderText(_t('quality_trigger_ph'))
        self._min_tags_lbl.setText(_t('quality_min_tags'))
        self._run_btn.setText(_t('quality_run_audit'))
        self._add_trigger_btn.setText(_t('quality_add_trigger'))
        self._delete_btn.setText(_t('quality_delete_sel'))
        _stat_keys = {
            'total': 'quality_stat_total', 'missing': 'quality_stat_missing',
            'empty': 'quality_stat_empty', 'low_tags': 'quality_stat_lowtags',
            'no_trigger': 'quality_stat_notrigger', 'dup_tags': 'quality_stat_duptags',
        }
        for key, card in self._cards.items():
            card._lbl.setText(_t(_stat_keys[key]))
        self._table.setHorizontalHeaderLabels([
            _t('quality_col_image'), _t('quality_col_issues'),
            _t('quality_col_tags'), _t('quality_col_caption')])
        # Re-localize the issue table rows if an audit has already run
        if self._issues:
            self._populate_table()
        # Status: only reset idle text; preserve an active audit/result message
        if not self._folder:
            self._status.setText(_t('quality_status_initial'))

    def refresh_styles(self):
        """Refresh existing controls, including dynamically added children."""
        return theme.refresh_styles(self)


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
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        self.setLayout(root)

        # Keep hidden tools from inflating the active Studio editor's size hints.
        if getattr(theme, '_studio_design', False):
            from src.ui.design_system.widgets import ActiveTabWidget
            self.tabs = ActiveTabWidget()
        else:
            self.tabs = QTabWidget()
        self.tabs.setObjectName("caption_tabs")
        self.generate_tab = _GenerateTab(self.lang, self)
        self.edit_tab = _EditTab(self.lang, self)
        self.quality_tab = _QualityTab(self.lang, self)
        self.clothing_tab = ClothingTaggerWidget(self.lang, self)
        from src.ui.dataset_balance_widget import DatasetBalanceWidget
        from src.ui.checkpoint_compare_widget import CheckpointCompareWidget
        self.balance_tab = DatasetBalanceWidget(self.lang, self, store=self.clothing_tab.store)
        self.compare_tab = CheckpointCompareWidget(self.lang, self)

        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setWidget(self.generate_tab)
        gen_scroll.setFrameShape(QFrame.NoFrame)
        gen_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.tabs.addTab(gen_scroll, "Generate")
        self.tabs.addTab(self.edit_tab, "Edit")
        self.tabs.addTab(self.quality_tab, get_text('quality_tab', self.lang))
        self.tabs.addTab(self.clothing_tab, "Kıyafetler" if self.lang == 'tr' else "Clothing")
        self.tabs.addTab(self.balance_tab, "Veri dengesi" if self.lang == 'tr' else "Data balance")
        self.tabs.addTab(self.compare_tab, "Checkpoint karşılaştır" if self.lang == 'tr' else "Compare checkpoints")
        theme.bind_style(self.tabs, lambda self=self: self._tab_style())
        root.addWidget(self.tabs, stretch=1)

        self.generate_tab.folder_changed.connect(self.edit_tab.reload_folder)
        self.generate_tab.folder_changed.connect(self.quality_tab.reload_folder)
        self.generate_tab.captioning_finished.connect(self._on_generation_done)
        self.edit_tab.folder_changed.connect(self.clothing_tab.reload_folder)
        self.edit_tab.captions_saved.connect(self.clothing_tab.refresh_captions)
        self.clothing_tab.captions_changed.connect(self._on_clothing_changed)
        self.edit_tab.captions_saved.connect(self._invalidate_caption_dependents)
        self.clothing_tab.captions_changed.connect(self._invalidate_caption_dependents)
        self.tabs.currentChanged.connect(self._on_tab_activated)
        self.clothing_tab.busy_changed.connect(self._on_clothing_busy)
        self.edit_tab.folder_changed.connect(self.balance_tab.reload_folder)
        self.generate_tab.folder_changed.connect(self.balance_tab.reload_folder)
        self.balance_tab.busy_changed.connect(lambda busy: self._on_tools_busy(self.balance_tab, busy))
        self.compare_tab.busy_changed.connect(lambda busy: self._on_tools_busy(self.compare_tab, busy))

    def has_unsaved_caption_edits(self):
        return self.edit_tab.has_unsaved_edits()

    def any_tool_busy(self):
        return any(getattr(self, name, None) is not None and getattr(self, name).is_busy()
                   for name in ('clothing_tab', 'balance_tab', 'compare_tab'))

    def _on_tools_busy(self, owner, busy):
        pages = [self.generate_tab, self.edit_tab, self.quality_tab, self.clothing_tab,
                 self.balance_tab, self.compare_tab]
        window = self.window()
        pages += [getattr(window, name, None) for name in
                  ('char_sort_page', 'upscale_page', 'training_page', 'review_grid_page', 'tag_freq_page', 'start_btn')]
        for page in pages:
            if page is None or page is owner:
                continue
            if busy:
                page._studio_previous_enabled = page.isEnabled()
                page.setEnabled(False)
            else:
                page.setEnabled(getattr(page, '_studio_previous_enabled', True))

    def _on_clothing_busy(self, busy):
        for name in ('balance_tab', 'compare_tab'):
            page = getattr(self, name, None)
            if page is not None:
                page.setEnabled(not busy)
        self.generate_tab.setEnabled(not busy)
        self.edit_tab.setEnabled(not busy)
        self.quality_tab.setEnabled(not busy)
        window = self.window()
        for name in ('char_sort_page', 'upscale_page', 'training_page', 'review_grid_page', 'tag_freq_page'):
            page = getattr(window, name, None)
            if page is not None:
                if busy:
                    page._clothing_previous_enabled = page.isEnabled()
                    page.setEnabled(False)
                else:
                    page.setEnabled(getattr(page, '_clothing_previous_enabled', True))

    def _on_clothing_changed(self, paths=None):
        self.edit_tab.refresh_saved_captions(paths)

    def _invalidate_caption_dependents(self, paths=None):
        """Do not leave a previous caption-derived balance report looking current."""
        page = getattr(self, 'balance_tab', None)
        if page is None or page.report is None:
            return
        wanted = None if paths is None else {path_key(p) for p in paths}
        if wanted is None or any(path_key(Path(item.path).with_suffix('.txt')) in wanted
                                 for item in page.report['items']):
            page._invalidate_scan()
            page.status.setText('Caption değişti; veri dengesini yeniden tara.' if self.lang == 'tr'
                                else 'Captions changed; rescan the dataset balance.')

    def _on_tab_activated(self, _index):
        current = self.tabs.currentWidget()
        if current is self.clothing_tab:
            image = self.clothing_tab._active_image
            if image:
                self.clothing_tab.refresh_captions([str(Path(image).with_suffix('.txt'))])
        elif current is self.edit_tab:
            index = self.edit_tab._current_idx
            if 0 <= index < len(self.edit_tab._items):
                self.edit_tab.refresh_saved_captions([self.edit_tab._items[index][1]])

    def _on_generation_done(self, folder: str):
        if folder:
            self.edit_tab.reload_folder(folder)

    def update_language(self, lang: str):
        self.lang = lang
        # tabs keep fixed English labels to match objectName test expectations
        self.generate_tab.update_language(lang)
        self.edit_tab.update_language(lang)
        self.quality_tab.update_language(lang)
        self.clothing_tab.update_language(lang)
        self.balance_tab.update_language(lang)
        self.compare_tab.update_language(lang)
        self.tabs.setTabText(4, "Veri dengesi" if lang == 'tr' else "Data balance")
        self.tabs.setTabText(5, "Checkpoint karşılaştır" if lang == 'tr' else "Compare checkpoints")
        self.tabs.setTabText(3, "Kıyafetler" if lang == 'tr' else "Clothing")

    def _tab_style(self) -> str:
        return f"""
            QTabWidget::pane {{
                border: none;
                background: {theme.BG_WINDOW};
            }}
            QTabBar {{
                background: {theme.BG_WINDOW};
                border-bottom: 1px solid {theme.BORDER};
            }}
            QTabBar::tab {{
                background: transparent;
                color: {theme.TEXT_MUTED};
                padding: 10px 24px;
                min-width: 80px;
                border-bottom: 2px solid transparent;
                font-size: {theme.fs(12)};
                font-weight: 700;
                letter-spacing: 0.04em;
            }}
            QTabBar::tab:selected {{
                color: {theme.ORANGE};
                border-bottom: 2px solid {theme.ORANGE};
            }}
            QTabBar::tab:hover {{ color: {theme.TEXT_PRIMARY}; }}
        """

    def refresh_styles(self):
        """Refresh existing controls, including dynamically added children."""
        return theme.refresh_styles(self)
