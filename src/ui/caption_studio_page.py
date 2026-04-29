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
        # Start loading tags immediately — don't wait for focus
        self._ensure_tags_loaded()

    def _setup_completer(self):
        global _shared_tag_model
        if _shared_tag_model is None:
            _shared_tag_model = QStringListModel([])

        c = QCompleter(_shared_tag_model, self)
        # For QTextEdit the popup must be parented to the viewport
        c.setWidget(self)
        c.setCompletionMode(QCompleter.PopupCompletion)
        c.setCaseSensitivity(Qt.CaseInsensitive)
        c.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
        c.setMaxVisibleItems(10)
        c.activated[str].connect(self._insert_completion)

        # Style the popup to match the dark theme
        popup = c.popup()
        popup.setStyleSheet(
            f"QListView {{"
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
            f"}}"
        )
        self._completer = c

    # ── Tag loading ─────────────────────────────────────────────

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
    finished = pyqtSignal(dict)
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
        # Model usage driven by independent checkboxes
        use_wd14 = self.settings.get('use_wd14', True)
        use_f2 = self.settings.get('use_florence2', False)

        try:
            # ── Load WD14 if needed ────────────────────────────
            if use_wd14 and self.captioner:
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

            # ── Load Florence-2 if needed ──────────────────────
            if use_f2 and self.florence2:
                try:
                    self.log_message.emit("Loading Florence-2 model...")
                    self.florence2._load_model()
                    self.log_message.emit("✅ Florence-2 model loaded")
                except Exception as e:
                    self.log_message.emit(f"❌ Florence-2 FAILED: {e}")
                    if mode == 'florence2':
                        self.error.emit(f"Florence-2 load failed: {e}")
                        return
                    self.florence2 = None

            if not self._running:
                return

            # Route: use fast AdvancedCaptioner path only when WD14-only + no Florence-2
            if use_wd14 and not use_f2 and mode == 'tags_only':
                stats = self.captioner.caption_directory(
                    self.image_folder,
                    mode='tags_only',
                    overwrite=self.settings.get('overwrite', False),
                    save_json=self.settings.get('save_json', False),
                    progress_callback=self._progress_callback,
                    recursive=self.settings.get('recursive', False),
                )
            else:
                stats = self._run_with_florence2()

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
            try:
                if self.florence2 and hasattr(self.florence2, 'cleanup'):
                    self.florence2.cleanup()
            except Exception:
                pass

    def _run_with_florence2(self) -> Dict:
        """Process images with Florence-2 (or combined WD14+Florence-2)."""
        import cv2
        from pathlib import Path as _Path

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
        glob_fn = dir_path.rglob if recursive else dir_path.glob
        for ext in extensions:
            all_images.extend(glob_fn(ext))
            all_images.extend(glob_fn(ext.upper()))

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
                image = cv2.imread(str(img_path))
                if image is None:
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

    def _init_ui(self):
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 12, 20, 12)

        # ── Step 1: Folder ──────────────────────────────────────
        step1_frame = QFrame()
        step1_frame.setStyleSheet(theme.card_frame())
        step1_layout = QVBoxLayout(step1_frame)
        step1_layout.setContentsMargins(2, 8, 2, 8)

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
        self.folder_label.setMinimumWidth(0)
        # Allow label to shrink below its preferred width — prevents horizontal scroll
        from PyQt5.QtWidgets import QSizePolicy as _SP
        self.folder_label.setSizePolicy(_SP.Ignored, _SP.Preferred)
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

        # ── Output Mode Selection Card ───────────────────────────
        # ── Step 2: Clean settings card ─────────────────────────
        step2_frame = QFrame()
        step2_frame.setStyleSheet(theme.card_frame())
        step2_layout = QVBoxLayout(step2_frame)
        step2_layout.setSpacing(12)
        step2_layout.setContentsMargins(16, 12, 16, 12)

        _LABEL_W = 100  # fixed label column width for alignment

        def _make_row(label_text, widget, stretch_widget=True):
            row = QHBoxLayout()
            row.setSpacing(12)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(_LABEL_W)
            lbl.setStyleSheet(
                f"background: transparent; border: none; "
                f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(11)};"
            )
            row.addWidget(lbl)
            if stretch_widget:
                row.addWidget(widget, stretch=1)
            else:
                row.addWidget(widget)
                row.addStretch()
            return row, lbl

        # ── Row 1: Model ────────────────────────────────────────
        self.model_combo = QComboBox()
        self.model_combo.addItem("WD14 Tagger",      'wd14')
        self.model_combo.addItem("Florence-2",        'florence2')
        self.model_combo.addItem("WD14 + Florence-2", 'both')
        self.model_combo.setStyleSheet(theme.combo())
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        row_model, self._model_lbl = _make_row("Model", self.model_combo)
        step2_layout.addLayout(row_model)

        # ── Row 1b: Variant ─────────────────────────────────────
        self.variant_combo = QComboBox()
        self.variant_combo.setStyleSheet(theme.combo())
        self.variant_combo.currentIndexChanged.connect(self._on_variant_changed)
        row_variant, self._variant_lbl = _make_row("Variant", self.variant_combo)
        step2_layout.addLayout(row_variant)
        # _populate_variants deferred — called after conf_spin/max_tags_spin are created

        # ── Row 2: Mode ─────────────────────────────────────────
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(get_text('caption_mode_tags', self.lang),      'tags_only')
        self.mode_combo.addItem(get_text('caption_mode_tag_first', self.lang), 'tag_first')
        self.mode_combo.addItem(get_text('caption_mode_combined', self.lang),  'combined')
        self.mode_combo.setStyleSheet(theme.combo())
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row_mode, self._mode_lbl = _make_row("Mode", self.mode_combo)
        step2_layout.addLayout(row_mode)

        # ── Row 3: Trigger Word ─────────────────────────────────
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("sks person...")
        self.trigger_edit.setStyleSheet(theme.line_edit())
        row_trigger, self._trigger_lbl = _make_row("Trigger Word", self.trigger_edit)
        step2_layout.addLayout(row_trigger)

        # ── Row 4: Last Words ───────────────────────────────────
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("masterpiece, best quality...")
        self.suffix_edit.setStyleSheet(theme.line_edit())
        row_suffix, self._suffix_lbl = _make_row("Last Words", self.suffix_edit)
        step2_layout.addLayout(row_suffix)

        # ── Divider ─────────────────────────────────────────────
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"color: {theme.BORDER}; background: {theme.BORDER};")
        div.setFixedHeight(1)
        step2_layout.addWidget(div)

        # ── Row 5: Max Tags (kelime sayısı) — sol hizalı ────────
        self.max_tags_spin = QSpinBox()
        self.max_tags_spin.setRange(5, 100)
        self.max_tags_spin.setValue(25)
        self.max_tags_spin.setStyleSheet(theme.spinbox())
        self.max_tags_spin.setFixedWidth(120)
        self.max_tags_spin.setKeyboardTracking(True)
        self.max_tags_spin.setFocusPolicy(Qt.StrongFocus)
        self.max_tags_spin.lineEdit().setReadOnly(False)
        row_maxtags, self._maxtags_lbl = _make_row("Max Tags", self.max_tags_spin, stretch_widget=False)
        step2_layout.addLayout(row_maxtags)

        # ── Row 6: Confidence (ağırlık) — sol hizalı ────────────
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.10, 0.90)
        self.conf_spin.setValue(0.35)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setStyleSheet(theme.spinbox())
        self.conf_spin.setFixedWidth(120)
        self.conf_spin.setKeyboardTracking(True)
        self.conf_spin.setFocusPolicy(Qt.StrongFocus)
        self.conf_spin.lineEdit().setReadOnly(False)
        row_conf, self._conf_lbl = _make_row("Confidence", self.conf_spin, stretch_widget=False)
        step2_layout.addLayout(row_conf)

        # Populate variant dropdown now that spinboxes exist
        self._populate_variants('wd14')

        layout.addWidget(step2_frame)

        # ── Hidden advanced widgets (used by get_settings / backend) ──
        self._adv = QFrame()   # keep reference — prevents GC of child widgets
        self._adv.hide()
        _adv_lay = QVBoxLayout(self._adv)
        _adv = self._adv

        self.mode_label  = QLabel()  # compat placeholder
        self.mode_info   = QLabel()
        self.f2_label    = QLabel(get_text('florence2_model_label', self.lang))
        self.f2_combo    = QComboBox()
        self.f2_combo.addItem('Florence-2 Base (Fast)', 'florence-2-base')
        self.f2_combo.addItem('Florence-2 Large (Accurate)', 'florence-2-large')
        self.f2_task_combo = QComboBox()
        self.f2_task_combo.addItem(get_text('florence2_task_detailed', self.lang), '<DETAILED_CAPTION>')
        self.f2_task_combo.addItem(get_text('florence2_task_more', self.lang), '<MORE_DETAILED_CAPTION>')
        self.f2_task_combo.addItem(get_text('florence2_task_short', self.lang), '<CAPTION>')
        self._f2_settings_widget = QFrame()
        _f2l = QHBoxLayout(self._f2_settings_widget)
        _f2l.setContentsMargins(0, 0, 0, 0)
        _f2l.addWidget(self.f2_label)
        _f2l.addWidget(self.f2_combo)
        _f2l.addWidget(self.f2_task_combo)
        _adv_lay.addWidget(self._f2_settings_widget)

        self.preset_label = QLabel()
        self.preset_info  = QLabel()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem('', 'high_accuracy')
        self.preset_combo.addItem('', 'balanced')
        self.preset_combo.addItem('', 'high_speed')
        self.preset_combo.addItem('', 'custom')
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self.max_label    = QLabel()
        self.max_info     = QLabel()
        self.conf_label   = QLabel()
        self.conf_info    = QLabel()
        self.neg_label    = QLabel()
        self.neg_info     = QLabel()
        self.neg_edit     = QLineEdit()
        self.neg_edit.setText(
            "watermark, signature, text, username, artist_name, "
            "twitter_username, patreon_username, dated"
        )

        self.wd14_cb      = QCheckBox()
        self.wd14_cb.setChecked(True)
        self.wd14_info    = QLabel()
        self.wd14_combo   = QComboBox()
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
        _mrl.addWidget(self.wd14_cb)
        _mrl.addWidget(self.wd14_combo)
        _adv_lay.addWidget(self._model_row_widget)

        self.keep_char_cb = QCheckBox()
        self.keep_char_cb.setChecked(True)
        self.json_cb      = QCheckBox()
        self.trigger_label = QLabel()
        self.trigger_info  = QLabel()
        self.suffix_label  = QLabel()
        self.suffix_info   = QLabel()

        # _adv is NOT added to layout — keeps it as an attribute store
        # without contributing to the minimum width of _GenerateTab

        # ── Step 3: Start / Stop / Progress ─────────────────────
        step3_frame = QFrame()
        step3_frame.setStyleSheet(theme.card_frame())
        step3_layout = QVBoxLayout(step3_frame)

        self.step3_title = QLabel(get_text('step3_start', self.lang))
        self.step3_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step3_title.setStyleSheet(theme.label_section())
        step3_layout.addWidget(self.step3_title)

        self._install_thread = None

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
        model_key = self.model_combo.currentData() or 'wd14'
        use_wd14      = model_key in ('wd14', 'both')
        use_florence2 = model_key in ('florence2', 'both')
        return {
            'mode':          self.mode_combo.currentData() or 'tags_only',
            'trigger_word':  self.trigger_edit.text().strip(),
            'caption_suffix': self.suffix_edit.text().strip(),
            'max_tags':      self.max_tags_spin.value(),
            'min_confidence': self.conf_spin.value(),
            'negative_tags': [t.strip() for t in self.neg_edit.text().split(',') if t.strip()],
            'keep_character_tags': self.keep_char_cb.isChecked(),
            'save_json':     self.json_cb.isChecked(),
            'overwrite':     self.overwrite_cb.isChecked(),
            'recursive':     self.recursive_cb.isChecked(),
            'use_wd14':      use_wd14,
            'use_florence2': use_florence2,
            'wd14_model':    self.wd14_combo.currentText(),
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
        self.captioning_thread.finished.connect(self._on_finished)
        self.captioning_thread.error.connect(self._on_error)
        self.captioning_thread.finished.connect(self._safe_delete_thread)
        self.captioning_thread.error.connect(self._safe_delete_thread)
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

        # Visible labels in the clean settings card
        _s(getattr(self, '_model_lbl',   None), "Model")
        _s(getattr(self, '_mode_lbl',    None), "Mode")
        _s(getattr(self, '_trigger_lbl', None), "Trigger Word")
        _s(getattr(self, '_suffix_lbl',  None), "Last Words")
        _s(getattr(self, '_maxtags_lbl', None), "Max Tags")
        _s(getattr(self, '_conf_lbl',    None), "Confidence")

        # Mode combo items
        mc = getattr(self, 'mode_combo', None)
        if mc:
            idx = mc.currentIndex()
            mc.blockSignals(True)
            for i, key in enumerate(['caption_mode_tags', 'caption_mode_tag_first',
                                      'caption_mode_combined']):
                if i < mc.count():
                    mc.setItemText(i, get_text(key, lang))
            mc.setCurrentIndex(idx)
            mc.blockSignals(False)

        # Buttons & misc
        _s(getattr(self, 'browse_btn',  None), get_text('select_input_folder', lang))
        _s(getattr(self, 'start_btn',   None), get_text('start_captioning', lang))
        _s(getattr(self, 'stop_btn',    None), get_text('stop_btn', lang))
        if not self.selected_folder:
            _s(getattr(self, 'folder_label', None), get_text('drag_drop_folder', lang))

        # Checkboxes
        _s(getattr(self, 'recursive_cb',   None), get_text('recursive_search', lang))
        _s(getattr(self, 'overwrite_cb',   None), get_text('overwrite_existing', lang))

        # Tooltips
        _tt(getattr(self, 'recursive_cb',  None), get_text('recursive_tooltip', lang))
        _tt(getattr(self, 'overwrite_cb',  None), get_text('overwrite_tooltip', lang))

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
        # Florence-2 / mode widgets
        self.mode_label.setStyleSheet(theme.label_frame())
        self.mode_info.setStyleSheet(theme.info_icon_frame())
        self.mode_combo.setStyleSheet(theme.combo())
        self.f2_label.setStyleSheet(theme.label_frame())
        self.f2_combo.setStyleSheet(theme.combo())
        self.f2_task_combo.setStyleSheet(theme.combo())
        self._f2_settings_widget.setStyleSheet("background-color: transparent;")
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
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        self.setLayout(root)

        # Tab widget — objectName for tests
        self.tabs = QTabWidget()
        self.tabs.setObjectName("caption_tabs")
        self.generate_tab = _GenerateTab(self.lang, self)
        self.edit_tab = _EditTab(self.lang, self)

        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setWidget(self.generate_tab)
        gen_scroll.setFrameShape(QFrame.NoFrame)
        gen_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.tabs.addTab(gen_scroll, "Generate")
        self.tabs.addTab(self.edit_tab, "Edit")
        self.tabs.setStyleSheet(self._tab_style())
        root.addWidget(self.tabs, stretch=1)

        self.generate_tab.folder_changed.connect(self.edit_tab.reload_folder)
        self.generate_tab.captioning_finished.connect(self._on_generation_done)

    def _on_generation_done(self, folder: str):
        if folder:
            self.edit_tab.reload_folder(folder)

    def update_language(self, lang: str):
        self.lang = lang
        # tabs keep fixed English labels to match objectName test expectations
        self.generate_tab.update_language(lang)
        self.edit_tab.update_language(lang)

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
                padding: 12px 20px;
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
        self.tabs.setStyleSheet(self._tab_style())
        self.generate_tab.refresh_styles()
        self.edit_tab.refresh_styles()
