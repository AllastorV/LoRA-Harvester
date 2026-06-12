"""
Advanced Settings Panel for LoRA-Harvester v3.0
Contains Quality Analysis, Captioning, and Tag Settings UI components
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox,
    QTextEdit, QFrame, QPushButton, QDialog, QDialogButtonBox,
    QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal
from typing import Dict
from src.ui.translations import get_text
from src.ui import theme


# ══════════════════════════════════════════════════════════════
#  _AccordionFrame — self-contained collapsible panel
# ══════════════════════════════════════════════════════════════

class _AccordionFrame(QFrame):
    """Base class for collapsible accordion settings panel."""

    def __init__(self, icon: str, title: str, parent=None):
        super().__init__(parent)
        self._icon = icon
        self._title_text = title
        self._expanded = False
        self._setup_accordion()
        self._build_content()
        self._apply_accordion_style()

    def _setup_accordion(self):
        self.setStyleSheet(f"""
            QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER};
                      border-radius: 10px; }}
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toggle_btn = QPushButton(f"  {self._title_text}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setFixedHeight(46)
        self._toggle_btn.setCursor(Qt.PointingHandCursor)
        self._toggle_btn.toggled.connect(self._on_toggle)
        root.addWidget(self._toggle_btn)

        self._body = QWidget()
        self._body.setVisible(False)
        self._body.setStyleSheet(f"background: transparent; border-top: 1px solid {theme.BORDER};")
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(16, 12, 16, 14)
        self._body_lay.setSpacing(10)
        root.addWidget(self._body)

        self._anim = QPropertyAnimation(self._body, b"maximumHeight")
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)

    def _apply_accordion_style(self):
        self._toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {theme.TEXT_PRIMARY};
                border: none; font-size: {theme.fs(13)}; font-weight: 600;
                text-align: left; padding: 0 16px;
                border-radius: 10px;
            }}
            QPushButton:checked {{
                color: {theme.ORANGE};
                border-radius: 10px 10px 0 0;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.02); }}
        """)

    def _build_content(self):
        """Override in subclasses to populate self._body_lay."""
        pass

    def _on_toggle(self, checked: bool):
        if checked:
            self._body.setVisible(True)
            self._body.setMaximumHeight(0)
            self._anim.stop()
            self._anim.setStartValue(0)
            self._anim.setEndValue(self._body.sizeHint().height() + 60)
            self._anim.start()
        else:
            self._anim.stop()
            self._anim.setStartValue(self._body.maximumHeight())
            self._anim.setEndValue(0)
            self._anim.finished.connect(lambda: self._body.setVisible(False) if not self._toggle_btn.isChecked() else None)
            self._anim.start()

    def refresh_accordion_styles(self):
        self.setStyleSheet(f"""
            QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER};
                      border-radius: 10px; }}
        """)
        self._apply_accordion_style()


class QualitySettingsPanel(_AccordionFrame):
    """Quality Analysis Settings Panel"""

    settings_changed = pyqtSignal()

    def __init__(self, lang: str = 'en', parent=None):
        self.lang = lang
        super().__init__("", get_text('quality_title', lang), parent)

    def _build_content(self):
        """Build quality settings content into accordion body."""
        lay = self._body_lay

        # Enable checkbox
        enable_row = QHBoxLayout()
        self.enable_cb = QCheckBox(get_text('quality_enabled', self.lang))
        self.enable_cb.setStyleSheet(theme.checkbox_frame())
        self.enable_cb.setToolTip(get_text('quality_enabled_tooltip', self.lang))
        enable_row.addWidget(self.enable_cb); enable_row.addStretch()
        lay.addLayout(enable_row)

        # Blur threshold
        blur_row = QHBoxLayout()
        self.blur_label = QLabel(get_text('blur_threshold', self.lang))
        self.blur_label.setStyleSheet(theme.label_default())
        self.blur_spinbox = QDoubleSpinBox()
        self.blur_spinbox.setRange(10, 500)
        self.blur_spinbox.setValue(80.0)
        self.blur_spinbox.setStyleSheet(self._spinbox_style())
        self.blur_spinbox.setToolTip(get_text('blur_threshold_tooltip', self.lang))
        blur_row.addWidget(self.blur_label)
        blur_row.addStretch()
        blur_row.addWidget(self.blur_spinbox)
        lay.addLayout(blur_row)

        # Brightness range
        bright_row = QHBoxLayout()
        self.bright_label = QLabel(get_text('brightness_range', self.lang))
        self.bright_label.setStyleSheet(theme.label_default())
        self.bright_min = QSpinBox()
        self.bright_min.setRange(0, 255); self.bright_min.setValue(35)
        self.bright_min.setStyleSheet(self._spinbox_style())
        self.bright_min.setToolTip(get_text('brightness_tooltip', self.lang))
        self.bright_max = QSpinBox()
        self.bright_max.setRange(0, 255); self.bright_max.setValue(225)
        self.bright_max.setStyleSheet(self._spinbox_style())
        self.bright_max.setToolTip(get_text('brightness_tooltip', self.lang))
        bright_row.addWidget(self.bright_label)
        bright_row.addStretch()
        bright_row.addWidget(self.bright_min)
        bright_row.addWidget(QLabel("-"))
        bright_row.addWidget(self.bright_max)
        lay.addLayout(bright_row)

        # Skip duplicates
        self.skip_dup_cb = QCheckBox(get_text('skip_duplicates', self.lang))
        self.skip_dup_cb.setChecked(True)
        self.skip_dup_cb.setStyleSheet(theme.checkbox_frame())
        self.skip_dup_cb.setToolTip(get_text('skip_duplicates_tooltip', self.lang))
        lay.addWidget(self.skip_dup_cb)
    
    def _spinbox_style(self) -> str:
        return theme.spinbox_compact()

    def get_settings(self) -> Dict:
        return {
            'enabled': self.enable_cb.isChecked(),
            'blur_threshold': self.blur_spinbox.value(),
            'brightness_min': self.bright_min.value(),
            'brightness_max': self.bright_max.value(),
            'skip_duplicates': self.skip_dup_cb.isChecked()
        }

    def update_language(self, lang: str):
        self.lang = lang
        self._toggle_btn.setText(f"  {get_text('quality_title', lang)}")
        self.enable_cb.setText(get_text('quality_enabled', lang))
        self.enable_cb.setToolTip(get_text('quality_enabled_tooltip', lang))
        self.blur_label.setText(get_text('blur_threshold', lang))
        self.blur_spinbox.setToolTip(get_text('blur_threshold_tooltip', lang))
        self.bright_label.setText(get_text('brightness_range', lang))
        self.bright_min.setToolTip(get_text('brightness_tooltip', lang))
        self.bright_max.setToolTip(get_text('brightness_tooltip', lang))
        self.skip_dup_cb.setText(get_text('skip_duplicates', lang))
        self.skip_dup_cb.setToolTip(get_text('skip_duplicates_tooltip', lang))

    def refresh_styles(self):
        self.refresh_accordion_styles()


class CaptioningSettingsPanel(_AccordionFrame):
    """Captioning Settings Panel - WD14 Tagger"""

    settings_changed = pyqtSignal()

    def __init__(self, lang: str = 'en', parent=None):
        self.lang = lang
        super().__init__("", get_text('caption_title', lang), parent)

    def _build_content(self):
        lay = self._body_lay

        # Enable checkbox
        enable_row = QHBoxLayout()
        self.enable_cb = QCheckBox(get_text('caption_enabled', self.lang))
        self.enable_cb.setStyleSheet(theme.checkbox_frame())
        self.enable_cb.setToolTip(get_text('caption_enabled_tooltip', self.lang))
        enable_row.addWidget(self.enable_cb); enable_row.addStretch()
        lay.addLayout(enable_row)

        # Caption mode
        mode_row = QHBoxLayout()
        self.mode_label = QLabel(get_text('caption_mode_label', self.lang))
        self.mode_label.setStyleSheet(theme.label_default())
        self.mode_info = QLabel(""); self.mode_info.hide()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(get_text('caption_mode_tags', self.lang), 'tags_only')
        self.mode_combo.addItem(get_text('caption_mode_nlp', self.lang), 'florence2')
        self.mode_combo.addItem(get_text('caption_mode_combined', self.lang), 'combined')
        self.mode_combo.setStyleSheet(self._combo_style())
        self.mode_combo.setToolTip(get_text('caption_mode_tooltip', self.lang))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_label)
        mode_row.addWidget(self.mode_combo); mode_row.addStretch()
        lay.addLayout(mode_row)

        # Preset
        preset_row = QHBoxLayout()
        self.preset_label = QLabel(get_text('preset_label', self.lang))
        self.preset_label.setStyleSheet(theme.label_default())
        self.preset_info = QLabel(""); self.preset_info.hide()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem(get_text('preset_high_accuracy', self.lang), 'high_accuracy')
        self.preset_combo.addItem(get_text('preset_balanced', self.lang), 'balanced')
        self.preset_combo.addItem(get_text('preset_high_speed', self.lang), 'high_speed')
        self.preset_combo.addItem(get_text('preset_custom', self.lang), 'custom')
        self.preset_combo.setStyleSheet(self._combo_style())
        self.preset_combo.setToolTip(get_text('preset_tooltip', self.lang))
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_label)
        preset_row.addWidget(self.preset_combo); preset_row.addStretch()
        lay.addLayout(preset_row)

        # WD14
        self.wd14_row = QWidget()
        wd14_lay = QHBoxLayout(self.wd14_row); wd14_lay.setContentsMargins(0, 0, 0, 0)
        self.wd14_cb = QCheckBox(get_text('wd14_enabled', self.lang))
        self.wd14_cb.setChecked(True); self.wd14_cb.setStyleSheet(theme.label_default())
        self.wd14_cb.setToolTip(get_text('wd14_tooltip', self.lang))
        self.wd14_combo = QComboBox()
        self.wd14_combo.addItems([
            'SmilingWolf/wd-swinv2-tagger-v3',
            'SmilingWolf/wd-convnext-tagger-v3',
            'SmilingWolf/wd-vit-tagger-v3',
            'SmilingWolf/wd-v1-4-moat-tagger-v2',
            'SmilingWolf/wd-v1-4-swinv2-tagger-v2'
        ])
        self.wd14_combo.setStyleSheet(self._combo_style())
        self.wd14_combo.setToolTip(get_text('wd14_model_tooltip', self.lang))
        wd14_lay.addWidget(self.wd14_cb); wd14_lay.addWidget(self.wd14_combo); wd14_lay.addStretch()
        lay.addWidget(self.wd14_row)

        # Florence-2
        self.f2_row = QWidget()
        f2_lay = QHBoxLayout(self.f2_row); f2_lay.setContentsMargins(0, 0, 0, 0)
        self.f2_label = QLabel(get_text('florence2_model_label', self.lang))
        self.f2_label.setStyleSheet(theme.label_default())
        self.f2_label.setToolTip(get_text('florence2_model_label', self.lang))
        self.f2_combo = QComboBox()
        self.f2_combo.addItem(get_text('florence2_base', self.lang), 'florence-2-base')
        self.f2_combo.addItem(get_text('florence2_large', self.lang), 'florence-2-large')
        self.f2_combo.setStyleSheet(self._combo_style())
        self.f2_combo.setToolTip(get_text('florence2_model_label', self.lang))
        self.f2_task_combo = QComboBox()
        self.f2_task_combo.addItem(get_text('florence2_task_detailed', self.lang), '<DETAILED_CAPTION>')
        self.f2_task_combo.addItem(get_text('florence2_task_more', self.lang), '<MORE_DETAILED_CAPTION>')
        self.f2_task_combo.addItem(get_text('florence2_task_short', self.lang), '<CAPTION>')
        self.f2_task_combo.setStyleSheet(self._combo_style())
        self.f2_task_combo.setToolTip(get_text('caption_mode_label', self.lang))
        f2_lay.addWidget(self.f2_label); f2_lay.addWidget(self.f2_combo)
        f2_lay.addSpacing(10); f2_lay.addWidget(self.f2_task_combo); f2_lay.addStretch()
        self.f2_row.setVisible(False)
        lay.addWidget(self.f2_row)

        # Apply default preset
        self.preset_combo.setCurrentIndex(1)
        self._on_preset_changed(1)
    
    def _combo_style(self) -> str:
        return theme.combo_compact()

    # Preset → (wd14_model, min_confidence). Tag count is never touched.
    _PRESETS = {
        'high_accuracy': ('SmilingWolf/wd-swinv2-tagger-v3', 0.30),
        'balanced':      ('SmilingWolf/wd-convnext-tagger-v3', 0.35),
        'high_speed':    ('SmilingWolf/wd-vit-tagger-v3', 0.40),
    }

    def _on_mode_changed(self, index: int):
        mode = self.mode_combo.itemData(index)
        self.wd14_row.setVisible(mode in ('tags_only', 'combined'))
        self.f2_row.setVisible(mode in ('florence2', 'combined'))
        self.preset_label.setVisible(mode in ('tags_only', 'combined'))
        self.preset_info.setVisible(mode in ('tags_only', 'combined'))
        self.preset_combo.setVisible(mode in ('tags_only', 'combined'))
        self.settings_changed.emit()

    def _on_preset_changed(self, index: int):
        key = self.preset_combo.itemData(index)
        if key == 'custom' or key is None:
            return
        preset = self._PRESETS.get(key)
        if not preset:
            return
        model, _conf = preset
        idx = self.wd14_combo.findText(model)
        if idx >= 0:
            self.wd14_combo.setCurrentIndex(idx)
        self.settings_changed.emit()

    def get_settings(self) -> Dict:
        return {
            'enabled': self.enable_cb.isChecked(),
            'mode': self.mode_combo.currentData() or 'tags_only',
            'wd14_enabled': self.wd14_cb.isChecked(),
            'wd14_model': self.wd14_combo.currentText(),
            'florence2_model': self.f2_combo.currentData() or 'florence-2-base',
            'florence2_task': self.f2_task_combo.currentData() or '<DETAILED_CAPTION>',
            'min_confidence': self._PRESETS.get(
                self.preset_combo.currentData() or 'balanced',
                (None, 0.35))[1],
        }

    def update_language(self, lang: str):
        self.lang = lang
        self._toggle_btn.setText(f"  {get_text('caption_title', lang)}")
        self.enable_cb.setText(get_text('caption_enabled', lang))
        self.enable_cb.setToolTip(get_text('caption_enabled_tooltip', lang))
        self.mode_label.setText(get_text('caption_mode_label', lang))
        self.mode_combo.setToolTip(get_text('caption_mode_tooltip', lang))
        for idx, key in enumerate(['caption_mode_tags', 'caption_mode_nlp', 'caption_mode_combined']):
            self.mode_combo.blockSignals(True); self.mode_combo.setItemText(idx, get_text(key, lang)); self.mode_combo.blockSignals(False)
        self.preset_label.setText(get_text('preset_label', lang))
        self.preset_combo.setToolTip(get_text('preset_tooltip', lang))
        for idx, key in enumerate(['preset_high_accuracy', 'preset_balanced', 'preset_high_speed', 'preset_custom']):
            self.preset_combo.blockSignals(True); self.preset_combo.setItemText(idx, get_text(key, lang)); self.preset_combo.blockSignals(False)
        self.wd14_cb.setText(get_text('wd14_enabled', lang))
        self.wd14_cb.setToolTip(get_text('wd14_tooltip', lang))
        self.wd14_combo.setToolTip(get_text('wd14_model_tooltip', lang))
        self.f2_label.setText(get_text('florence2_model_label', lang))
        self.f2_label.setToolTip(get_text('florence2_model_label', lang))
        self.f2_combo.setToolTip(get_text('florence2_model_label', lang))
        self.f2_combo.blockSignals(True)
        self.f2_combo.setItemText(0, get_text('florence2_base', lang))
        self.f2_combo.setItemText(1, get_text('florence2_large', lang))
        self.f2_combo.blockSignals(False)
        self.f2_task_combo.setToolTip(get_text('caption_mode_label', lang))
        for idx, key in enumerate(['florence2_task_detailed', 'florence2_task_more', 'florence2_task_short']):
            self.f2_task_combo.blockSignals(True); self.f2_task_combo.setItemText(idx, get_text(key, lang)); self.f2_task_combo.blockSignals(False)

    def refresh_styles(self):
        self.refresh_accordion_styles()


class TagSettingsPanel(_AccordionFrame):
    """Tag Settings Panel - Trigger, Negative Tags, etc."""

    settings_changed = pyqtSignal()

    def __init__(self, lang: str = 'en', parent=None):
        self.lang = lang
        super().__init__("", get_text('tag_settings_title', lang), parent)

    def _build_content(self):
        layout = self._body_lay
        
        # Preset selector
        preset_layout = QHBoxLayout()
        self.preset_label = QLabel(get_text('tag_preset', self.lang))
        self.preset_label.setStyleSheet(theme.label_default())
        self.preset_info = QLabel(""); self.preset_info.hide()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem(get_text('tag_preset_none', self.lang), 'none')
        self.preset_combo.addItem(get_text('tag_preset_anime_character', self.lang), 'anime_character')
        self.preset_combo.addItem(get_text('tag_preset_style_lora', self.lang), 'style_lora')
        self.preset_combo.addItem(get_text('tag_preset_realistic_photo', self.lang), 'realistic_photo')
        self.preset_combo.addItem(get_text('tag_preset_concept_art', self.lang), 'concept_art')
        self.preset_combo.setStyleSheet(self._combo_style())
        self.preset_combo.setToolTip(get_text('tag_preset_tooltip', self.lang))
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        # Trigger word
        trigger_layout = QHBoxLayout()
        self.trigger_label = QLabel(get_text('trigger_word', self.lang))
        self.trigger_label.setStyleSheet(theme.label_default())
        self.trigger_info = QLabel(""); self.trigger_info.hide()
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText(get_text('trigger_word_ph', self.lang))
        self.trigger_edit.setStyleSheet(self._edit_style())
        self.trigger_edit.setToolTip(get_text('trigger_word_tooltip', self.lang))
        trigger_layout.addWidget(self.trigger_label)
        trigger_layout.addWidget(self.trigger_edit)
        layout.addLayout(trigger_layout)

        # Max tags and confidence
        limits_layout = QHBoxLayout()
        self.max_tags_label = QLabel(get_text('max_tags', self.lang))
        self.max_tags_label.setStyleSheet(theme.label_default())
        self.max_tags_info = QLabel(""); self.max_tags_info.hide()
        self.max_tags_spin = QSpinBox()
        self.max_tags_spin.setRange(5, 100)
        self.max_tags_spin.setValue(30)
        self.max_tags_spin.setStyleSheet(self._spinbox_style())
        self.max_tags_spin.setToolTip(get_text('max_tags_tooltip', self.lang))

        self.conf_label = QLabel(get_text('min_confidence', self.lang))
        self.conf_label.setStyleSheet(theme.label_default())
        self.conf_info = QLabel(""); self.conf_info.hide()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.9)
        self.conf_spin.setValue(0.35)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setStyleSheet(self._spinbox_style())
        self.conf_spin.setToolTip(get_text('min_confidence_tooltip', self.lang))

        limits_layout.addWidget(self.max_tags_label)
        limits_layout.addWidget(self.max_tags_spin)
        limits_layout.addSpacing(10)
        limits_layout.addWidget(self.conf_label)
        limits_layout.addWidget(self.conf_spin)
        limits_layout.addStretch()
        layout.addLayout(limits_layout)
        
        # Negative tags
        neg_layout = QVBoxLayout()
        neg_header = QHBoxLayout()
        self.neg_label = QLabel(get_text('negative_tags', self.lang))
        self.neg_label.setStyleSheet(theme.label_default())
        self.neg_help = QLabel(""); self.neg_help.hide()
        neg_header.addWidget(self.neg_label)
        neg_header.addStretch()
        neg_layout.addLayout(neg_header)
        
        self.neg_edit = QTextEdit()
        self.neg_edit.setMinimumHeight(45)
        self.neg_edit.setMaximumHeight(90)
        self.neg_edit.setPlaceholderText(get_text('negative_tags_ph', self.lang))
        self.neg_edit.setStyleSheet(theme.text_edit_input())
        self.neg_edit.setToolTip(get_text('negative_tags_tooltip', self.lang))
        # Set default negative tags
        self.neg_edit.setPlainText("watermark, signature, text, username, artist_name, twitter_username, patreon_username, dated")
        neg_layout.addWidget(self.neg_edit)
        layout.addLayout(neg_layout)
        
        # Priority tags
        priority_layout = QVBoxLayout()
        priority_header = QHBoxLayout()
        self.priority_label = QLabel(get_text('priority_tags', self.lang))
        self.priority_label.setStyleSheet(theme.label_default())
        self.priority_info = QLabel(""); self.priority_info.hide()
        priority_header.addWidget(self.priority_label)
        priority_header.addStretch()
        priority_layout.addLayout(priority_header)

        self.priority_edit = QLineEdit()
        self.priority_edit.setPlaceholderText(get_text('priority_tags_ph', self.lang))
        self.priority_edit.setStyleSheet(self._edit_style())
        self.priority_edit.setToolTip(get_text('priority_tags_tooltip', self.lang))
        priority_layout.addWidget(self.priority_edit)
        layout.addLayout(priority_layout)

        # Checkboxes row 1
        cb_layout1 = QHBoxLayout()
        self.keep_char_cb = QCheckBox(get_text('keep_character_tags', self.lang))
        self.keep_char_cb.setChecked(True)
        self.keep_char_cb.setStyleSheet(theme.label_default())
        self.keep_char_cb.setToolTip(get_text('keep_char_tooltip', self.lang))
        self.keep_series_cb = QCheckBox(get_text('keep_series_tags', self.lang))
        self.keep_series_cb.setStyleSheet(theme.label_default())
        self.keep_series_cb.setToolTip(get_text('keep_series_tooltip', self.lang))
        cb_layout1.addWidget(self.keep_char_cb)
        cb_layout1.addWidget(self.keep_series_cb)
        cb_layout1.addStretch()
        layout.addLayout(cb_layout1)

        # Checkboxes row 2
        cb_layout2 = QHBoxLayout()
        self.quality_tags_cb = QCheckBox(get_text('include_quality_tags', self.lang))
        self.quality_tags_cb.setStyleSheet(theme.label_default())
        self.quality_tags_cb.setToolTip(get_text('include_quality_tags', self.lang))
        self.rating_tags_cb = QCheckBox(get_text('include_rating_tags', self.lang))
        self.rating_tags_cb.setStyleSheet(theme.label_default())
        self.rating_tags_cb.setToolTip(get_text('include_rating_tags', self.lang))
        cb_layout2.addWidget(self.quality_tags_cb)
        cb_layout2.addWidget(self.rating_tags_cb)
        cb_layout2.addStretch()
        layout.addLayout(cb_layout2)

        # Formatting options
        format_layout = QHBoxLayout()
        self.underscore_cb = QCheckBox(get_text('use_underscores', self.lang))
        self.underscore_cb.setChecked(True)
        self.underscore_cb.setStyleSheet(theme.label_default())
        self.underscore_cb.setToolTip(get_text('use_underscores', self.lang))
        self.json_cb = QCheckBox(get_text('save_json', self.lang))
        self.json_cb.setStyleSheet(theme.label_default())
        self.json_cb.setToolTip(get_text('json_tooltip', self.lang))
        format_layout.addWidget(self.underscore_cb)
        format_layout.addWidget(self.json_cb)
        format_layout.addStretch()
        layout.addLayout(format_layout)
        
        # Prefix/Suffix
        prefix_layout = QHBoxLayout()
        self.prefix_label = QLabel(get_text('caption_prefix', self.lang))
        self.prefix_label.setStyleSheet(theme.label_default())
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setStyleSheet(self._edit_style())
        self.prefix_edit.setMaximumWidth(200)
        self.prefix_edit.setToolTip(get_text('caption_prefix', self.lang))

        self.suffix_label = QLabel(get_text('caption_suffix', self.lang))
        self.suffix_label.setStyleSheet(theme.label_default())
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setStyleSheet(self._edit_style())
        self.suffix_edit.setMaximumWidth(200)
        self.suffix_edit.setToolTip(get_text('caption_suffix', self.lang))
        
        prefix_layout.addWidget(self.prefix_label)
        prefix_layout.addWidget(self.prefix_edit)
        prefix_layout.addWidget(self.suffix_label)
        prefix_layout.addWidget(self.suffix_edit)
        prefix_layout.addStretch()
        layout.addLayout(prefix_layout)
    
    def _combo_style(self) -> str:
        return theme.combo_compact()
    
    def _edit_style(self) -> str:
        return theme.line_edit_compact()
    
    def _spinbox_style(self) -> str:
        return theme.spinbox_compact()
    
    def _on_preset_changed(self, index: int):
        """Load preset settings"""
        presets = {
            1: {  # anime_character
                'max_tags': 25,
                'min_confidence': 0.35,
                'keep_character': True,
                'keep_series': False,
                'include_quality': False,
                'negative_tags': 'watermark, signature, text, username, artist_name, twitter_username, simple_background, white_background'
            },
            2: {  # style_lora
                'max_tags': 20,
                'min_confidence': 0.4,
                'keep_character': False,
                'keep_series': False,
                'include_quality': True,
                'negative_tags': '1girl, 1boy, solo, character_name, watermark, signature'
            },
            3: {  # realistic_photo
                'max_tags': 15,
                'min_confidence': 0.5,
                'keep_character': False,
                'keep_series': False,
                'include_quality': False,
                'negative_tags': 'anime, manga, illustration, drawing, sketch, watermark, signature'
            },
            4: {  # concept_art
                'max_tags': 30,
                'min_confidence': 0.3,
                'keep_character': False,
                'keep_series': False,
                'include_quality': True,
                'negative_tags': 'watermark, signature, text, username'
            }
        }
        
        if index in presets:
            p = presets[index]
            self.max_tags_spin.setValue(p['max_tags'])
            self.conf_spin.setValue(p['min_confidence'])
            self.keep_char_cb.setChecked(p['keep_character'])
            self.keep_series_cb.setChecked(p['keep_series'])
            self.quality_tags_cb.setChecked(p['include_quality'])
            self.neg_edit.setPlainText(p['negative_tags'])
        
        self.settings_changed.emit()
    
    def get_settings(self) -> Dict:
        return {
            'preset': self.preset_combo.currentData() if self.preset_combo.currentIndex() > 0 else None,
            'trigger_word': self.trigger_edit.text().strip(),
            'max_tags': self.max_tags_spin.value(),
            'min_confidence': self.conf_spin.value(),
            'negative_tags': [t.strip() for t in self.neg_edit.toPlainText().split(',') if t.strip()],
            'priority_tags': [t.strip() for t in self.priority_edit.text().split(',') if t.strip()],
            'keep_character_tags': self.keep_char_cb.isChecked(),
            'keep_series_tags': self.keep_series_cb.isChecked(),
            'include_quality_tags': self.quality_tags_cb.isChecked(),
            'include_rating_tags': self.rating_tags_cb.isChecked(),
            'use_underscores': self.underscore_cb.isChecked(),
            'save_json': self.json_cb.isChecked(),
            'caption_prefix': self.prefix_edit.text().strip(),
            'caption_suffix': self.suffix_edit.text().strip()
        }
    
    def refresh_styles(self):
        self.refresh_accordion_styles()

    def update_language(self, lang: str):
        self.lang = lang
        self._toggle_btn.setText(f"  {get_text('tag_settings_title', lang)}")
        self.preset_label.setText(get_text('tag_preset', lang))
        self.preset_info.setToolTip(get_text('tag_preset_tooltip', lang))
        self.preset_combo.setToolTip(get_text('tag_preset_tooltip', lang))
        for idx, key in enumerate(['tag_preset_none', 'tag_preset_anime_character',
                                   'tag_preset_style_lora', 'tag_preset_realistic_photo',
                                   'tag_preset_concept_art']):
            self.preset_combo.blockSignals(True)
            self.preset_combo.setItemText(idx, get_text(key, lang))
            self.preset_combo.blockSignals(False)
        self.trigger_label.setText(get_text('trigger_word', lang))
        self.trigger_info.setToolTip(get_text('trigger_word_tooltip', lang))
        self.trigger_edit.setToolTip(get_text('trigger_word_tooltip', lang))
        self.trigger_edit.setPlaceholderText(get_text('trigger_word_ph', lang))
        self.max_tags_label.setText(get_text('max_tags', lang))
        self.max_tags_info.setToolTip(get_text('max_tags_tooltip', lang))
        self.max_tags_spin.setToolTip(get_text('max_tags_tooltip', lang))
        self.conf_label.setText(get_text('min_confidence', lang))
        self.conf_info.setToolTip(get_text('min_confidence_tooltip', lang))
        self.conf_spin.setToolTip(get_text('min_confidence_tooltip', lang))
        self.neg_label.setText(get_text('negative_tags', lang))
        self.neg_help.setToolTip(get_text('negative_tags_tooltip', lang))
        self.neg_edit.setToolTip(get_text('negative_tags_tooltip', lang))
        self.neg_edit.setPlaceholderText(get_text('negative_tags_ph', lang))
        self.priority_label.setText(get_text('priority_tags', lang))
        self.priority_info.setToolTip(get_text('priority_tags_tooltip', lang))
        self.priority_edit.setToolTip(get_text('priority_tags_tooltip', lang))
        self.priority_edit.setPlaceholderText(get_text('priority_tags_ph', lang))
        self.keep_char_cb.setText(get_text('keep_character_tags', lang))
        self.keep_char_cb.setToolTip(get_text('keep_char_tooltip', lang))
        self.keep_series_cb.setText(get_text('keep_series_tags', lang))
        self.keep_series_cb.setToolTip(get_text('keep_series_tooltip', lang))
        self.quality_tags_cb.setText(get_text('include_quality_tags', lang))
        self.quality_tags_cb.setToolTip(get_text('include_quality_tags', lang))
        self.rating_tags_cb.setText(get_text('include_rating_tags', lang))
        self.rating_tags_cb.setToolTip(get_text('include_rating_tags', lang))
        self.underscore_cb.setText(get_text('use_underscores', lang))
        self.underscore_cb.setToolTip(get_text('use_underscores', lang))
        self.json_cb.setText(get_text('save_json', lang))
        self.json_cb.setToolTip(get_text('json_tooltip', lang))
        self.prefix_label.setText(get_text('caption_prefix', lang))
        self.prefix_edit.setToolTip(get_text('caption_prefix', lang))
        self.suffix_label.setText(get_text('caption_suffix', lang))
        self.suffix_edit.setToolTip(get_text('caption_suffix', lang))


# ══════════════════════════════════════════════════════════════
#  UpscaleSettingsPanel — Real-ESRGAN upscale settings
# ══════════════════════════════════════════════════════════════

class _AddModelDialog(QDialog):
    """Small dialog for registering a custom .pth upscale model."""

    def __init__(self, parent=None, lang: str = 'en'):
        super().__init__(parent)
        self.lang = lang
        _t = lambda k: get_text(k, lang)
        self.setWindowTitle(_t('addmodel_title'))
        self.setMinimumWidth(420)
        self._pth_path = None

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # .pth file picker
        file_row = QHBoxLayout()
        self._file_lbl = QLabel(_t('addmodel_no_file'))
        self._file_lbl.setStyleSheet(f"color:{theme.TEXT_SECONDARY}; font-size:{theme.fs(11)};")
        file_btn = QPushButton(_t('addmodel_browse'))
        file_btn.setStyleSheet(theme.btn_secondary())
        file_btn.clicked.connect(self._browse)
        file_row.addWidget(self._file_lbl, 1)
        file_row.addWidget(file_btn)
        lay.addLayout(file_row)

        # Name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(_t('addmodel_name')))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText(_t('addmodel_name_ph'))
        name_row.addWidget(self._name_edit, 1)
        lay.addLayout(name_row)

        # Scale
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel(_t('addmodel_scale')))
        self._scale_spin = QSpinBox()
        self._scale_spin.setRange(1, 8)
        self._scale_spin.setValue(4)
        scale_row.addWidget(self._scale_spin)
        scale_row.addStretch()
        lay.addLayout(scale_row)

        # Arch
        arch_row = QHBoxLayout()
        arch_row.addWidget(QLabel(_t('addmodel_arch')))
        self._arch_combo = QComboBox()
        self._arch_combo.addItems(["RRDBNet", "SRVGGNetCompact"])
        arch_row.addWidget(self._arch_combo)
        arch_row.addStretch()
        lay.addLayout(arch_row)

        # num_block
        block_row = QHBoxLayout()
        block_row.addWidget(QLabel(_t('addmodel_blocks')))
        self._block_spin = QSpinBox()
        self._block_spin.setRange(1, 64)
        self._block_spin.setValue(23)
        block_row.addWidget(self._block_spin)
        block_row.addStretch()
        lay.addLayout(block_row)

        # Description
        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel(_t('addmodel_desc')))
        self._desc_edit = QLineEdit()
        self._desc_edit.setPlaceholderText(_t('addmodel_desc_ph'))
        desc_row.addWidget(self._desc_edit, 1)
        lay.addLayout(desc_row)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, get_text('addmodel_browse_title', self.lang), "",
            get_text('addmodel_browse_filter', self.lang)
        )
        if path:
            from pathlib import Path
            self._pth_path = Path(path)
            self._file_lbl.setText(self._pth_path.name)
            if not self._name_edit.text():
                self._name_edit.setText(self._pth_path.stem)

    def get_values(self):
        return {
            'pth_path': self._pth_path,
            'name': self._name_edit.text().strip(),
            'scale': self._scale_spin.value(),
            'arch': self._arch_combo.currentText(),
            'num_block': self._block_spin.value(),
            'description': self._desc_edit.text().strip(),
        }


# Shared max-resolution cap presets (longest side, px). Used by both the
# main-screen upscale panel and the standalone Upscale tab.
MAX_RES_PRESETS = [1024, 1080, 2048, 3840, 4096]


class UpscaleSettingsPanel(_AccordionFrame):
    """Real-ESRGAN upscale settings accordion panel."""

    settings_changed = pyqtSignal()

    def __init__(self, lang: str = 'en', parent=None):
        self.lang = lang
        super().__init__("", get_text('upscale_title', lang), parent)

    def _build_content(self):
        lay = self._body_lay
        _t = lambda k: get_text(k, self.lang)

        # Enable
        enable_row = QHBoxLayout()
        self.enable_cb = QCheckBox(_t('upscale_enable'))
        self.enable_cb.setStyleSheet(theme.checkbox_frame())
        self.enable_cb.setToolTip(_t('upscale_enable_tip'))
        self.enable_cb.toggled.connect(self._on_enable_toggle)
        enable_row.addWidget(self.enable_cb)
        enable_row.addStretch()
        lay.addLayout(enable_row)

        # Model row
        model_row = QHBoxLayout()
        self._model_lbl = QLabel(_t('upscale_model'))
        self._model_lbl.setStyleSheet(theme.label_default())
        self._model_lbl.setFixedWidth(100)
        self._model_combo = QComboBox()
        self._model_combo.setStyleSheet(theme.spinbox_compact())
        self._model_combo.setToolTip(_t('upscale_model_tip'))
        self._populate_models()

        self._refresh_btn = QPushButton("🔄")
        self._refresh_btn.setFixedSize(28, 28)
        self._refresh_btn.setToolTip(_t('upscale_refresh_tip'))
        self._refresh_btn.setStyleSheet(theme.btn_secondary())
        self._refresh_btn.clicked.connect(self._refresh_models)

        self._add_btn = QPushButton(_t('upscale_add_model'))
        self._add_btn.setStyleSheet(theme.btn_secondary())
        self._add_btn.setToolTip(_t('upscale_add_model_tip'))
        self._add_btn.clicked.connect(self._add_model)

        model_row.addWidget(self._model_lbl)
        model_row.addWidget(self._model_combo, 1)
        model_row.addWidget(self._refresh_btn)
        model_row.addWidget(self._add_btn)
        lay.addLayout(model_row)

        # Target
        target_row = QHBoxLayout()
        self._target_lbl = QLabel(_t('upscale_target'))
        self._target_lbl.setStyleSheet(theme.label_default())
        self._target_lbl.setFixedWidth(100)
        self._target_combo = QComboBox()
        self._target_combo.setStyleSheet(theme.spinbox_compact())
        self._target_combo.addItem(_t('upscale_target_crop'), "crop")
        self._target_combo.addItem(_t('upscale_target_frame'), "frame")
        self._target_combo.setToolTip(_t('upscale_target_tip'))
        target_row.addWidget(self._target_lbl)
        target_row.addWidget(self._target_combo)
        target_row.addStretch()
        lay.addLayout(target_row)

        # Min resolution + tile
        params_row = QHBoxLayout()
        self._min_res_lbl = QLabel(_t('upscale_min_res'))
        self._min_res_lbl.setStyleSheet(theme.label_default())
        self._min_res_spin = QSpinBox()
        self._min_res_spin.setRange(0, 4096)
        self._min_res_spin.setValue(512)
        self._min_res_spin.setSuffix(" px")
        self._min_res_spin.setStyleSheet(theme.spinbox_compact())
        self._min_res_spin.setToolTip(_t('upscale_min_res_tip'))

        self._tile_lbl = QLabel(_t('upscale_tile'))
        self._tile_lbl.setStyleSheet(theme.label_default())
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(0, 2048)
        self._tile_spin.setValue(0)
        self._tile_spin.setSuffix(" px")
        self._tile_spin.setStyleSheet(theme.spinbox_compact())
        self._tile_spin.setToolTip(_t('upscale_tile_tip'))

        params_row.addWidget(self._min_res_lbl)
        params_row.addWidget(self._min_res_spin)
        params_row.addSpacing(12)
        params_row.addWidget(self._tile_lbl)
        params_row.addWidget(self._tile_spin)
        params_row.addStretch()
        lay.addLayout(params_row)

        # Max resolution cap (presets) — downscale if upscale overshoots
        maxres_row = QHBoxLayout()
        self._max_res_lbl = QLabel(_t('upscale_max_res'))
        self._max_res_lbl.setStyleSheet(theme.label_default())
        self._max_res_combo = QComboBox()
        self._max_res_combo.setStyleSheet(theme.spinbox_compact())
        self._max_res_combo.setToolTip(_t('upscale_max_res_tip'))
        self._max_res_combo.addItem(_t('upscale_max_res_off'), 0)
        for _p in MAX_RES_PRESETS:
            self._max_res_combo.addItem(f"{_p} px", _p)
        maxres_row.addWidget(self._max_res_lbl)
        maxres_row.addWidget(self._max_res_combo)
        maxres_row.addStretch()
        lay.addLayout(maxres_row)

        # Face enhance
        self._face_cb = QCheckBox(_t('upscale_face_enhance'))
        self._face_cb.setStyleSheet(theme.checkbox_frame())
        self._face_cb.setToolTip(_t('upscale_face_tip'))
        lay.addWidget(self._face_cb)

        self._on_enable_toggle(False)

    def _on_enable_toggle(self, enabled: bool):
        """Grey-out sub-widgets when upscale is disabled."""
        for w in [self._model_combo, self._target_combo,
                  self._min_res_spin, self._tile_spin,
                  self._max_res_combo, self._face_cb]:
            w.setEnabled(enabled)

    def _populate_models(self):
        """Fill model combo from registry."""
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        first_available = -1
        try:
            from src.core.upscale_models import list_models
            models = list_models()
            for idx, (name, cfg) in enumerate(models.items()):
                scale = cfg.get('scale', '?')
                available = cfg.get('available', False)
                label = f"{name}  [{scale}×]" + ("" if available else "  ⬇ not downloaded")
                self._model_combo.addItem(label, name)
                if available and first_available < 0:
                    first_available = idx
        except Exception as e:
            self._model_combo.addItem(f"(error: {e})", "RealESRGAN_x4plus_anime_6B")
        # Default to the first DOWNLOADED model so enabling upscale out of the
        # box doesn't pick a model whose weights are missing (which surfaced as
        # a confusing "deps missing" / upscale-disabled message).
        if first_available >= 0:
            self._model_combo.setCurrentIndex(first_available)
        self._model_combo.blockSignals(False)

    def _refresh_models(self):
        current = self._model_combo.currentData()
        self._populate_models()
        # Restore previous selection if still present
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == current:
                self._model_combo.setCurrentIndex(i)
                break

    def _add_model(self):
        dlg = _AddModelDialog(self.window(), lang=self.lang)
        if dlg.exec_() != QDialog.Accepted:
            return
        vals = dlg.get_values()
        if not vals['pth_path'] or not vals['name']:
            QMessageBox.warning(self, get_text('addmodel_missing_title', self.lang),
                                get_text('addmodel_missing_body', self.lang))
            return
        try:
            from src.core.upscale_models import add_custom_model
            add_custom_model(
                pth_path=vals['pth_path'],
                name=vals['name'],
                scale=vals['scale'],
                arch=vals['arch'],
                num_block=vals['num_block'],
                description=vals['description'],
            )
            self._refresh_models()
            # Select the newly added model
            for i in range(self._model_combo.count()):
                if self._model_combo.itemData(i) == vals['name']:
                    self._model_combo.setCurrentIndex(i)
                    break
            QMessageBox.information(self, get_text('addmodel_added_title', self.lang),
                                    get_text('addmodel_added_body', self.lang).format(vals['name']))
        except Exception as e:
            QMessageBox.critical(self, get_text('addmodel_error_title', self.lang), str(e))

    def get_settings(self) -> dict:
        return {
            'enabled': self.enable_cb.isChecked(),
            'model': self._model_combo.currentData() or "RealESRGAN_x4plus_anime_6B",
            'target': self._target_combo.currentData() or "crop",
            'min_resolution': self._min_res_spin.value(),
            'tile': self._tile_spin.value(),
            'max_resolution': self._max_res_combo.currentData() or 0,
            'face_enhance': self._face_cb.isChecked(),
            'use_gpu': True,
        }

    def set_settings(self, cfg: dict):
        self.enable_cb.setChecked(cfg.get('enabled', False))
        model = cfg.get('model', 'RealESRGAN_x4plus_anime_6B')
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == model:
                self._model_combo.setCurrentIndex(i)
                break
        target = cfg.get('target', 'crop')
        for i in range(self._target_combo.count()):
            if self._target_combo.itemData(i) == target:
                self._target_combo.setCurrentIndex(i)
                break
        self._min_res_spin.setValue(cfg.get('min_resolution', 512))
        self._tile_spin.setValue(cfg.get('tile', 0))
        max_res = int(cfg.get('max_resolution', 0) or 0)
        _mi = self._max_res_combo.findData(max_res)
        self._max_res_combo.setCurrentIndex(_mi if _mi >= 0 else 0)
        self._face_cb.setChecked(cfg.get('face_enhance', False))
        self._on_enable_toggle(cfg.get('enabled', False))

    def update_language(self, lang: str):
        self.lang = lang
        _t = lambda k: get_text(k, lang)
        self._toggle_btn.setText(f"  {_t('upscale_title')}")
        # Enable checkbox
        self.enable_cb.setText(_t('upscale_enable'))
        self.enable_cb.setToolTip(_t('upscale_enable_tip'))
        # Model row
        self._model_lbl.setText(_t('upscale_model'))
        self._model_combo.setToolTip(_t('upscale_model_tip'))
        self._refresh_btn.setToolTip(_t('upscale_refresh_tip'))
        self._add_btn.setText(_t('upscale_add_model'))
        self._add_btn.setToolTip(_t('upscale_add_model_tip'))
        # Target row
        self._target_lbl.setText(_t('upscale_target'))
        self._target_combo.setToolTip(_t('upscale_target_tip'))
        self._target_combo.blockSignals(True)
        self._target_combo.setItemText(0, _t('upscale_target_crop'))
        self._target_combo.setItemText(1, _t('upscale_target_frame'))
        self._target_combo.blockSignals(False)
        # Min resolution + tile
        self._min_res_lbl.setText(_t('upscale_min_res'))
        self._min_res_spin.setToolTip(_t('upscale_min_res_tip'))
        self._tile_lbl.setText(_t('upscale_tile'))
        self._tile_spin.setToolTip(_t('upscale_tile_tip'))
        # Max resolution cap
        self._max_res_lbl.setText(_t('upscale_max_res'))
        self._max_res_combo.setToolTip(_t('upscale_max_res_tip'))
        self._max_res_combo.blockSignals(True)
        self._max_res_combo.setItemText(0, _t('upscale_max_res_off'))
        self._max_res_combo.blockSignals(False)
        # Face enhance
        self._face_cb.setText(_t('upscale_face_enhance'))
        self._face_cb.setToolTip(_t('upscale_face_tip'))

    def refresh_styles(self):
        self.refresh_accordion_styles()


# ══════════════════════════════════════════════════════════════
#  NsfwSettingsPanel — SFW/NSFW detection settings
