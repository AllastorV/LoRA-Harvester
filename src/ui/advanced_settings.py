"""
Advanced Settings Panel for LoRA-Harvester v2.0
Contains Quality Analysis, Captioning, and Tag Settings UI components
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox,
    QTextEdit, QPushButton, QSlider, QTabWidget, QScrollArea,
    QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import Dict
from src.ui.translations import get_text
from src.ui import theme


class QualitySettingsPanel(QGroupBox):
    """Quality Analysis Settings Panel"""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(get_text('quality_title', lang), parent)
        self.lang = lang
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setStyleSheet(theme.panel_group())
        
        layout = QVBoxLayout()
        
        # Enable checkbox
        enable_layout = QHBoxLayout()
        self.enable_cb = QCheckBox(get_text('quality_enabled', self.lang))
        self.enable_cb.setStyleSheet(theme.label_accent())
        self.enable_cb.toggled.connect(self._on_enable_toggled)
        enable_layout.addWidget(self.enable_cb)
        enable_layout.addStretch()
        layout.addLayout(enable_layout)
        
        # Settings container
        self.settings_widget = QWidget()
        self.settings_widget.setVisible(False)
        settings_layout = QVBoxLayout(self.settings_widget)
        
        # Blur threshold
        blur_layout = QHBoxLayout()
        self.blur_label = QLabel(get_text('blur_threshold', self.lang))
        self.blur_label.setStyleSheet(theme.label_default())
        self.blur_spinbox = QDoubleSpinBox()
        self.blur_spinbox.setRange(10, 500)
        self.blur_spinbox.setValue(80.0)
        self.blur_spinbox.setStyleSheet(self._spinbox_style())
        blur_layout.addWidget(self.blur_label)
        blur_layout.addWidget(self.blur_spinbox)
        blur_layout.addStretch()
        settings_layout.addLayout(blur_layout)
        
        # Brightness range
        bright_layout = QHBoxLayout()
        self.bright_label = QLabel(get_text('brightness_range', self.lang))
        self.bright_label.setStyleSheet(theme.label_default())
        self.bright_min = QSpinBox()
        self.bright_min.setRange(0, 255)
        self.bright_min.setValue(35)
        self.bright_min.setStyleSheet(self._spinbox_style())
        self.bright_max = QSpinBox()
        self.bright_max.setRange(0, 255)
        self.bright_max.setValue(225)
        self.bright_max.setStyleSheet(self._spinbox_style())
        bright_layout.addWidget(self.bright_label)
        bright_layout.addWidget(self.bright_min)
        bright_layout.addWidget(QLabel("-"))
        bright_layout.addWidget(self.bright_max)
        bright_layout.addStretch()
        settings_layout.addLayout(bright_layout)
        
        # Skip duplicates
        dup_layout = QHBoxLayout()
        self.skip_dup_cb = QCheckBox(get_text('skip_duplicates', self.lang))
        self.skip_dup_cb.setChecked(True)
        self.skip_dup_cb.setStyleSheet(theme.label_default())
        dup_layout.addWidget(self.skip_dup_cb)
        dup_layout.addStretch()
        settings_layout.addLayout(dup_layout)
        
        layout.addWidget(self.settings_widget)
        self.setLayout(layout)
    
    def _spinbox_style(self) -> str:
        return theme.spinbox_compact()
    
    def _on_enable_toggled(self, checked: bool):
        self.settings_widget.setVisible(checked)
        self.settings_changed.emit()
    
    def get_settings(self) -> Dict:
        return {
            'enabled': self.enable_cb.isChecked(),
            'blur_threshold': self.blur_spinbox.value(),
            'brightness_min': self.bright_min.value(),
            'brightness_max': self.bright_max.value(),
            'skip_duplicates': self.skip_dup_cb.isChecked()
        }
    
    def update_language(self, lang: str):
        """Update UI language"""
        self.lang = lang
        self.setTitle(get_text('quality_title', lang))
        self.enable_cb.setText(get_text('quality_enabled', lang))
        self.blur_label.setText(get_text('blur_threshold', lang))
        self.bright_label.setText(get_text('brightness_range', lang))
        self.skip_dup_cb.setText(get_text('skip_duplicates', lang))


class CaptioningSettingsPanel(QGroupBox):
    """Captioning Settings Panel - BLIP + WD14"""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(get_text('caption_title', lang), parent)
        self.lang = lang
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setStyleSheet(theme.panel_group())
        
        layout = QVBoxLayout()
        
        # Enable checkbox
        enable_layout = QHBoxLayout()
        self.enable_cb = QCheckBox(get_text('caption_enabled', self.lang))
        self.enable_cb.setStyleSheet(theme.label_accent())
        self.enable_cb.toggled.connect(self._on_enable_toggled)
        enable_layout.addWidget(self.enable_cb)
        enable_layout.addStretch()
        layout.addLayout(enable_layout)
        
        # Settings container
        self.settings_widget = QWidget()
        self.settings_widget.setVisible(False)
        settings_layout = QVBoxLayout(self.settings_widget)
        
        # Caption mode
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel(get_text('caption_mode', self.lang))
        self.mode_label.setStyleSheet(theme.label_default())
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            'tags_only', 'blip_only', 'blip_first', 'tags_first', 'combined'
        ])
        self.mode_combo.setStyleSheet(self._combo_style())
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        settings_layout.addLayout(mode_layout)
        
        # BLIP settings
        blip_layout = QHBoxLayout()
        self.blip_cb = QCheckBox(get_text('blip_enabled', self.lang))
        self.blip_cb.setChecked(True)
        self.blip_cb.setStyleSheet(theme.label_default())
        self.blip_combo = QComboBox()
        self.blip_combo.addItems(['blip-base', 'blip-large'])
        self.blip_combo.setStyleSheet(self._combo_style())
        blip_layout.addWidget(self.blip_cb)
        blip_layout.addWidget(self.blip_combo)
        blip_layout.addStretch()
        settings_layout.addLayout(blip_layout)
        
        # WD14 settings
        wd14_layout = QHBoxLayout()
        self.wd14_cb = QCheckBox(get_text('wd14_enabled', self.lang))
        self.wd14_cb.setChecked(True)
        self.wd14_cb.setStyleSheet(theme.label_default())
        self.wd14_combo = QComboBox()
        self.wd14_combo.addItems([
            'wd-v1-4-vit-tagger-v2',
            'wd-v1-4-convnext-tagger-v2',
            'wd-v1-4-swinv2-tagger-v2'
        ])
        self.wd14_combo.setStyleSheet(self._combo_style())
        wd14_layout.addWidget(self.wd14_cb)
        wd14_layout.addWidget(self.wd14_combo)
        wd14_layout.addStretch()
        settings_layout.addLayout(wd14_layout)
        
        layout.addWidget(self.settings_widget)
        self.setLayout(layout)
    
    def _combo_style(self) -> str:
        return theme.combo_compact()
    
    def _on_enable_toggled(self, checked: bool):
        self.settings_widget.setVisible(checked)
        self.settings_changed.emit()
    
    def get_settings(self) -> Dict:
        return {
            'enabled': self.enable_cb.isChecked(),
            'mode': self.mode_combo.currentText(),
            'blip_enabled': self.blip_cb.isChecked(),
            'blip_model': self.blip_combo.currentText(),
            'wd14_enabled': self.wd14_cb.isChecked(),
            'wd14_model': self.wd14_combo.currentText()
        }
    
    def update_language(self, lang: str):
        """Update UI language"""
        self.lang = lang
        self.setTitle(get_text('caption_title', lang))
        self.enable_cb.setText(get_text('caption_enabled', lang))
        self.mode_label.setText(get_text('caption_mode', lang))
        self.blip_cb.setText(get_text('blip_enabled', lang))
        self.wd14_cb.setText(get_text('wd14_enabled', lang))


class TagSettingsPanel(QGroupBox):
    """Tag Settings Panel - Trigger, Negative Tags, etc."""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(get_text('tag_settings_title', lang), parent)
        self.lang = lang
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setStyleSheet(theme.panel_group())
        
        layout = QVBoxLayout()
        
        # Preset selector
        preset_layout = QHBoxLayout()
        self.preset_label = QLabel(get_text('tag_preset', self.lang))
        self.preset_label.setStyleSheet(theme.label_default())
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            '-- Select Preset --',
            'anime_character',
            'style_lora', 
            'realistic_photo',
            'concept_art'
        ])
        self.preset_combo.setStyleSheet(self._combo_style())
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)
        
        # Trigger word
        trigger_layout = QHBoxLayout()
        self.trigger_label = QLabel(get_text('trigger_word', self.lang))
        self.trigger_label.setStyleSheet(theme.label_default())
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("e.g., sks person, my_character")
        self.trigger_edit.setStyleSheet(self._edit_style())
        trigger_layout.addWidget(self.trigger_label)
        trigger_layout.addWidget(self.trigger_edit)
        layout.addLayout(trigger_layout)
        
        # Max tags and confidence
        limits_layout = QHBoxLayout()
        self.max_tags_label = QLabel(get_text('max_tags', self.lang))
        self.max_tags_label.setStyleSheet(theme.label_default())
        self.max_tags_spin = QSpinBox()
        self.max_tags_spin.setRange(5, 100)
        self.max_tags_spin.setValue(30)
        self.max_tags_spin.setStyleSheet(self._spinbox_style())
        
        self.conf_label = QLabel(get_text('min_confidence', self.lang))
        self.conf_label.setStyleSheet(theme.label_default())
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.9)
        self.conf_spin.setValue(0.35)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setStyleSheet(self._spinbox_style())
        
        limits_layout.addWidget(self.max_tags_label)
        limits_layout.addWidget(self.max_tags_spin)
        limits_layout.addWidget(self.conf_label)
        limits_layout.addWidget(self.conf_spin)
        limits_layout.addStretch()
        layout.addLayout(limits_layout)
        
        # Negative tags
        neg_layout = QVBoxLayout()
        neg_header = QHBoxLayout()
        self.neg_label = QLabel(get_text('negative_tags', self.lang))
        self.neg_label.setStyleSheet(theme.label_default())
        self.neg_help = QLabel("ℹ️")
        self.neg_help.setToolTip(get_text('negative_tags_tooltip', self.lang))
        self.neg_help.setStyleSheet(theme.info_icon())
        self.neg_help.setCursor(Qt.WhatsThisCursor)
        neg_header.addWidget(self.neg_label)
        neg_header.addWidget(self.neg_help)
        neg_header.addStretch()
        neg_layout.addLayout(neg_header)
        
        self.neg_edit = QTextEdit()
        self.neg_edit.setMinimumHeight(45)
        self.neg_edit.setMaximumHeight(90)
        self.neg_edit.setPlaceholderText("watermark, signature, text, username...")
        self.neg_edit.setStyleSheet(theme.text_edit_input())
        # Set default negative tags
        self.neg_edit.setPlainText("watermark, signature, text, username, artist_name, twitter_username, patreon_username, dated")
        neg_layout.addWidget(self.neg_edit)
        layout.addLayout(neg_layout)
        
        # Priority tags
        priority_layout = QVBoxLayout()
        priority_header = QHBoxLayout()
        self.priority_label = QLabel(get_text('priority_tags', self.lang))
        self.priority_label.setStyleSheet(theme.label_default())
        priority_header.addWidget(self.priority_label)
        priority_header.addStretch()
        priority_layout.addLayout(priority_header)
        
        self.priority_edit = QLineEdit()
        self.priority_edit.setPlaceholderText("Tags always included if detected...")
        self.priority_edit.setStyleSheet(self._edit_style())
        priority_layout.addWidget(self.priority_edit)
        layout.addLayout(priority_layout)
        
        # Checkboxes row 1
        cb_layout1 = QHBoxLayout()
        self.keep_char_cb = QCheckBox(get_text('keep_character_tags', self.lang))
        self.keep_char_cb.setChecked(True)
        self.keep_char_cb.setStyleSheet(theme.label_default())
        self.keep_series_cb = QCheckBox(get_text('keep_series_tags', self.lang))
        self.keep_series_cb.setStyleSheet(theme.label_default())
        cb_layout1.addWidget(self.keep_char_cb)
        cb_layout1.addWidget(self.keep_series_cb)
        cb_layout1.addStretch()
        layout.addLayout(cb_layout1)
        
        # Checkboxes row 2
        cb_layout2 = QHBoxLayout()
        self.quality_tags_cb = QCheckBox(get_text('include_quality_tags', self.lang))
        self.quality_tags_cb.setStyleSheet(theme.label_default())
        self.rating_tags_cb = QCheckBox(get_text('include_rating_tags', self.lang))
        self.rating_tags_cb.setStyleSheet(theme.label_default())
        cb_layout2.addWidget(self.quality_tags_cb)
        cb_layout2.addWidget(self.rating_tags_cb)
        cb_layout2.addStretch()
        layout.addLayout(cb_layout2)
        
        # Formatting options
        format_layout = QHBoxLayout()
        self.underscore_cb = QCheckBox(get_text('use_underscores', self.lang))
        self.underscore_cb.setChecked(True)
        self.underscore_cb.setStyleSheet(theme.label_default())
        self.json_cb = QCheckBox(get_text('save_json', self.lang))
        self.json_cb.setStyleSheet(theme.label_default())
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
        
        self.suffix_label = QLabel(get_text('caption_suffix', self.lang))
        self.suffix_label.setStyleSheet(theme.label_default())
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setStyleSheet(self._edit_style())
        self.suffix_edit.setMaximumWidth(200)
        
        prefix_layout.addWidget(self.prefix_label)
        prefix_layout.addWidget(self.prefix_edit)
        prefix_layout.addWidget(self.suffix_label)
        prefix_layout.addWidget(self.suffix_edit)
        prefix_layout.addStretch()
        layout.addLayout(prefix_layout)
        
        self.setLayout(layout)
    
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
            'preset': self.preset_combo.currentText() if self.preset_combo.currentIndex() > 0 else None,
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
    
    def update_language(self, lang: str):
        """Update UI language"""
        self.lang = lang
        self.setTitle(get_text('tag_settings_title', lang))
        self.preset_label.setText(get_text('tag_preset', lang))
        self.trigger_label.setText(get_text('trigger_word', lang))
        self.max_tags_label.setText(get_text('max_tags', lang))
        self.conf_label.setText(get_text('min_confidence', lang))
        self.neg_label.setText(get_text('negative_tags', lang))
        self.priority_label.setText(get_text('priority_tags', lang))
        self.keep_char_cb.setText(get_text('keep_character_tags', lang))
        self.keep_series_cb.setText(get_text('keep_series_tags', lang))
        self.quality_tags_cb.setText(get_text('include_quality_tags', lang))
        self.rating_tags_cb.setText(get_text('include_rating_tags', lang))
        self.underscore_cb.setText(get_text('use_underscores', lang))
        self.json_cb.setText(get_text('save_json', lang))
        self.prefix_label.setText(get_text('caption_prefix', lang))
        self.suffix_label.setText(get_text('caption_suffix', lang))


class AdvancedSettingsDialog(QWidget):
    """
    Advanced Settings Dialog containing all v2.0 features
    Can be embedded in main window or shown as separate dialog
    """
    
    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Create tab widget for organized settings
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(theme.tab_widget())
        
        # Quality tab
        quality_tab = QWidget()
        quality_layout = QVBoxLayout(quality_tab)
        self.quality_panel = QualitySettingsPanel(self.lang)
        quality_layout.addWidget(self.quality_panel)
        quality_layout.addStretch()
        self.tabs.addTab(quality_tab, "🔍 Quality")
        
        # Captioning tab
        caption_tab = QWidget()
        caption_layout = QVBoxLayout(caption_tab)
        self.caption_panel = CaptioningSettingsPanel(self.lang)
        caption_layout.addWidget(self.caption_panel)
        caption_layout.addStretch()
        self.tabs.addTab(caption_tab, "📝 Captioning")
        
        # Tags tab
        tags_tab = QWidget()
        tags_scroll = QScrollArea()
        tags_scroll.setWidgetResizable(True)
        tags_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.tags_panel = TagSettingsPanel(self.lang)
        tags_scroll.setWidget(self.tags_panel)
        tags_layout = QVBoxLayout(tags_tab)
        tags_layout.addWidget(tags_scroll)
        self.tabs.addTab(tags_tab, "🏷️ Tags")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def get_all_settings(self) -> Dict:
        """Get all settings from all panels"""
        return {
            'quality': self.quality_panel.get_settings(),
            'captioning': self.caption_panel.get_settings(),
            'tags': self.tags_panel.get_settings()
        }
    
    def update_language(self, lang: str):
        """Update UI language"""
        self.lang = lang
        # Would need to rebuild panels for full translation support
