"""
Standalone Captioning Page for LoRA-Harvester v2.0
Simple and clean interface for captioning images from any folder
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox,
    QTextEdit, QPushButton, QProgressBar, QFileDialog, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent
from typing import Dict
from src.ui.translations import get_text
from src.ui import theme


class CaptioningThread(QThread):
    """Background thread for captioning"""

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
        self._last_progress_pct = -1  # Throttle progress updates

    def run(self):
        try:
            # Pre-load WD14 model and report errors to UI
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

            # Pre-load BLIP model if needed
            if self.captioner.blip and self.captioner.enable_blip:
                try:
                    self.log_message.emit("Loading BLIP model...")
                    self.captioner.blip._load_model()
                    self.log_message.emit("BLIP model loaded")
                except Exception as e:
                    self.log_message.emit(f"BLIP FAILED: {e} - captioning disabled")
                    self.captioner.enable_blip = False

            if not self._running:
                return

            stats = self.captioner.caption_directory(
                self.image_folder,
                mode=self.settings.get('mode', 'tags_only'),
                overwrite=self.settings.get('overwrite', False),
                save_json=self.settings.get('save_json', False),
                progress_callback=self._progress_callback,
                recursive=self.settings.get('recursive', False)
            )
            if self._running:
                self.finished.emit(stats)
        except Exception as e:
            if self._running:
                self.error.emit(str(e))
        finally:
            # Cleanup captioner models in the worker thread (safe)
            try:
                if self.captioner and hasattr(self.captioner, 'cleanup'):
                    self.captioner.cleanup()
            except Exception:
                pass

    def _progress_callback(self, current: int, total: int, filename: str):
        if not self._running:
            return False
        # Throttle: only emit when percentage changes (prevents UI flood)
        pct = int((current / total) * 100) if total > 0 else 0
        if pct != self._last_progress_pct:
            self._last_progress_pct = pct
            self.progress.emit(current, total, filename)
        return True

    def stop(self):
        self._running = False
        self.requestInterruption()


class StandaloneCaptioningPage(QWidget):
    """Simple and clean captioning interface"""
    
    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self.selected_folder = None
        self.captioner = None
        self.captioning_thread = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize simplified UI"""
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 20, 30, 20)
        
        # Tooltip style handled by global theme

        # ========== TITLE ==========
        self.main_title = QLabel(get_text('captioning_standalone_title', self.lang))
        self.main_title.setFont(QFont('Arial', 20, QFont.Bold))
        self.main_title.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self.main_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.main_title)
        
        # ========== STEP 1: SELECT FOLDER ==========
        step1_frame = QFrame()
        step1_frame.setStyleSheet(theme.card_frame())
        step1_layout = QVBoxLayout(step1_frame)
        
        self.step1_title = QLabel(get_text('step1_select_folder', self.lang))
        self.step1_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step1_title.setStyleSheet(theme.label_section())
        step1_layout.addWidget(self.step1_title)
        
        folder_row = QHBoxLayout()
        
        # Drop zone frame
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
        
        # Options
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
        
        # ========== STEP 2: SETTINGS ==========
        step2_frame = QFrame()
        step2_frame.setStyleSheet(theme.card_frame())
        step2_layout = QVBoxLayout(step2_frame)
        step2_layout.setSpacing(8)
        step2_layout.setContentsMargins(10, 8, 10, 8)
        
        self.step2_title = QLabel(get_text('step2_settings', self.lang))
        self.step2_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step2_title.setStyleSheet(theme.label_section())
        step2_layout.addWidget(self.step2_title)
        
        # Mode & Trigger in one row
        row1 = QHBoxLayout()
        
        self.mode_label = QLabel(get_text('caption_mode', self.lang))
        self.mode_label.setStyleSheet(theme.label_frame())
        
        self.mode_info = QLabel("ℹ️")
        self.mode_info.setToolTip(get_text('mode_tooltip', self.lang))
        self.mode_info.setStyleSheet(theme.info_icon_frame())
        self.mode_info.setCursor(Qt.WhatsThisCursor)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['tags_only', 'blip_only', 'blip_first', 'tags_first', 'combined'])
        self.mode_combo.setStyleSheet(self._combo_style())
        self.mode_combo.setMinimumWidth(120)
        
        self.trigger_label = QLabel(get_text('trigger_word', self.lang))
        self.trigger_label.setStyleSheet(theme.label_frame())
        
        self.trigger_info = QLabel("ℹ️")
        self.trigger_info.setToolTip(get_text('trigger_tooltip', self.lang))
        self.trigger_info.setStyleSheet(theme.info_icon_frame())
        self.trigger_info.setCursor(Qt.WhatsThisCursor)
        
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("sks person...")
        self.trigger_edit.setStyleSheet(self._edit_style())
        self.trigger_edit.setMinimumWidth(150)
        
        row1.addWidget(self.mode_label)
        row1.addWidget(self.mode_info)
        row1.addWidget(self.mode_combo)
        row1.addSpacing(20)
        row1.addWidget(self.trigger_label)
        row1.addWidget(self.trigger_info)
        row1.addWidget(self.trigger_edit)
        row1.addStretch()
        step2_layout.addLayout(row1)
        
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
        self.max_tags_spin.setStyleSheet(self._spinbox_style())
        
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
        self.conf_spin.setStyleSheet(self._spinbox_style())
        
        row2.addWidget(self.max_label)
        row2.addWidget(self.max_info)
        row2.addWidget(self.max_tags_spin)
        row2.addSpacing(20)
        row2.addWidget(self.conf_label)
        row2.addWidget(self.conf_info)
        row2.addWidget(self.conf_spin)
        row2.addStretch()
        step2_layout.addLayout(row2)
        
        # Negative tags row with info
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
        self.neg_edit.setText("watermark, signature, text, username, artist_name, twitter_username, patreon_username, dated")
        self.neg_edit.setStyleSheet(self._edit_style())
        step2_layout.addWidget(self.neg_edit)
        
        # ===== MODEL SELECTION =====
        model_row = QHBoxLayout()
        
        # BLIP Model Selection
        self.blip_cb = QCheckBox(get_text('use_blip', self.lang))
        self.blip_cb.setChecked(False)  # Default off
        self.blip_cb.setStyleSheet(theme.checkbox_frame())
        
        self.blip_info = QLabel("ℹ️")
        self.blip_info.setToolTip(get_text('blip_tooltip', self.lang))
        self.blip_info.setStyleSheet(theme.info_icon_frame())
        self.blip_info.setCursor(Qt.WhatsThisCursor)
        
        self.blip_combo = QComboBox()
        self.blip_combo.addItems([
            'blip-base',
            'blip-large'
        ])
        self.blip_combo.setStyleSheet(self._combo_style())
        self.blip_combo.setMinimumWidth(280)
        self.blip_combo.setToolTip(get_text('blip_model_tooltip', self.lang))
        
        model_row.addWidget(self.blip_cb)
        model_row.addWidget(self.blip_info)
        model_row.addWidget(self.blip_combo)
        model_row.addStretch()
        step2_layout.addLayout(model_row)
        
        # WD14/Danbooru Model Selection
        model_row2 = QHBoxLayout()
        
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
            'SmilingWolf/wd-v1-4-swinv2-tagger-v2'
        ])
        self.wd14_combo.setStyleSheet(self._combo_style())
        self.wd14_combo.setMinimumWidth(280)
        self.wd14_combo.setToolTip(get_text('wd14_model_tooltip', self.lang))
        
        model_row2.addWidget(self.wd14_cb)
        model_row2.addWidget(self.wd14_info)
        model_row2.addWidget(self.wd14_combo)
        model_row2.addStretch()
        step2_layout.addLayout(model_row2)
        
        # Simple checkboxes
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
        
        # ========== STEP 3: START ==========
        step3_frame = QFrame()
        step3_frame.setStyleSheet(theme.card_frame())
        step3_layout = QVBoxLayout(step3_frame)
        
        self.step3_title = QLabel(get_text('step3_start', self.lang))
        self.step3_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step3_title.setStyleSheet(theme.label_section())
        step3_layout.addWidget(self.step3_title)
        
        # Buttons
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
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(theme.progress_bar())
        step3_layout.addWidget(self.progress_bar)
        
        layout.addWidget(step3_frame)
        
        # ========== LOG ==========
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(105)
        self.log_text.setMaximumHeight(225)
        self.log_text.setStyleSheet(theme.log_area())
        layout.addWidget(self.log_text)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _combo_style(self) -> str:
        return theme.combo()

    def _edit_style(self) -> str:
        return theme.line_edit()

    def _spinbox_style(self) -> str:
        return theme.spinbox()

    def refresh_styles(self):
        """Re-apply all stylesheets after a theme change."""
        self.main_title.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        # Step frames — iterate the three card frames
        for frame in self.findChildren(QFrame):
            # Only the three top-level step frames have card_frame style
            if frame is self.drop_zone:
                continue
            ss = frame.styleSheet()
            if "border-radius: 10px" in ss:
                frame.setStyleSheet(theme.card_frame())
        self.step1_title.setStyleSheet(theme.label_section())
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())
        self.drop_icon.setStyleSheet(theme.icon_transparent())
        self.folder_label.setStyleSheet(theme.label_transparent())
        self.browse_btn.setStyleSheet(theme.btn_browse())
        self.image_count_label.setStyleSheet(theme.label_success())
        self.step2_title.setStyleSheet(theme.label_section())
        self.mode_label.setStyleSheet(theme.label_frame())
        self.mode_info.setStyleSheet(theme.info_icon_frame())
        self.mode_combo.setStyleSheet(self._combo_style())
        self.trigger_label.setStyleSheet(theme.label_frame())
        self.trigger_info.setStyleSheet(theme.info_icon_frame())
        self.trigger_edit.setStyleSheet(self._edit_style())
        self.max_label.setStyleSheet(theme.label_frame())
        self.max_info.setStyleSheet(theme.info_icon_frame())
        self.max_tags_spin.setStyleSheet(self._spinbox_style())
        self.conf_label.setStyleSheet(theme.label_frame())
        self.conf_info.setStyleSheet(theme.info_icon_frame())
        self.conf_spin.setStyleSheet(self._spinbox_style())
        self.neg_label.setStyleSheet(theme.label_frame())
        self.neg_info.setStyleSheet(theme.info_icon_frame_compact())
        self.neg_edit.setStyleSheet(self._edit_style())
        self.blip_cb.setStyleSheet(theme.checkbox_frame())
        self.blip_info.setStyleSheet(theme.info_icon_frame())
        self.blip_combo.setStyleSheet(self._combo_style())
        self.wd14_cb.setStyleSheet(theme.checkbox_frame())
        self.wd14_info.setStyleSheet(theme.info_icon_frame())
        self.wd14_combo.setStyleSheet(self._combo_style())
        self.step3_title.setStyleSheet(theme.label_section())
        self.start_btn.setStyleSheet(theme.btn_primary())
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.progress_bar.setStyleSheet(theme.progress_bar())
        self.log_text.setStyleSheet(theme.log_area())
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    self.drop_zone.setStyleSheet(theme.drop_zone_frame_active())
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
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
        """Select image folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self._set_folder(folder)
    
    def _set_folder(self, folder: str):
        """Set the selected folder and update UI"""
        self.selected_folder = folder
        
        # Shorten path for display
        display_path = folder
        if len(folder) > 50:
            display_path = "..." + folder[-47:]
        
        self.folder_label.setText(display_path)
        self.folder_label.setStyleSheet(theme.label_success())
        
        self.drop_icon.setText("✅")
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_success())
        self.drop_zone.setToolTip(folder)
        
        # Count images
        count = self._count_images(folder)
        self.image_count_label.setText(get_text('images_found', self.lang).format(count))
        self.start_btn.setEnabled(count > 0)
        self.log(f"📁 {get_text('folder_selected', self.lang)}: {folder}")
        self.log(f"🖼️ {count} images found")
    
    def _count_images(self, folder: str) -> int:
        """Count image files in folder"""
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        count = 0
        folder_path = Path(folder)
        
        if self.recursive_cb.isChecked():
            for ext in extensions:
                count += len(list(folder_path.rglob(f'*{ext}')))
        else:
            for ext in extensions:
                count += len(list(folder_path.glob(f'*{ext}')))
        
        return count
    
    def get_settings(self) -> Dict:
        """Get all settings"""
        return {
            'mode': self.mode_combo.currentText(),
            'trigger_word': self.trigger_edit.text().strip(),
            'max_tags': self.max_tags_spin.value(),
            'min_confidence': self.conf_spin.value(),
            'negative_tags': [t.strip() for t in self.neg_edit.text().split(',') if t.strip()],
            'keep_character_tags': self.keep_char_cb.isChecked(),
            'save_json': self.json_cb.isChecked(),
            'overwrite': self.overwrite_cb.isChecked(),
            'recursive': self.recursive_cb.isChecked(),
            'use_blip': self.blip_cb.isChecked(),
            'blip_model': self.blip_combo.currentText(),
            'use_wd14': self.wd14_cb.isChecked(),
            'wd14_model': self.wd14_combo.currentText()
        }
    
    def start_captioning(self):
        """Start captioning process"""
        if not self.selected_folder:
            return
        
        self.log(f"\n{'='*40}")
        self.log("🚀 " + get_text('log_captioning', self.lang))
        
        settings = self.get_settings()
        self.log(f"   Mode: {settings['mode']}")
        if settings['trigger_word']:
            self.log(f"   Trigger: {settings['trigger_word']}")
        
        try:
            from src.core.advanced_captioner import AdvancedCaptioner, TagSettings
            
            tag_settings = TagSettings(
                trigger_word=settings['trigger_word'] or "",
                max_tags=settings['max_tags'],
                min_confidence=settings['min_confidence'],
                negative_tags=settings['negative_tags'],
                keep_character_tags=settings['keep_character_tags']
            )
            
            self.captioner = AdvancedCaptioner(
                tag_settings=tag_settings,
                blip_model=settings['blip_model'],
                wd14_model=settings['wd14_model'],
                enable_blip=settings['use_blip'],
                enable_wd14=settings['use_wd14']
            )
            
            # Store caption mode for thread
            self.caption_mode = settings['mode']
            
            self.log("✅ Captioner initialized")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        
        self.captioning_thread = CaptioningThread(
            self.captioner,
            self.selected_folder,
            settings
        )
        self.captioning_thread.progress.connect(self._on_progress)
        self.captioning_thread.log_message.connect(self.log)
        self.captioning_thread.finished.connect(self._on_finished)
        self.captioning_thread.error.connect(self._on_error)
        self.captioning_thread.finished.connect(self._safe_delete_thread)
        self.captioning_thread.start()
    
    def stop_captioning(self):
        """Stop captioning"""
        if self.captioning_thread and self.captioning_thread.isRunning():
            self.captioning_thread.stop()
            self.log("Stopping... (waiting for current batch to finish)")
            self.stop_btn.setEnabled(False)
            # Don't re-enable buttons here — _on_finished or _on_error will handle it
    
    def _on_progress(self, current: int, total: int, filename: str):
        """Progress update"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{current}/{total} - {filename}")
    
    def _on_finished(self, stats: Dict):
        """Captioning finished"""
        self.progress_bar.setValue(100)
        self.log(f"\n{'='*40}")
        self.log(f"✅ Complete! Captioned: {stats.get('captioned', 0)} images")
        self.log(f"   Skipped: {stats.get('skipped', 0)}")
        self.log(f"   Errors: {stats.get('errors', 0)}")
        zero_tags = stats.get('zero_tags', 0)
        if zero_tags > 0:
            self.log(f"   ⚠️ Zero auto-tags: {zero_tags} images (only trigger word written)")
            self.log(f"   → Check WD14 model loading and min_confidence setting")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self._cleanup_captioner()

    def _on_error(self, error: str):
        """Error occurred"""
        self.log(f"❌ Error: {error}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._cleanup_captioner()

    def _safe_delete_thread(self):
        """Safely delete the captioning thread after it finishes"""
        if self.captioning_thread:
            self.captioning_thread.wait(3000)  # Wait up to 3s for thread to finish
            self.captioning_thread.deleteLater()
            self.captioning_thread = None
        # Captioner cleanup is done in the thread's finally block
        self.captioner = None

    def _cleanup_captioner(self):
        """Release captioner models from memory"""
        # Stop thread first if still running
        if self.captioning_thread and self.captioning_thread.isRunning():
            self.captioning_thread.stop()
            self.captioning_thread.wait(5000)
        if hasattr(self, 'captioner') and self.captioner:
            if hasattr(self.captioner, 'cleanup'):
                try:
                    self.captioner.cleanup()
                except Exception:
                    pass
            self.captioner = None
        self.browse_btn.setEnabled(True)
    
    def update_language(self, lang: str):
        """Update UI language - rebuild UI text elements"""
        self.lang = lang
        
        # Update titles (without emoji duplicates)
        self.main_title.setText(get_text('captioning_standalone_title', lang))
        self.step1_title.setText(get_text('step1_select_folder', lang))
        self.step2_title.setText(get_text('step2_settings', lang))
        self.step3_title.setText(get_text('step3_start', lang))
        
        # Update buttons (without emoji duplicates)
        self.browse_btn.setText(get_text('select_input_folder', lang))
        self.start_btn.setText(get_text('start_captioning', lang))
        self.stop_btn.setText(get_text('stop_btn', lang))
        
        # Update folder label
        if not self.selected_folder:
            self.folder_label.setText(get_text('drag_drop_folder', lang))
        
        # Update labels
        self.mode_label.setText(get_text('caption_mode', lang))
        self.trigger_label.setText(get_text('trigger_word', lang))
        self.max_label.setText(get_text('max_tags', lang))
        self.conf_label.setText(get_text('min_confidence', lang))
        self.neg_label.setText(get_text('negative_tags', lang))
        
        # Update checkboxes
        self.recursive_cb.setText(get_text('recursive_search', lang))
        self.overwrite_cb.setText(get_text('overwrite_existing', lang))
        self.keep_char_cb.setText(get_text('keep_character_tags', lang))
        self.json_cb.setText(get_text('save_json', lang))
        self.blip_cb.setText(get_text('use_blip', lang))
        self.wd14_cb.setText(get_text('use_wd14', lang))
        
        # Update info tooltips
        self.mode_info.setToolTip(get_text('mode_tooltip', lang))
        self.trigger_info.setToolTip(get_text('trigger_tooltip', lang))
        self.max_info.setToolTip(get_text('max_tags_tooltip', lang))
        self.conf_info.setToolTip(get_text('confidence_tooltip', lang))
        self.neg_info.setToolTip(get_text('negative_tooltip', lang))
        self.blip_info.setToolTip(get_text('blip_tooltip', lang))
        self.blip_combo.setToolTip(get_text('blip_model_tooltip', lang))
        self.wd14_info.setToolTip(get_text('wd14_tooltip', lang))
        self.wd14_combo.setToolTip(get_text('wd14_model_tooltip', lang))
        self.keep_char_cb.setToolTip(get_text('keep_char_tooltip', lang))
        self.json_cb.setToolTip(get_text('json_tooltip', lang))
        self.recursive_cb.setToolTip(get_text('recursive_tooltip', lang))
        self.overwrite_cb.setToolTip(get_text('overwrite_tooltip', lang))
