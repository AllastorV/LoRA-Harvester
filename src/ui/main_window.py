"""
Modern UI Module for LoRA-Harvester v2.0
AI-Powered Dataset Collection Tool with PyQt5 interface
Now with Quality Analysis, BLIP + WD14 Captioning, and Advanced Tag Settings
Featuring Page-based navigation: Video Processing + Standalone Captioning
"""

import sys
import os
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QComboBox, QCheckBox, QProgressBar, QFileDialog,
                             QTextEdit, QGroupBox, QSpinBox, QToolButton,
                             QScrollArea, QStackedWidget, QFrame, QDesktopWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent, QFontMetrics
from typing import Optional, List, Dict
from src.ui.translations import get_text
from src.ui import theme
from src.ui.advanced_settings import (
    QualitySettingsPanel, 
    CaptioningSettingsPanel, 
    TagSettingsPanel
)
from src.ui.captioning_page import StandaloneCaptioningPage
from src.ui.character_sort_page import CharacterSortPage


class ProcessingThread(QThread):
    """Background thread for video processing - all heavy work runs here"""

    progress_update = pyqtSignal(float, dict)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_running = True
        self._finished_emitted = False
        self.processor = None

    def run(self):
        """Initialize models and run video processing"""
        try:
            cfg = self.config

            # --- All model loading happens here (background thread) ---
            from src.core.text_detector import SubtitleDetector
            from src.core.cropper import SmartCropper
            from src.core.unified_processor import UnifiedVideoProcessor

            # Detector
            if cfg['use_ensemble']:
                from src.core.ensemble_detector import EnsembleDetector
                self.log_message.emit("Loading ensemble models...")
                detector = EnsembleDetector(
                    models_to_use=cfg['models_to_use'],
                    confidence_threshold=cfg['confidence'],
                    voting_threshold=cfg['voting_threshold']
                )
                self.log_message.emit(f"Ensemble loaded: {', '.join(cfg['models_to_use'])}")
            else:
                from src.core.detector import ObjectDetector
                self.log_message.emit("Loading YOLO model...")
                detector = ObjectDetector(confidence=cfg['confidence'])
                self.log_message.emit("YOLO model loaded")

            if not self._is_running:
                return

            text_detector = SubtitleDetector() if cfg['skip_text'] else None
            cropper = SmartCropper(
                target_format=cfg['aspect_ratio'],
                min_padding=cfg['min_padding']
            )

            # Quality analyzer
            quality_analyzer = None
            if cfg['quality_settings']['enabled']:
                try:
                    from src.core.quality_analyzer import QualityAnalyzer
                    qs = cfg['quality_settings']
                    quality_analyzer = QualityAnalyzer(
                        blur_threshold=qs['blur_threshold'],
                        brightness_range=(qs['brightness_min'], qs['brightness_max']),
                        check_duplicates=qs.get('skip_duplicates', True),
                    )
                    self.log_message.emit("Quality analyzer initialized")
                except ImportError as e:
                    self.log_message.emit(f"Quality analyzer not available: {e}")

            if not self._is_running:
                return

            # Captioner
            captioner = None
            caption_mode = cfg['caption_settings'].get('mode', 'tags_only')
            if cfg['caption_settings']['enabled']:
                try:
                    from src.core.advanced_captioner import AdvancedCaptioner, TagSettings
                    ts = cfg['tag_settings']
                    tag_cfg = TagSettings(
                        trigger_word=ts['trigger_word'] or "",
                        max_tags=ts['max_tags'],
                        min_confidence=ts['min_confidence'],
                        negative_tags=ts['negative_tags'],
                        priority_tags=ts['priority_tags'],
                        keep_character_tags=ts['keep_character_tags'],
                        keep_series_tags=ts['keep_series_tags'],
                        include_quality_tags=ts['include_quality_tags'],
                        include_rating_tags=ts['include_rating_tags'],
                        use_underscores=ts['use_underscores'],
                        caption_prefix=ts['caption_prefix'] or "",
                        caption_suffix=ts['caption_suffix'] or "",
                    )
                    cs = cfg['caption_settings']
                    captioner = AdvancedCaptioner(
                        enable_blip=cs['blip_enabled'],
                        enable_wd14=cs['wd14_enabled'],
                        blip_model=cs['blip_model'],
                        wd14_model=cs['wd14_model'],
                        tag_settings=tag_cfg,
                    )
                except Exception as e:
                    self.log_message.emit(f"Captioner init error: {e}")
                    captioner = None

                # Pre-load models (errors disable the failed model)
                if captioner:
                    if captioner.wd14 and captioner.enable_wd14:
                        try:
                            self.log_message.emit("Loading WD14 model...")
                            captioner.wd14._load_model()
                            n_tags = len(captioner.wd14.tags) if captioner.wd14.tags else 0
                            self.log_message.emit(f"✅ WD14 model loaded ({n_tags} tags)")
                            if n_tags == 0:
                                self.log_message.emit("⚠️ WD14 tag list is empty — auto-tags will NOT be generated!")
                            # Quick validation: check that model output matches tag count
                            n_outputs = captioner.wd14.model.get_outputs()[0].shape
                            self.log_message.emit(f"WD14 model output shape: {n_outputs}")
                        except Exception as e:
                            self.log_message.emit(f"❌ WD14 FAILED: {e}")
                            self.log_message.emit("⚠️ Auto-tagging disabled — captions will only contain trigger word!")
                            captioner.enable_wd14 = False

                    if not self._is_running:
                        return

                    if captioner.blip and captioner.enable_blip:
                        try:
                            self.log_message.emit("Loading BLIP model...")
                            captioner.blip._load_model()
                            self.log_message.emit("BLIP model loaded")
                        except Exception as e:
                            self.log_message.emit(f"BLIP FAILED: {e} - captioning disabled")
                            captioner.enable_blip = False

                    # Log final captioning status
                    wd14_ok = captioner.wd14 and captioner.enable_wd14
                    blip_ok = captioner.blip and captioner.enable_blip
                    self.log_message.emit(
                        f"📝 Captioning: mode={caption_mode} WD14={'ON' if wd14_ok else 'OFF'} BLIP={'ON' if blip_ok else 'OFF'}"
                    )
                    # If no model is active, warn user
                    if not wd14_ok and not blip_ok:
                        self.log_message.emit("⚠️ Both captioning models disabled - no tags will be generated!")
                        captioner = None
            else:
                self.log_message.emit("📝 Auto-captioning: disabled (enable in Captioning settings)")

            if not self._is_running:
                return

            # Create processor
            self.processor = UnifiedVideoProcessor(
                video_paths=cfg['video_paths'],
                output_dir="output",
                detector=detector,
                text_detector=text_detector,
                cropper=cropper,
                use_turbo=cfg['use_turbo'],
                batch_size=4,
                quality_analyzer=quality_analyzer,
                captioner=captioner,
                caption_mode=caption_mode,
                log_callback=lambda msg: self.log_message.emit(msg),
            )

            self.log_message.emit("All models loaded, processing started...")

            # Process
            stats = self.processor.process_all_videos(
                frame_interval=cfg['frame_interval'],
                skip_text=cfg['skip_text'],
                progress_callback=self.progress_callback,
                stop_callback=self.should_stop,
            )

            if not self._is_running:
                # Stopped by user - emit stopped signal (only once)
                if not self._finished_emitted:
                    self._finished_emitted = True
                    self.finished.emit({'stopped': True, 'total_frames_saved': 0})
            else:
                # Normal completion
                if not self._finished_emitted:
                    self._finished_emitted = True
                    self.finished.emit(stats)

        except Exception as e:
            if not self._finished_emitted:
                self._finished_emitted = True
                self.error.emit(str(e))
        finally:
            self._cleanup()
            # Safety net: if stopped during model loading (early return), emit finished
            if not self._is_running and not self._finished_emitted:
                self._finished_emitted = True
                self.finished.emit({'stopped': True, 'total_frames_saved': 0})

    def _cleanup(self):
        """Release models and free GPU memory"""
        try:
            if self.processor:
                if hasattr(self.processor, 'detector') and self.processor.detector:
                    if hasattr(self.processor.detector, 'cleanup'):
                        self.processor.detector.cleanup()
                if hasattr(self.processor, 'captioner') and self.processor.captioner:
                    if hasattr(self.processor.captioner, 'cleanup'):
                        self.processor.captioner.cleanup()
                self.processor = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def progress_callback(self, progress, stats):
        """Callback for progress updates"""
        self.progress_update.emit(progress, stats)

    def should_stop(self):
        """Check if processing should stop"""
        return not self._is_running

    def stop(self):
        """Stop processing gracefully"""
        self._is_running = False

    def safe_wait(self, timeout_ms=5000):
        """Wait for thread to finish with timeout, return True if finished"""
        if self.isRunning():
            return self.wait(timeout_ms)
        return True


class DropZone(QLabel):
    """Drag and drop zone for video files - supports multiple files"""
    
    files_dropped = pyqtSignal(list)  # Changed to list
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setStyleSheet(theme.drop_zone_default())
        self.setText("🎬 Drag & Drop Video File(s) Here\n(Supports multiple videos)\nor click 'Browse' button")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(theme.drop_zone_active())
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave"""
        self.setStyleSheet(theme.drop_zone_default())
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop - supports multiple files"""
        self.setStyleSheet(theme.drop_zone_default())
        
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        
        # Filter valid video files
        video_files = [f for f in files if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        if video_files:
            self.files_dropped.emit(video_files)
            if len(video_files) == 1:
                self.setText(f"✅ {Path(video_files[0]).name}")
            else:
                self.setText(f"✅ {len(video_files)} videos selected")
        else:
            self.setText("❌ Invalid file type. Please drop video file(s).")


class VideoSmartCropperUI(QMainWindow):
    """Main UI window with page-based navigation"""
    
    def __init__(self):
        super().__init__()
        self.video_paths = []  # Changed to list for batch support
        self.processor = None
        self.processing_thread = None
        self.current_lang = 'en'  # Default to English
        
        self.init_ui()
    
    @staticmethod
    def _set_taskbar_icon(hwnd):
        """Set taskbar and title bar icons via WM_SETICON."""
        try:
            import ctypes
            from ctypes import wintypes

            icon_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'icon.ico'
            ))
            if not os.path.exists(icon_path):
                return

            WM_SETICON = 0x0080
            ICON_BIG = 1
            ICON_SMALL = 0
            LR_LOADFROMFILE = 0x0010
            IMAGE_ICON = 1

            user32 = ctypes.windll.user32
            user32.LoadImageW.restype = wintypes.HANDLE
            user32.SendMessageW.restype = wintypes.LPARAM

            # Big icon (taskbar, alt-tab) — 256x256 or system default
            sm_cxicon = user32.GetSystemMetrics(11)   # SM_CXICON (usually 32)
            sm_cyicon = user32.GetSystemMetrics(12)   # SM_CYICON
            hicon_big = user32.LoadImageW(
                None, icon_path, IMAGE_ICON, sm_cxicon, sm_cyicon, LR_LOADFROMFILE
            )
            # Small icon (title bar) — 16x16
            sm_cxsmicon = user32.GetSystemMetrics(49)  # SM_CXSMICON
            sm_cysmicon = user32.GetSystemMetrics(50)  # SM_CYSMICON
            hicon_small = user32.LoadImageW(
                None, icon_path, IMAGE_ICON, sm_cxsmicon, sm_cysmicon, LR_LOADFROMFILE
            )

            if hicon_big:
                user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
            if hicon_small:
                user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
        except Exception:
            pass

    def showEvent(self, event):
        """Override showEvent to set native icon after window is fully realized."""
        super().showEvent(event)
        if sys.platform == 'win32' and not getattr(self, '_icon_set', False):
            self._icon_set = True
            # Must defer slightly — HWND may not be fully ready on first showEvent
            QTimer.singleShot(50, lambda: self._set_taskbar_icon(int(self.winId())))

    @staticmethod
    def _enable_dark_title_bar(hwnd):
        """Enable Windows dark title bar via DWM API"""
        try:
            import ctypes
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20  # Windows 11 / 10 build 18985+
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(value), ctypes.sizeof(value)
            )
        except Exception:
            pass
    
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle(get_text('app_title', self.current_lang))
        # Set window icon from assets (taskbar + title bar)
        _icon_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets'))
        for _icon_name in ('icon.ico', 'icon.png'):
            _icon_path = os.path.join(_icon_dir, _icon_name)
            if os.path.exists(_icon_path):
                self.setWindowIcon(QIcon(_icon_path))
                break
        # Dark title bar on Windows
        if sys.platform == 'win32':
            self._enable_dark_title_bar(int(self.winId()))
        
        # Screen-relative sizing
        screen = QDesktopWidget().availableGeometry()
        win_w = min(1000, int(screen.width() * 0.65))
        win_h = min(850, int(screen.height() * 0.85))
        self.resize(win_w, win_h)
        # Centre on screen
        self.move((screen.width() - win_w) // 2,
                  (screen.height() - win_h) // 2)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # ========== TOP BAR: Language + Page Navigation ==========
        top_bar = QHBoxLayout()
        
        # Page navigation buttons
        self.page_video_btn = QPushButton(get_text('page_video_processing', self.current_lang))
        self.page_video_btn.setStyleSheet(self._page_btn_style(True))
        self.page_video_btn.clicked.connect(lambda: self.switch_page(0))

        self.page_caption_btn = QPushButton(get_text('page_captioning', self.current_lang))
        self.page_caption_btn.setStyleSheet(self._page_btn_style(False))
        self.page_caption_btn.clicked.connect(lambda: self.switch_page(1))

        self.page_char_sort_btn = QPushButton(get_text('page_character_sort', self.current_lang))
        self.page_char_sort_btn.setStyleSheet(self._page_btn_style(False))
        self.page_char_sort_btn.clicked.connect(lambda: self.switch_page(2))

        top_bar.addWidget(self.page_video_btn)
        top_bar.addWidget(self.page_caption_btn)
        top_bar.addWidget(self.page_char_sort_btn)
        top_bar.addStretch()
        
        # Language selector
        lang_label = QLabel("🌐")
        lang_label.setFont(QFont('Arial', 14))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['🇬🇧 English', '🇹🇷 Türkçe'])
        self.lang_combo.setCurrentIndex(0)  # English default
        self.lang_combo.setStyleSheet(theme.combo())
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        top_bar.addWidget(lang_label)
        top_bar.addWidget(self.lang_combo)
        main_layout.addLayout(top_bar)
        
        # ========== STACKED WIDGET FOR PAGES ==========
        self.page_stack = QStackedWidget()
        
        # Page 1: Video Processing (wrapped in scroll area)
        self.video_page = QWidget()
        self.setup_video_page()
        
        video_scroll = QScrollArea()
        video_scroll.setWidgetResizable(True)
        video_scroll.setWidget(self.video_page)
        video_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(video_scroll)
        
        # Page 2: Standalone Captioning (wrapped in scroll area)
        self.captioning_page = StandaloneCaptioningPage(self.current_lang)
        caption_scroll = QScrollArea()
        caption_scroll.setWidgetResizable(True)
        caption_scroll.setWidget(self.captioning_page)
        caption_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(caption_scroll)

        # Page 3: Character Sort
        self.char_sort_page = CharacterSortPage(self.current_lang)
        char_scroll = QScrollArea()
        char_scroll.setWidgetResizable(True)
        char_scroll.setWidget(self.char_sort_page)
        char_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(char_scroll)

        main_layout.addWidget(self.page_stack)
        
        # Apply unified dark theme (black/gray/orange)
        self.setStyleSheet(theme.global_stylesheet())
    
    def _page_btn_style(self, active: bool) -> str:
        """Style for page navigation buttons"""
        if active:
            return theme.page_btn_active()
        else:
            return theme.page_btn_inactive()
    
    def _collapsible_btn_style(self, color: str = None) -> str:
        """Style for collapsible panel buttons (unified orange)"""
        return theme.collapsible_btn()
    
    def _toggle_panel(self, panel_name: str):
        """Toggle visibility of collapsible panels"""
        panels = {
            'quality': (self.quality_btn, self.quality_panel),
            'caption': (self.caption_btn, self.caption_panel),
            'tags': (self.tags_btn, self.tags_panel)
        }
        
        btn, panel = panels[panel_name]
        is_checked = btn.isChecked()
        panel.setVisible(is_checked)
    
    def switch_page(self, index: int):
        """Switch between pages"""
        self.page_stack.setCurrentIndex(index)
        self.page_video_btn.setStyleSheet(self._page_btn_style(index == 0))
        self.page_caption_btn.setStyleSheet(self._page_btn_style(index == 1))
        self.page_char_sort_btn.setStyleSheet(self._page_btn_style(index == 2))
    
    def setup_video_page(self):
        """Setup video processing page"""
        layout = QVBoxLayout()
        self.video_page.setLayout(layout)
        
        # Title
        self.title_label = QLabel(get_text('title', self.current_lang))
        self.title_label.setFont(QFont('Arial', 24, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(theme.label_title())
        layout.addWidget(self.title_label)
        
        # Subtitle
        self.subtitle_label = QLabel(get_text('subtitle', self.current_lang))
        self.subtitle_label.setFont(QFont('Arial', 11))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet(theme.label_muted())
        layout.addWidget(self.subtitle_label)
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.on_files_dropped)  # Updated signal
        self.update_drop_zone_text()
        layout.addWidget(self.drop_zone)
        
        # Browse button
        self.browse_btn = QPushButton(get_text('browse_btn', self.current_lang))
        self.browse_btn.setStyleSheet(theme.btn_browse())
        self.browse_btn.clicked.connect(self.browse_video)
        layout.addWidget(self.browse_btn)
        
        # Settings Group
        self.settings_group = QGroupBox(get_text('settings_title', self.current_lang))
        self.settings_group.setStyleSheet(theme.group_box())
        settings_layout = QVBoxLayout()
        self.settings_group.setLayout(settings_layout)
        
        # Frame interval slider
        interval_layout = QHBoxLayout()
        self.interval_label = QLabel(get_text('frame_interval', self.current_lang))
        self.interval_label.setStyleSheet(theme.label_default())
        self.interval_help = QLabel("ℹ️")
        self.interval_help.setStyleSheet(theme.info_icon())
        self.interval_help.setToolTip(get_text('frame_interval_tooltip', self.current_lang))
        self.interval_help.setCursor(Qt.WhatsThisCursor)
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setMinimum(1)
        self.interval_slider.setMaximum(120)
        self.interval_slider.setValue(30)
        self.interval_slider.setTickPosition(QSlider.TicksBelow)
        self.interval_slider.setTickInterval(10)
        self.interval_slider.setStyleSheet(theme.slider())
        self.interval_value_label = QLabel("30")
        self.interval_value_label.setStyleSheet(theme.label_value())
        self.interval_slider.valueChanged.connect(
            lambda v: self.interval_value_label.setText(str(v))
        )
        interval_layout.addWidget(self.interval_label)
        interval_layout.addWidget(self.interval_help)
        interval_layout.addWidget(self.interval_slider)
        interval_layout.addWidget(self.interval_value_label)
        settings_layout.addLayout(interval_layout)
        
        # Aspect ratio selector
        ratio_layout = QHBoxLayout()
        self.ratio_label = QLabel(get_text('output_format', self.current_lang))
        self.ratio_label.setStyleSheet(theme.label_default())
        self.ratio_help = QLabel("ℹ️")
        self.ratio_help.setStyleSheet(theme.info_icon())
        self.ratio_help.setToolTip(get_text('output_format_tooltip', self.current_lang))
        self.ratio_help.setCursor(Qt.WhatsThisCursor)
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(['9:16', '3:4', '1:1', '4:5', '16:9', '4:3'])
        self.ratio_combo.setStyleSheet(theme.combo())
        ratio_layout.addWidget(self.ratio_label)
        ratio_layout.addWidget(self.ratio_help)
        ratio_layout.addWidget(self.ratio_combo)
        ratio_layout.addStretch()
        settings_layout.addLayout(ratio_layout)
        
        # Detection confidence
        conf_layout = QHBoxLayout()
        self.conf_label = QLabel(get_text('confidence', self.current_lang))
        self.conf_label.setStyleSheet(theme.label_default())
        self.conf_help = QLabel("ℹ️")
        self.conf_help.setStyleSheet(theme.info_icon())
        self.conf_help.setToolTip(get_text('confidence_tooltip', self.current_lang))
        self.conf_help.setCursor(Qt.WhatsThisCursor)
        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setMinimum(10)
        self.conf_spinbox.setMaximum(95)
        self.conf_spinbox.setValue(50)
        self.conf_spinbox.setSuffix("%")
        self.conf_spinbox.setStyleSheet(theme.spinbox())
        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_help)
        conf_layout.addWidget(self.conf_spinbox)
        conf_layout.addStretch()
        settings_layout.addLayout(conf_layout)
        
        # Ensemble mode checkbox
        ensemble_layout_cb = QHBoxLayout()
        self.ensemble_cb = QCheckBox(get_text('ensemble_mode', self.current_lang))
        self.ensemble_cb.setChecked(False)
        self.ensemble_cb.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {theme.ORANGE_LIGHT};")
        self.ensemble_help = QLabel("ℹ️")
        self.ensemble_help.setStyleSheet(theme.info_icon())
        self.ensemble_help.setToolTip(get_text('ensemble_mode_tooltip', self.current_lang))
        self.ensemble_help.setCursor(Qt.WhatsThisCursor)
        ensemble_layout_cb.addWidget(self.ensemble_cb)
        ensemble_layout_cb.addWidget(self.ensemble_help)
        ensemble_layout_cb.addStretch()
        settings_layout.addLayout(ensemble_layout_cb)
        
        # Ensemble settings (initially hidden)
        self.ensemble_group = QGroupBox(get_text('ensemble_settings', self.current_lang))
        self.ensemble_group.setVisible(False)
        self.ensemble_group.setStyleSheet(theme.panel_group())
        ensemble_layout = QVBoxLayout()
        
        # Model selection checkboxes
        models_layout = QHBoxLayout()
        self.models_label = QLabel(get_text('active_models', self.current_lang))
        self.models_label.setStyleSheet(theme.label_default())
        self.yolo_cb = QCheckBox("YOLOv8")
        self.yolo_cb.setChecked(True)
        self.yolo_cb.setStyleSheet(theme.label_default())
        self.detr_cb = QCheckBox("DETR (Transformer)")
        self.detr_cb.setChecked(True)
        self.detr_cb.setStyleSheet(theme.label_default())
        self.fasterrcnn_cb = QCheckBox("Faster R-CNN")
        self.fasterrcnn_cb.setChecked(True)
        self.fasterrcnn_cb.setStyleSheet(theme.label_default())
        models_layout.addWidget(self.models_label)
        models_layout.addWidget(self.yolo_cb)
        models_layout.addWidget(self.detr_cb)
        models_layout.addWidget(self.fasterrcnn_cb)
        models_layout.addStretch()
        ensemble_layout.addLayout(models_layout)
        
        # Voting threshold
        voting_layout = QHBoxLayout()
        self.voting_label = QLabel(get_text('voting_threshold', self.current_lang))
        self.voting_label.setStyleSheet(theme.label_default())
        self.voting_help = QLabel("ℹ️")
        self.voting_help.setStyleSheet(theme.info_icon())
        self.voting_help.setToolTip(get_text('voting_threshold_tooltip', self.current_lang))
        self.voting_help.setCursor(Qt.WhatsThisCursor)
        self.voting_spinbox = QSpinBox()
        self.voting_spinbox.setMinimum(1)
        self.voting_spinbox.setMaximum(3)
        self.voting_spinbox.setValue(2)
        self.voting_spinbox.setStyleSheet(theme.spinbox())
        voting_layout.addWidget(self.voting_label)
        voting_layout.addWidget(self.voting_help)
        voting_layout.addWidget(self.voting_spinbox)
        voting_layout.addStretch()
        ensemble_layout.addLayout(voting_layout)
        
        self.ensemble_group.setLayout(ensemble_layout)
        settings_layout.addWidget(self.ensemble_group)
        
        # Connect ensemble checkbox to show/hide settings
        self.ensemble_cb.toggled.connect(self.ensemble_group.setVisible)
        
        # Skip subtitle checkbox
        skip_layout_cb = QHBoxLayout()
        self.skip_subtitle_cb = QCheckBox(get_text('skip_subtitle', self.current_lang))
        self.skip_subtitle_cb.setChecked(True)
        self.skip_subtitle_cb.setStyleSheet(f"font-size: 12px; color: {theme.TEXT_PRIMARY};")
        self.skip_help = QLabel("ℹ️")
        self.skip_help.setStyleSheet(theme.info_icon())
        self.skip_help.setToolTip(get_text('skip_subtitle_tooltip', self.current_lang))
        self.skip_help.setCursor(Qt.WhatsThisCursor)
        skip_layout_cb.addWidget(self.skip_subtitle_cb)
        skip_layout_cb.addWidget(self.skip_help)
        skip_layout_cb.addStretch()
        settings_layout.addLayout(skip_layout_cb)
        
        # Turbo mode checkbox
        turbo_layout_cb = QHBoxLayout()
        self.turbo_cb = QCheckBox(get_text('turbo_mode', self.current_lang))
        self.turbo_cb.setChecked(True)
        self.turbo_cb.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {theme.ORANGE_LIGHT};")
        self.turbo_help = QLabel("ℹ️")
        self.turbo_help.setStyleSheet(theme.info_icon())
        self.turbo_help.setToolTip(get_text('turbo_mode_tooltip', self.current_lang))
        self.turbo_help.setCursor(Qt.WhatsThisCursor)
        turbo_layout_cb.addWidget(self.turbo_cb)
        turbo_layout_cb.addWidget(self.turbo_help)
        turbo_layout_cb.addStretch()
        settings_layout.addLayout(turbo_layout_cb)
        
        # ============ V2.0 Advanced Settings - Collapsible Buttons ============
        v2_container = QVBoxLayout()
        v2_container.setSpacing(5)
        
        # --- Quality Analysis Button & Panel ---
        self.quality_btn = QPushButton(get_text('quality_title', self.current_lang))
        self.quality_btn.setCheckable(True)
        self.quality_btn.setStyleSheet(self._collapsible_btn_style())
        self.quality_btn.clicked.connect(lambda: self._toggle_panel('quality'))
        v2_container.addWidget(self.quality_btn)
        
        self.quality_panel = QualitySettingsPanel(self.current_lang)
        self.quality_panel.setVisible(False)
        v2_container.addWidget(self.quality_panel)
        
        # --- Auto Captioning Button & Panel ---
        self.caption_btn = QPushButton(get_text('caption_title', self.current_lang))
        self.caption_btn.setCheckable(True)
        self.caption_btn.setStyleSheet(self._collapsible_btn_style())
        self.caption_btn.clicked.connect(lambda: self._toggle_panel('caption'))
        v2_container.addWidget(self.caption_btn)
        
        self.caption_panel = CaptioningSettingsPanel(self.current_lang)
        self.caption_panel.setVisible(False)
        v2_container.addWidget(self.caption_panel)
        
        # --- Tag Settings Button & Panel ---
        self.tags_btn = QPushButton(get_text('tag_settings_title', self.current_lang))
        self.tags_btn.setCheckable(True)
        self.tags_btn.setStyleSheet(self._collapsible_btn_style())
        self.tags_btn.clicked.connect(lambda: self._toggle_panel('tags'))
        v2_container.addWidget(self.tags_btn)
        
        self.tags_panel = TagSettingsPanel(self.current_lang)
        self.tags_panel.setVisible(False)
        v2_container.addWidget(self.tags_panel)
        
        settings_layout.addLayout(v2_container)
        # ============ End V2.0 Advanced Settings ============

        # Minimum padding
        padding_layout = QHBoxLayout()
        self.padding_label = QLabel(get_text('min_padding', self.current_lang))
        self.padding_label.setStyleSheet(theme.label_default())
        self.padding_help = QLabel("ℹ️")
        self.padding_help.setStyleSheet(theme.info_icon())
        self.padding_help.setToolTip(get_text('min_padding_tooltip', self.current_lang))
        self.padding_help.setCursor(Qt.WhatsThisCursor)
        self.padding_spinbox = QSpinBox()
        self.padding_spinbox.setMinimum(100)
        self.padding_spinbox.setMaximum(1000)
        self.padding_spinbox.setValue(500)
        self.padding_spinbox.setSingleStep(50)
        self.padding_spinbox.setStyleSheet(theme.spinbox())
        padding_layout.addWidget(self.padding_label)
        padding_layout.addWidget(self.padding_help)
        padding_layout.addWidget(self.padding_spinbox)
        padding_layout.addStretch()
        settings_layout.addLayout(padding_layout)
        
        layout.addWidget(self.settings_group)
        
        # Process and Stop buttons
        buttons_layout = QHBoxLayout()
        
        self.process_btn = QPushButton(get_text('start_btn', self.current_lang))
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet(theme.btn_primary())
        self.process_btn.clicked.connect(self.start_processing)
        
        self.stop_btn = QPushButton(get_text('stop_btn', self.current_lang))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.stop_btn.clicked.connect(self.stop_processing)
        
        # Open output folder button
        self.open_output_btn = QPushButton(get_text('open_output_btn', self.current_lang))
        self.open_output_btn.setStyleSheet(theme.btn_secondary())
        self.open_output_btn.clicked.connect(self.open_output_folder)
        
        buttons_layout.addWidget(self.process_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.open_output_btn)
        layout.addLayout(buttons_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(theme.progress_bar())
        layout.addWidget(self.progress_bar)
        
        # Status/Log area
        self.log_label = QLabel(get_text('log_title', self.current_lang))
        self.log_label.setStyleSheet(f"font-weight: bold; margin-top: 10px; color: {theme.TEXT_PRIMARY};")
        layout.addWidget(self.log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet(theme.log_area())
        layout.addWidget(self.log_text)
        
        self.log(get_text('log_started', self.current_lang))
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def browse_video(self):
        """Browse for video file(s) - supports multiple selection"""
        file_paths, _ = QFileDialog.getOpenFileNames(  # Changed to getOpenFileNames for multiple
            self,
            "Select Video File(s)",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v);;All Files (*)"
        )
        
        if file_paths:
            self.on_files_dropped(file_paths)
    
    def open_output_folder(self):
        """Open output folder in file explorer"""
        output_path = Path("output")
        
        # Create output folder if it doesn't exist
        output_path.mkdir(exist_ok=True)
        
        # Open folder in file explorer
        try:
            if sys.platform == "win32":
                os.startfile(str(output_path))
            elif sys.platform == "darwin":  # macOS
                subprocess.call(["open", str(output_path)])
            else:  # Linux
                subprocess.call(["xdg-open", str(output_path)])
            
            self.log("📂 Output folder opened")
        except Exception as e:
            self.log(f"❌ Failed to open output folder: {str(e)}")
    
    def on_files_dropped(self, file_paths: List[str]):
        """Handle file selection - supports multiple files"""
        self.video_paths = file_paths
        self.process_btn.setEnabled(True)
        
        if len(file_paths) == 1:
            self.log(get_text('log_loaded', self.current_lang).format(Path(file_paths[0]).name))
            self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(1))
        else:
            names = ', '.join([Path(f).name for f in file_paths[:3]])
            if len(file_paths) > 3:
                names += f' ... (+{len(file_paths)-3} more)'
            self.log(get_text('log_loaded', self.current_lang).format(names))
            self.log(get_text('log_batch_mode', self.current_lang).format(len(file_paths)))
            self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(len(file_paths)))
    
    def _cleanup_processing_thread(self):
        """Disconnect signals and schedule deletion of the current processing thread"""
        if self.processing_thread is not None:
            try:
                self.processing_thread.progress_update.disconnect(self.on_progress)
                self.processing_thread.log_message.disconnect(self.log)
                self.processing_thread.finished.disconnect(self.on_finished)
                self.processing_thread.error.disconnect(self.on_error)
            except (TypeError, RuntimeError):
                pass  # Already disconnected
            # Only deleteLater if thread is not running to avoid crash
            if not self.processing_thread.isRunning():
                self.processing_thread.deleteLater()
            else:
                # Thread still alive - connect finished to deleteLater so it cleans up when done
                try:
                    self.processing_thread.finished.connect(
                        self.processing_thread.deleteLater)
                except (TypeError, RuntimeError):
                    pass
            self.processing_thread = None

    def stop_processing(self):
        """Stop video processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.log(get_text('log_stopping', self.current_lang))
            self.processing_thread.stop()
            self.stop_btn.setEnabled(False)
            # Don't block UI — the thread's finished signal will trigger cleanup
    
    def start_processing(self):
        """Start video processing with v2.0 features"""
        if not self.video_paths:
            self.log(get_text('log_no_file', self.current_lang))
            return

        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.drop_zone.setEnabled(False)

        # Get settings
        frame_interval = self.interval_slider.value()
        skip_text = self.skip_subtitle_cb.isChecked()
        confidence = self.conf_spinbox.value() / 100.0
        min_padding = self.padding_spinbox.value()
        aspect_ratio = self.ratio_combo.currentText()
        use_ensemble = self.ensemble_cb.isChecked()
        use_turbo = self.turbo_cb.isChecked()

        # V2.0 Settings
        quality_settings = self.quality_panel.get_settings()
        caption_settings = self.caption_panel.get_settings()
        tag_settings = self.tags_panel.get_settings()

        # Ensemble model selection
        models_to_use = []
        voting_threshold = 2
        if use_ensemble:
            if self.yolo_cb.isChecked():
                models_to_use.append('yolo')
            if self.detr_cb.isChecked():
                models_to_use.append('detr')
            if self.fasterrcnn_cb.isChecked():
                models_to_use.append('fasterrcnn')

            if not models_to_use:
                self.log(get_text('log_error_model', self.current_lang))
                self.process_btn.setEnabled(True)
                self.drop_zone.setEnabled(True)
                return
            voting_threshold = self.voting_spinbox.value()

        # Log settings
        self.log(get_text('log_settings', self.current_lang).format(frame_interval, aspect_ratio, confidence))
        if use_ensemble:
            self.log(get_text('log_ensemble_on', self.current_lang))
        if use_turbo:
            self.log(get_text('log_turbo', self.current_lang))
        self.log(get_text('log_init', self.current_lang))

        # Build config dict - all heavy model loading happens in the thread
        config = {
            'video_paths': self.video_paths,
            'frame_interval': frame_interval,
            'skip_text': skip_text,
            'confidence': confidence,
            'min_padding': min_padding,
            'aspect_ratio': aspect_ratio,
            'use_ensemble': use_ensemble,
            'models_to_use': models_to_use,
            'voting_threshold': voting_threshold,
            'use_turbo': use_turbo,
            'quality_settings': quality_settings,
            'caption_settings': caption_settings,
            'tag_settings': tag_settings,
        }

        # Clean up any leftover thread before starting a new one
        self._cleanup_processing_thread()

        # Start processing thread (models load in background)
        self.processing_thread = ProcessingThread(config)
        self.processing_thread.progress_update.connect(self.on_progress)
        self.processing_thread.log_message.connect(self.log)
        self.processing_thread.finished.connect(self.on_finished)
        self.processing_thread.error.connect(self.on_error)
        self.processing_thread.start()
    
    def on_progress(self, progress: float, stats: dict):
        """Update progress"""
        self.progress_bar.setValue(int(progress))
        
        # Update log with stats
        if stats['processed_frames'] % 10 == 0:  # Update every 10 frames
            self.log(get_text('log_progress', self.current_lang).format(
                progress, stats['saved_frames'], stats['person_frames'],
                stats['animal_frames'], stats['object_frames']
            ))
    
    def on_finished(self, stats: dict):
        """Processing finished"""
        # If stopped by user, thread was already cleaned up in stop_processing
        if stats.get('stopped'):
            return
        # Guard against stale signal from an old thread
        if self.processing_thread is None:
            return

        self.progress_bar.setValue(100)
        self.log("\n" + "="*50)
        self.log(get_text('log_complete', self.current_lang))

        # Check if this is batch processing (overall_stats) or single video
        if 'total_videos' in stats:
            # Batch processing - show overall summary
            self.log(f"📹 Processed videos: {stats['processed_videos']}/{stats['total_videos']}")
            self.log(get_text('log_total', self.current_lang).format(stats['total_frames_saved']))
            self.log(f"⏱️  Total time: {stats.get('total_time', 0):.1f}s")
            # V2.0: Show per-video quality/captioning stats
            for vs in stats.get('videos_stats', []):
                vname = vs.get('video_name', '?')
                vst = vs.get('stats', {})
                skipped_q = vst.get('skipped_quality', 0)
                captioned = vst.get('captioned_frames', 0)
                overlay = vst.get('overlay_crops', 0)
                extra = []
                if skipped_q > 0:
                    extra.append(f"quality_skip={skipped_q}")
                if captioned > 0:
                    extra.append(f"captioned={captioned}")
                if overlay > 0:
                    extra.append(f"overlay_crops={overlay}")
                if extra:
                    self.log(f"   {vname}: {', '.join(extra)}")
        else:
            # Single video - show detailed stats
            self.log(get_text('log_total', self.current_lang).format(stats['saved_frames']))
            self.log(get_text('log_persons', self.current_lang).format(stats['person_frames']))
            self.log(get_text('log_animals', self.current_lang).format(stats['animal_frames']))
            self.log(get_text('log_objects', self.current_lang).format(stats['object_frames']))
            self.log(get_text('log_skipped_text', self.current_lang).format(stats['skipped_text']))
            self.log(get_text('log_skipped_none', self.current_lang).format(stats['skipped_no_detection']))
            # V2.0 stats
            if stats.get('skipped_quality', 0) > 0:
                self.log(f"🔍 Skipped (quality): {stats['skipped_quality']}")
            captioned = stats.get('captioned_frames', 0)
            saved = stats.get('saved_frames', 0)
            self.log(f"📝 Captioned: {captioned}/{saved} frames")
            if stats.get('overlay_crops', 0) > 0:
                self.log(f"🛡️ Overlay-aware crops: {stats['overlay_crops']}")
        
        self.log("="*50)
        
        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.drop_zone.setEnabled(True)
        self._cleanup_processing_thread()

    def on_error(self, error_msg: str):
        """Handle error"""
        self.log(get_text('log_error', self.current_lang).format(error_msg))
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.drop_zone.setEnabled(True)
        self._cleanup_processing_thread()

    def closeEvent(self, event):
        """Ensure all threads stop when window is closed"""
        # Stop processing thread
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.safe_wait(5000)

        # Stop captioning thread (from captioning page)
        if hasattr(self, 'captioning_page'):
            cp = self.captioning_page
            if hasattr(cp, 'captioning_thread') and cp.captioning_thread:
                if cp.captioning_thread.isRunning():
                    cp.captioning_thread.stop()
                    cp.captioning_thread.wait(5000)
            # Cleanup captioner models
            if hasattr(cp, '_cleanup_captioner'):
                cp._cleanup_captioner()

        event.accept()
        # Force process exit so no threads linger
        import os
        os._exit(0)
    
    def change_language(self, index):
        """Change UI language"""
        self.current_lang = 'en' if index == 0 else 'tr'
        self.update_ui_texts()
    
    def update_ui_texts(self):
        """Update all UI texts with current language"""
        self.setWindowTitle(get_text('app_title', self.current_lang))
        
        # Page navigation buttons
        self.page_video_btn.setText(get_text('page_video_processing', self.current_lang))
        self.page_caption_btn.setText(get_text('page_captioning', self.current_lang))
        self.page_char_sort_btn.setText(get_text('page_character_sort', self.current_lang))

        # Update captioning page language
        if hasattr(self, 'captioning_page'):
            self.captioning_page.update_language(self.current_lang)

        # Update character sort page language
        if hasattr(self, 'char_sort_page'):
            self.char_sort_page.update_language(self.current_lang)
        
        # Video page elements
        self.title_label.setText(get_text('title', self.current_lang))
        self.subtitle_label.setText(get_text('subtitle', self.current_lang))
        self.browse_btn.setText(get_text('browse_btn', self.current_lang))
        self.settings_group.setTitle(get_text('settings_title', self.current_lang))
        self.interval_label.setText(get_text('frame_interval', self.current_lang))
        self.interval_help.setToolTip(get_text('frame_interval_tooltip', self.current_lang))
        self.ratio_label.setText(get_text('output_format', self.current_lang))
        self.ratio_help.setToolTip(get_text('output_format_tooltip', self.current_lang))
        self.conf_label.setText(get_text('confidence', self.current_lang))
        self.conf_help.setToolTip(get_text('confidence_tooltip', self.current_lang))
        self.ensemble_cb.setText(get_text('ensemble_mode', self.current_lang))
        self.ensemble_help.setToolTip(get_text('ensemble_mode_tooltip', self.current_lang))
        self.ensemble_group.setTitle(get_text('ensemble_settings', self.current_lang))
        self.models_label.setText(get_text('active_models', self.current_lang))
        self.voting_label.setText(get_text('voting_threshold', self.current_lang))
        self.voting_help.setToolTip(get_text('voting_threshold_tooltip', self.current_lang))
        self.skip_subtitle_cb.setText(get_text('skip_subtitle', self.current_lang))
        self.skip_help.setToolTip(get_text('skip_subtitle_tooltip', self.current_lang))
        self.turbo_cb.setText(get_text('turbo_mode', self.current_lang))
        self.turbo_help.setToolTip(get_text('turbo_mode_tooltip', self.current_lang))
        self.padding_label.setText(get_text('min_padding', self.current_lang))
        self.padding_help.setToolTip(get_text('min_padding_tooltip', self.current_lang))
        self.process_btn.setText(get_text('start_btn', self.current_lang))
        self.stop_btn.setText(get_text('stop_btn', self.current_lang))
        self.open_output_btn.setText(get_text('open_output_btn', self.current_lang))
        self.log_label.setText(get_text('log_title', self.current_lang))
        self.update_drop_zone_text()
        
        # Update v2 collapsible buttons (without emoji duplicates)
        if hasattr(self, 'quality_btn'):
            self.quality_btn.setText(get_text('quality_title', self.current_lang))
        if hasattr(self, 'caption_btn'):
            self.caption_btn.setText(get_text('caption_title', self.current_lang))
        if hasattr(self, 'tags_btn'):
            self.tags_btn.setText(get_text('tag_settings_title', self.current_lang))
        
        # Update v2 panels
        if hasattr(self, 'quality_panel'):
            self.quality_panel.update_language(self.current_lang)
        if hasattr(self, 'caption_panel'):
            self.caption_panel.update_language(self.current_lang)
        if hasattr(self, 'tags_panel'):
            self.tags_panel.update_language(self.current_lang)
    
    def update_drop_zone_text(self):
        """Update drop zone text"""
        if not self.video_paths:
            self.drop_zone.setText(get_text('drop_zone', self.current_lang))
        elif len(self.video_paths) == 1:
            self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(1))
        else:
            self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(len(self.video_paths)))


def create_app():
    """Create and return the application"""
    # Enable HiDPI scaling BEFORE creating QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)

    # Set taskbar icon at QApplication level (before window creation)
    _assets = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets'))
    for _name in ('icon.png', 'icon.ico'):
        _p = os.path.join(_assets, _name)
        if os.path.exists(_p):
            app.setWindowIcon(QIcon(_p))
            break

    # Set application style
    app.setStyle('Fusion')

    # Create main window
    window = VideoSmartCropperUI()
    window.show()

    # WM_SETICON is now handled in showEvent() for reliability

    return app, window
