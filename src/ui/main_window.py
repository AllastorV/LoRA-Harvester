"""
Modern UI Module for LoRA-Harvester v4.0
AI-Powered Dataset Collection Tool with PyQt5 interface
Sidebar-based navigation with live system monitor.
"""

import sys
import os
import subprocess
import threading
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QComboBox, QCheckBox, QProgressBar, QFileDialog,
                             QTextEdit, QGroupBox, QSpinBox,
                             QScrollArea, QStackedWidget, QFrame, QDesktopWidget,
                             QSizePolicy, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent
from typing import List
from src.ui.translations import get_text
from src.ui import theme
from src.ui.animations import (
    animate_page_switch, fade_in, smooth_expand,
    HoverLift, PulseEffect, NavIndicator, progress_smooth,
    StatusDot, count_up, shake_widget, RippleButton, scale_pop,
    ToastNotification, LoadingSpinner, SkeletonShimmer,
    stagger_fade_in, glitch_effect, badge_bounce,
    ProgressGlow, ShimmerLabel, SidebarPulse,
)
from src.ui.advanced_settings import (
    QualitySettingsPanel,
    CaptioningSettingsPanel,
    TagSettingsPanel
)
from src.ui.caption_studio_page import CaptionStudioPage
from src.ui.character_sort_page import CharacterSortPage
from src.ui.tag_frequency_page import TagFrequencyPage
from src.ui.resource_settings import ResourceSettingsDrawer


class ProcessingThread(QThread):
    """Background thread for video processing - all heavy work runs here"""

    progress_update = pyqtSignal(float, dict)
    log_message = pyqtSignal(str)
    processing_finished = pyqtSignal(dict)  # renamed to avoid shadowing QThread.finished
    error = pyqtSignal(str)
    # Emitted right after a cropped frame is written to disk. Carries
    # the absolute path as a string so the UI can load the thumbnail.
    frame_saved = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_running = True
        self._finished_emitted = False
        self.processor = None
        # One-shot "skip current video" signal. Set by ``skip()`` from the
        # UI thread, read by the processor's inner loops, and cleared by
        # the processor itself at the start of each new video.
        self._skip_event = threading.Event()
        # Pause/Resume. Inverted semantics for convenience: set() means
        # "running", clear() means "paused". Starts set (= running).
        self._pause_event = threading.Event()
        self._pause_event.set()

    def run(self):
        """Initialize models and run video processing"""
        # Outcome captured here so finally can emit AFTER _cleanup().
        # This is critical: main thread must not try to delete this
        # QThread object while _cleanup() (GPU model release) is still
        # running — that causes a crash.  By emitting signals only after
        # _cleanup() returns we guarantee the thread's heavy work is done
        # before the main thread touches the thread object.
        _err = None
        _stats = None
        _stopped = False

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
                _stopped = True
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
                _stopped = True
                return

            # Captioner
            captioner = None
            florence2 = None
            cs = cfg['caption_settings']
            caption_mode = cs.get('mode', 'tags_only')
            use_wd14 = caption_mode in ('tags_only', 'combined')
            use_florence2 = caption_mode in ('florence2', 'combined')
            if cs['enabled']:
                try:
                    from src.core.advanced_captioner import AdvancedCaptioner, TagSettings
                    ts = cfg['tag_settings']
                    # Prefer preset confidence from captioning panel, fall back to tag panel
                    min_conf = cs.get('min_confidence') or ts['min_confidence']
                    tag_cfg = TagSettings(
                        trigger_word=ts['trigger_word'] or "",
                        max_tags=ts['max_tags'],
                        min_confidence=min_conf,
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
                    captioner = AdvancedCaptioner(
                        enable_wd14=cs['wd14_enabled'] and use_wd14,
                        wd14_model=cs['wd14_model'],
                        tag_settings=tag_cfg,
                    )
                except Exception as e:
                    self.log_message.emit(f"Captioner init error: {e}")
                    captioner = None

                if use_florence2:
                    try:
                        from src.core.florence2_captioner import Florence2Captioner
                        florence2 = Florence2Captioner(
                            model_type=cs.get('florence2_model', 'florence-2-base'))
                        self.log_message.emit("✅ Florence-2 initialized")
                    except Exception as e:
                        self.log_message.emit(f"❌ Florence-2 init error: {e}")
                        florence2 = None

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
                        _stopped = True
                        return

                    # Log final captioning status
                    wd14_ok = captioner.wd14 and captioner.enable_wd14
                    f2_ok = florence2 is not None
                    self.log_message.emit(
                        f"📝 Captioning: mode={caption_mode} "
                        f"WD14={'ON' if wd14_ok else 'OFF'} "
                        f"Florence2={'ON' if f2_ok else 'OFF'}"
                    )
                    if caption_mode == 'tags_only' and not wd14_ok:
                        self.log_message.emit("⚠️ WD14 disabled - no tags will be generated!")
                        captioner = None
                    elif caption_mode == 'florence2' and not f2_ok:
                        self.log_message.emit("⚠️ Florence-2 disabled - falling back to tags only")
                        caption_mode = 'tags_only'
            else:
                self.log_message.emit("📝 Auto-captioning: disabled (enable in Captioning settings)")

            if not self._is_running:
                _stopped = True
                return

            # Create processor — pull batch_size and jpeg_quality from
            # the resource settings drawer instead of hard-coding them.
            res = cfg.get('resource_settings', {})
            self.processor = UnifiedVideoProcessor(
                video_paths=cfg['video_paths'],
                output_dir="output",
                detector=detector,
                text_detector=text_detector,
                cropper=cropper,
                use_turbo=cfg['use_turbo'],
                batch_size=res.get('batch_size', 4),
                quality_analyzer=quality_analyzer,
                captioner=captioner,
                caption_mode=caption_mode,
                florence2=florence2,
                florence2_task=cs.get('florence2_task', '<DETAILED_CAPTION>'),
                log_callback=lambda msg: self.log_message.emit(msg),
                jpeg_quality=res.get('jpeg_quality', 95),
            )

            self.log_message.emit("All models loaded, processing started...")

            # Process
            result = self.processor.process_all_videos(
                frame_interval=cfg['frame_interval'],
                skip_text=cfg['skip_text'],
                progress_callback=self.progress_callback,
                stop_callback=self.should_stop,
                skip_event=self._skip_event,
                pause_event=self._pause_event,
                start_skip_seconds=cfg.get('start_skip_seconds', 0.0),
                end_skip_seconds=cfg.get('end_skip_seconds', 0.0),
                frame_saved_callback=self._on_frame_saved,
            )

            if not self._is_running:
                _stopped = True
            else:
                _stats = result

        except Exception as e:
            _err = e

        finally:
            # ALWAYS cleanup first — GPU/model release happens here.
            # Only emit signals after cleanup so the main thread never
            # calls deleteLater() while this thread is still doing work.
            self._cleanup()

            if not self._finished_emitted:
                self._finished_emitted = True
                if _err is not None:
                    self.error.emit(str(_err))
                elif _stopped or not self._is_running:
                    self.processing_finished.emit({'stopped': True, 'total_frames_saved': 0})
                elif _stats is not None:
                    self.processing_finished.emit(_stats)
                else:
                    # Stopped during model loading (early return path)
                    self.processing_finished.emit({'stopped': True, 'total_frames_saved': 0})

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
                if hasattr(self.processor, 'florence2') and self.processor.florence2:
                    if hasattr(self.processor.florence2, 'cleanup'):
                        self.processor.florence2.cleanup()
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

    def _on_frame_saved(self, path: str):
        """Called from the processor thread. Emit a signal so the UI can
        load the thumbnail on the main thread."""
        if self._is_running:
            self.frame_saved.emit(path)

    def stop(self):
        """Stop processing gracefully"""
        self._is_running = False
        # Wake up any loop that's blocked on pause or skip.
        self._pause_event.set()
        self._skip_event.set()

    def toggle_pause(self) -> bool:
        """Toggle pause/resume. Returns True if now paused."""
        if self._pause_event.is_set():
            self._pause_event.clear()
            return True
        else:
            self._pause_event.set()
            return False

    def skip_current_video(self):
        """
        Abandon the currently processing video and continue with the next
        one in the batch. Idempotent — setting the event twice has no
        effect. The processor clears it before each new video starts.
        """
        self._skip_event.set()

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
        self._pulse = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(theme.drop_zone_active())
            if self._pulse is None:
                self._pulse = PulseEffect(self, min_opacity=0.70, duration=700)
                self._pulse.start()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave"""
        self.setStyleSheet(theme.drop_zone_default())
        if self._pulse:
            self._pulse.stop()
            self._pulse = None
    
    _VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

    def dropEvent(self, event: QDropEvent):
        """Handle drop - supports video files, folders, and .txt list files"""
        self.setStyleSheet(theme.drop_zone_default())
        if self._pulse:
            self._pulse.stop()
            self._pulse = None

        files = [u.toLocalFile() for u in event.mimeData().urls()]

        # Accept: video files, directories (recursive walk), and .txt list files.
        # The actual resolution (folder scanning, .txt parsing) happens in
        # on_files_dropped — here we just gate-keep obviously wrong drops.
        accepted = [
            f for f in files
            if (Path(f).is_dir()
                or Path(f).suffix.lower() in self._VIDEO_EXTENSIONS
                or Path(f).suffix.lower() == '.txt')
        ]

        if accepted:
            self.files_dropped.emit(accepted)
            dirs  = [f for f in accepted if Path(f).is_dir()]
            txts  = [f for f in accepted if Path(f).suffix.lower() == '.txt']
            vids  = [f for f in accepted if Path(f).suffix.lower() in self._VIDEO_EXTENSIONS]
            parts = []
            if vids:
                parts.append(f"{len(vids)} video(s)")
            if dirs:
                parts.append(f"{len(dirs)} folder(s)")
            if txts:
                parts.append(f"{len(txts)} list(s)")
            self.setText(f"✅ {', '.join(parts)} dropped")
        else:
            self.setText("❌ Invalid file type. Drop video file(s), folders, or .txt list.")


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
        """Initialize UI components — sidebar + topbar + main content."""
        self.setWindowTitle(get_text('app_title', self.current_lang))
        _icon_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets'))
        for _icon_name in ('icon.ico', 'icon.png'):
            _icon_path = os.path.join(_icon_dir, _icon_name)
            if os.path.exists(_icon_path):
                self.setWindowIcon(QIcon(_icon_path))
                break
        if sys.platform == 'win32':
            self._enable_dark_title_bar(int(self.winId()))

        screen = QDesktopWidget().availableGeometry()
        win_w = min(1200, int(screen.width() * 0.75))
        win_h = min(900, int(screen.height() * 0.88))
        self.resize(win_w, win_h)
        self.move((screen.width() - win_w) // 2,
                  (screen.height() - win_h) // 2)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        central_widget.setLayout(root_layout)

        # ═══════════ LEFT SIDEBAR (240px) ═══════════
        self._sidebar = QFrame()
        self._sidebar.setFixedWidth(240)
        self._sidebar.setStyleSheet(theme.sidebar_frame())
        sidebar_lay = QVBoxLayout(self._sidebar)
        sidebar_lay.setContentsMargins(12, 14, 12, 12)
        sidebar_lay.setSpacing(4)

        # Brand header — icon + label
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(8)

        # Icon from assets/icon.png
        _icon_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'icon.png'
        ))
        if not os.path.exists(_icon_path):
            _icon_path = _icon_path.replace('.png', '.ico')
        self._brand_icon_lbl = QLabel()
        if os.path.exists(_icon_path):
            from PyQt5.QtGui import QPixmap
            pix = QPixmap(_icon_path).scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._brand_icon_lbl.setPixmap(pix)
        else:
            self._brand_icon_lbl.setText("🌾")
            self._brand_icon_lbl.setStyleSheet("font-size: 20px;")
        self._brand_icon_lbl.setFixedSize(32, 32)
        self._brand_icon_lbl.setStyleSheet("background: transparent; border: none;")
        brand_row.addWidget(self._brand_icon_lbl)

        brand_text_col = QVBoxLayout()
        brand_text_col.setSpacing(0)
        brand_text_col.setContentsMargins(0, 0, 0, 0)

        # Brand — animated shimmer highlight travelling across the text
        self._brand_label = ShimmerLabel(
            "LoRA-Harvester",
            base_color=theme.TEXT_PRIMARY,
            highlight_color=theme.ORANGE_LIGHT,
        )
        self._brand_label.setStyleSheet(theme.sidebar_brand())
        brand_text_col.addWidget(self._brand_label)

        self._brand_sub = QLabel("v4.0 · Dataset Studio")
        self._brand_sub.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent;"
        )
        brand_text_col.addWidget(self._brand_sub)
        brand_row.addLayout(brand_text_col)
        brand_row.addStretch()

        brand_widget = QWidget()
        brand_widget.setStyleSheet("background: transparent;")
        brand_widget.setLayout(brand_row)
        brand_widget.setContentsMargins(0, 0, 0, 8)
        sidebar_lay.addWidget(brand_widget)

        # Section: WORKSPACE
        ws_label = QLabel("WORKSPACE")
        ws_label.setStyleSheet(theme.sidebar_section_label())
        sidebar_lay.addWidget(ws_label)
        sidebar_lay.addSpacing(4)
        self._sidebar_section_labels = [ws_label]

        self.page_video_btn = QPushButton("  🎬  Video Harvester")
        self.page_caption_studio_btn = QPushButton("  🏷  Etiketleme")
        self.page_char_sort_btn = QPushButton("  👥  Karakterler")

        self._nav_buttons = [
            self.page_video_btn,
            self.page_caption_studio_btn,
            self.page_char_sort_btn,
        ]
        for i, btn in enumerate(self._nav_buttons):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(self._page_btn_style(i == 0))
            btn.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            sidebar_lay.addWidget(btn)

        # Nav badge — count of queued files on the Video Harvester button
        self._video_badge = QLabel("0", self.page_video_btn)
        self._video_badge.setFixedSize(20, 20)
        self._video_badge.setAlignment(Qt.AlignCenter)
        self._video_badge.setStyleSheet(
            f"background-color: {theme.ORANGE}; color: white; "
            f"border-radius: 10px; font-size: {theme.fs(10)}; font-weight: 700;"
        )
        self._video_badge.hide()

        sidebar_lay.addSpacing(10)

        # Section: KÜTÜPHANE
        lib_label = QLabel("KÜTÜPHANE")
        lib_label.setStyleSheet(theme.sidebar_section_label())
        sidebar_lay.addWidget(lib_label)
        sidebar_lay.addSpacing(4)
        self._sidebar_section_labels.append(lib_label)

        self.page_tag_freq_btn = QPushButton("  📊  Etiket Sözlüğü")
        self.page_tag_freq_btn.setCursor(Qt.PointingHandCursor)
        self.page_tag_freq_btn.setStyleSheet(self._page_btn_style(False))
        self.page_tag_freq_btn.clicked.connect(lambda: self.switch_page(3))
        sidebar_lay.addWidget(self.page_tag_freq_btn)
        self._nav_buttons.append(self.page_tag_freq_btn)

        # Sliding underline indicator for the active nav button
        self._nav_indicator = NavIndicator(self._sidebar, color=theme.get_accent(), height=3)
        QTimer.singleShot(0, lambda: self._nav_indicator.move_under(self._nav_buttons[0]))

        sidebar_lay.addStretch()

        # Settings button (footer)
        self.page_settings_btn = QPushButton("  ⚙️  Settings")
        self.page_settings_btn.setCursor(Qt.PointingHandCursor)
        self.page_settings_btn.setStyleSheet(self._page_btn_style(False))
        self.page_settings_btn.clicked.connect(lambda: self.switch_page(4))
        sidebar_lay.addWidget(self.page_settings_btn)
        self._nav_buttons.append(self.page_settings_btn)

        root_layout.addWidget(self._sidebar)

        # ═══════════ RIGHT AREA (topbar + content) ═══════════
        right_area = QVBoxLayout()
        right_area.setContentsMargins(0, 0, 0, 0)
        right_area.setSpacing(0)

        # ── TOPBAR (60px) ──
        self._topbar = QFrame()
        self._topbar.setFixedHeight(60)
        self._topbar.setStyleSheet(theme.topbar_frame())
        topbar_lay = QHBoxLayout(self._topbar)
        topbar_lay.setContentsMargins(16, 0, 16, 0)
        topbar_lay.setSpacing(12)

        # System monitor (live CPU/RAM/GPU/VRAM pills)
        from src.ui.resource_settings import SystemMonitorBar
        self._topbar_monitor = SystemMonitorBar(self.current_lang, self._topbar)
        topbar_lay.addWidget(self._topbar_monitor)

        topbar_lay.addStretch()

        # ── Status indicator (dot + label) ──
        self._status_dot = StatusDot(self._topbar, size=10)
        topbar_lay.addWidget(self._status_dot)
        self._status_label = QLabel(get_text('status_idle', self.current_lang))
        self._status_label.setStyleSheet(
            f"background: transparent; color: {theme.TEXT_SECONDARY}; "
            f"border: none; padding: 0 8px 0 4px; font-size: {theme.fs(11)}; "
            f"font-weight: 600;"
        )
        topbar_lay.addWidget(self._status_label)

        # GPU status badge
        self._gpu_badge = QLabel()
        self._gpu_badge.setStyleSheet(
            f"background: transparent; color: {theme.TEXT_SECONDARY}; "
            f"border: none; padding: 0 4px; font-size: {theme.fs(11)}; "
            f"font-family: {theme.FONT_MONO}; font-weight: 600;"
        )
        self._update_gpu_badge()
        topbar_lay.addWidget(self._gpu_badge)

        # Settings button (opens resource drawer)
        self.res_settings_btn = QPushButton("⚙")
        self.res_settings_btn.setToolTip(get_text('res_menu_tooltip', self.current_lang))
        self.res_settings_btn.setStyleSheet(theme.btn_icon_square())
        self.res_settings_btn.setCursor(Qt.PointingHandCursor)
        self.res_settings_btn.clicked.connect(self._toggle_resource_drawer)
        topbar_lay.addWidget(self.res_settings_btn)

        right_area.addWidget(self._topbar)
        self._topbar_monitor.start()

        # ── PAGE STACK ──
        self.page_stack = QStackedWidget()

        self.video_page = QWidget()
        self.setup_video_page()
        video_scroll = QScrollArea()
        video_scroll.setWidgetResizable(True)
        video_scroll.setWidget(self.video_page)
        video_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(video_scroll)

        self.caption_studio_page = CaptionStudioPage(self.current_lang)
        studio_scroll = QScrollArea()
        studio_scroll.setWidgetResizable(True)
        studio_scroll.setWidget(self.caption_studio_page)
        studio_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(studio_scroll)

        self.char_sort_page = CharacterSortPage(self.current_lang)
        char_scroll = QScrollArea()
        char_scroll.setWidgetResizable(True)
        char_scroll.setWidget(self.char_sort_page)
        char_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(char_scroll)

        self.tag_freq_page = TagFrequencyPage(self.current_lang)
        tf_scroll = QScrollArea()
        tf_scroll.setWidgetResizable(True)
        tf_scroll.setWidget(self.tag_freq_page)
        tf_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(tf_scroll)

        # Settings page (index 4)
        self.settings_page = self._build_settings_page()
        self.page_stack.addWidget(self.settings_page)

        right_area.addWidget(self.page_stack, stretch=1)
        right_container = QWidget()
        right_container.setLayout(right_area)
        root_layout.addWidget(right_container, stretch=1)

        # ═══════════ RESOURCE SETTINGS DRAWER ═══════════
        self._resource_drawer = ResourceSettingsDrawer(
            lang=self.current_lang, parent=central_widget,
        )
        self._resource_drawer.settings_changed.connect(self._on_resource_settings_changed)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['English', 'Türkçe'])
        self.lang_combo.setCurrentIndex(0)
        self.lang_combo.setStyleSheet(theme.combo())
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        self._resource_drawer.embed_lang_combo(self.lang_combo)
        self._resource_cfg = self._resource_drawer.get_settings()

        self.setStyleSheet(theme.global_stylesheet())

        # Apply language-aware texts now that all widgets exist. Without
        # this, init_ui leaves hard-coded Turkish strings visible until
        # the user manually changes the language.
        self.update_ui_texts()
    
    def _update_gpu_badge(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_properties(0).name
                short = name.replace("NVIDIA ", "").replace("GeForce ", "")
                self._gpu_badge.setText(f"● {short}")
                self._gpu_badge.setStyleSheet(
                    f"background: transparent; color: {theme.GREEN}; "
                    f"border: none; padding: 0 4px; font-size: {theme.fs(11)}; "
                    f"font-family: {theme.FONT_MONO}; font-weight: 600;"
                )
                return
        except Exception:
            pass
        self._gpu_badge.setText("● CPU")
        self._gpu_badge.setStyleSheet(
            f"background: transparent; color: {theme.TEXT_MUTED}; "
            f"border: none; padding: 0 4px; font-size: {theme.fs(11)}; "
            f"font-family: {theme.FONT_MONO}; font-weight: 600;"
        )

    def _update_video_badge(self, count: int):
        """Update the queue-count badge on the Video Harvester nav button."""
        if not hasattr(self, '_video_badge'):
            return
        if count <= 0:
            self._video_badge.hide()
            return
        self._video_badge.setText(str(count))
        btn = self.page_video_btn
        self._video_badge.move(btn.width() - 26, 6)
        was_hidden = not self._video_badge.isVisible()
        self._video_badge.show()
        self._video_badge.raise_()
        if not was_hidden:
            badge_bounce(self._video_badge)

    def _set_status(self, state: str):
        """Sync status dot + label + sidebar pulse based on processing state."""
        if hasattr(self, '_status_dot'):
            self._status_dot.set_state(state)
        key_map = {
            'idle':       'status_idle',
            'processing': 'status_processing',
            'paused':     'status_paused',
            'done':       'status_done',
            'error':      'status_error',
        }
        if hasattr(self, '_status_label'):
            self._status_label.setText(
                get_text(key_map.get(state, 'status_idle'), self.current_lang)
            )
        if not hasattr(self, '_sidebar_pulse'):
            self._sidebar_pulse = SidebarPulse(
                self.page_video_btn, color=theme.get_accent(),
            )
        if state == 'processing':
            self._sidebar_pulse.start()
        else:
            self._sidebar_pulse.stop()

        # Title shimmer — animated only while actively processing
        if hasattr(self, '_brand_label'):
            if state == 'processing':
                self._brand_label.start_shimmer()
            else:
                self._brand_label.stop_shimmer()

    def _reset_stat(self, key: str):
        """Reset a stat card's value label to '0' without animation."""
        if not hasattr(self, '_stat_cards'):
            return
        card = self._stat_cards.get(key)
        if card and 'value' in card:
            card['value'].setText("0")

    def _bump_stat(self, key: str, new_value: int):
        """Animate a stat card's value up to new_value."""
        if not hasattr(self, '_stat_cards'):
            return
        card = self._stat_cards.get(key)
        if not card or 'value' not in card:
            return
        try:
            cur = int(card['value'].text())
        except (ValueError, TypeError):
            cur = 0
        if new_value == cur:
            return
        count_up(card['value'], cur, new_value)

    def _page_btn_style(self, active: bool) -> str:
        return theme.page_btn_active() if active else theme.page_btn_inactive()

    def _collapsible_btn_style(self, color: str = None) -> str:
        return theme.collapsible_btn()

    def _toggle_resource_drawer(self):
        self._resource_drawer.toggle()

    def _on_resource_settings_changed(self, data: dict):
        """Slot: user clicked Apply in the resource drawer."""
        self._resource_cfg = data
        # Apply theme changes if mode, font scale, or accent color changed
        new_mode = data.get("theme_mode", "dark")
        new_scale = data.get("font_scale", 100) / 100.0
        new_accent = data.get("accent", theme.get_accent())
        mode_changed = new_mode != theme.get_mode()
        scale_changed = abs(new_scale - theme.get_font_scale()) > 0.01
        accent_changed = (new_accent or "").lower() != theme.get_accent().lower()
        if mode_changed or scale_changed or accent_changed:
            theme.set_theme(new_mode, new_scale, accent=new_accent)
            self._refresh_all_styles()
        self.log(get_text('res_apply', self.current_lang))

    def _refresh_all_styles(self):
        """Re-apply all stylesheets after a theme change."""
        self.setStyleSheet(theme.global_stylesheet())

        # Sidebar
        self._sidebar.setStyleSheet(theme.sidebar_frame())
        self._brand_label.setStyleSheet(theme.sidebar_brand())
        self._brand_label.set_colors(theme.TEXT_PRIMARY, theme.ORANGE_LIGHT)
        self._brand_sub.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent; margin-bottom: 14px;"
        )
        for lbl in self._sidebar_section_labels:
            lbl.setStyleSheet(theme.sidebar_section_label())
        current_idx = self.page_stack.currentIndex()
        for i, btn in enumerate(self._nav_buttons):
            btn.setStyleSheet(self._page_btn_style(i == current_idx))
        if hasattr(self, '_nav_indicator'):
            self._nav_indicator.set_color(theme.get_accent())

        # Topbar
        self._topbar.setStyleSheet(theme.topbar_frame())
        self._update_gpu_badge()
        if hasattr(self, '_topbar_monitor'):
            self._topbar_monitor.refresh_styles()
        self.res_settings_btn.setStyleSheet(theme.btn_icon_square())
        self._resource_drawer.refresh_styles()

        # Video page widgets
        self.title_label.setStyleSheet(theme.label_title())
        self.subtitle_label.setStyleSheet(theme.label_muted())
        self.drop_zone.setStyleSheet(theme.drop_zone_default())
        self.browse_btn.setStyleSheet(theme.btn_browse())
        self.settings_group.setStyleSheet(theme.group_box())
        self.interval_slider.setStyleSheet(theme.slider())
        self.interval_value_label.setStyleSheet(theme.label_value())
        self.ratio_combo.setStyleSheet(theme.combo())
        self.conf_spinbox.setStyleSheet(theme.spinbox())
        self.padding_spinbox.setStyleSheet(theme.spinbox())
        self.trim_start_spin.setStyleSheet(theme.spinbox())
        self.trim_end_spin.setStyleSheet(theme.spinbox())
        self.process_btn.setStyleSheet(theme.btn_action_start())
        self.pause_btn.setStyleSheet(theme.btn_action_pause())
        self.skip_btn.setStyleSheet(theme.btn_action_skip())
        self.stop_btn.setStyleSheet(theme.btn_action_stop())
        self.open_output_btn.setStyleSheet(theme.btn_secondary())
        self.progress_bar.setStyleSheet(theme.progress_bar())
        self.log_text.setStyleSheet(theme.log_area())
        self.lang_combo.setStyleSheet(theme.combo())

        # Collapsible buttons & panels
        for btn in (self.quality_btn, self.caption_btn, self.tags_btn):
            btn.setStyleSheet(self._collapsible_btn_style())
        self.ensemble_group.setStyleSheet(theme.panel_group())
        for panel in (self.quality_panel, self.caption_panel):
            if hasattr(panel, 'refresh_styles'):
                panel.refresh_styles()
        if hasattr(self, 'tags_panel') and hasattr(self.tags_panel, 'refresh_styles'):
            self.tags_panel.refresh_styles()

        # Settings page
        if hasattr(self, '_theme_dark_btn'):
            self._update_theme_mode_btns()
        if hasattr(self, '_accent_swatches'):
            self._refresh_accent_swatches()
        if hasattr(self, '_font_scale_slider'):
            self._font_scale_slider.setStyleSheet(theme.slider())
            self._font_scale_val_lbl.setStyleSheet(
                f"color: {theme.ORANGE}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(11)}; background: transparent; border: none; min-width: 36px;"
            )

        # Sub-pages
        for attr in (
            'caption_studio_page', 'char_sort_page',
            'tag_freq_page',
        ):
            page = getattr(self, attr, None)
            if page is not None and hasattr(page, 'refresh_styles'):
                try:
                    page.refresh_styles()
                except Exception as e:
                    self.log(f"[theme] refresh_styles failed for {attr}: {e}")

    def resizeEvent(self, event):
        """Keep the resource drawer right-aligned when the window resizes."""
        super().resizeEvent(event)
        if hasattr(self, '_resource_drawer') and self._resource_drawer._is_open:
            cw = self.centralWidget()
            if cw:
                self._resource_drawer.setGeometry(
                    cw.width() - self._resource_drawer.DRAWER_WIDTH, 0,
                    self._resource_drawer.DRAWER_WIDTH, cw.height(),
                )
        # Reposition video nav badge in top-right of the button
        if hasattr(self, '_video_badge') and hasattr(self, 'page_video_btn'):
            btn = self.page_video_btn
            self._video_badge.move(btn.width() - 26, 6)

    # ═══════════ SETTINGS PAGE ═══════════

    def _build_settings_page(self) -> QWidget:
        """Build the Settings page with appearance, language and output settings."""
        page = QWidget()
        page.setStyleSheet(f"background: {theme.BG_WINDOW};")
        outer_lay = QVBoxLayout(page)
        outer_lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        inner = QWidget()
        inner.setStyleSheet(f"background: {theme.BG_WINDOW};")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        lay.addWidget(self._settings_section("🎨", "Appearance", self._settings_appearance_widget()))
        lay.addWidget(self._settings_section("🌐", "Language", self._settings_language_widget()))
        lay.addWidget(self._settings_section("📁", "Output Paths", self._settings_output_widget()))
        lay.addStretch()

        scroll.setWidget(inner)
        outer_lay.addWidget(scroll)
        return page

    def _settings_section(self, icon: str, title: str, content: QWidget) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER};
                      border-radius: 12px; }}
        """)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        hdr = QLabel(f"  {icon}  {title}")
        hdr.setFixedHeight(48)
        hdr.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(14)}; font-weight: 600;"
            f"border-bottom: 1px solid {theme.BORDER}; background: transparent; padding-left: 4px;"
        )
        lay.addWidget(hdr)
        lay.addWidget(content)
        return card

    def _settings_appearance_widget(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 12, 20, 16)
        lay.setSpacing(18)

        # Theme mode
        row1 = QHBoxLayout()
        lbl = QLabel("Theme Mode")
        lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; font-weight: 500; background: transparent; border: none;")
        row1.addWidget(lbl); row1.addStretch()
        dark_btn = QPushButton("☽  Dark")
        light_btn = QPushButton("☀  Light")
        for btn, mode in [(dark_btn, "dark"), (light_btn, "light")]:
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda _, m=mode: self._apply_theme_mode(m))
        self._theme_dark_btn = dark_btn
        self._theme_light_btn = light_btn
        self._update_theme_mode_btns()
        row1.addWidget(dark_btn); row1.addWidget(light_btn)
        lay.addLayout(row1)

        # Accent color swatches
        row2 = QHBoxLayout()
        lbl2 = QLabel("Accent Color")
        lbl2.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; font-weight: 500; background: transparent; border: none;")
        row2.addWidget(lbl2); row2.addStretch()
        self._accent_swatches = []
        for color, name in theme.ACCENT_PRESETS:
            sw = QPushButton()
            sw.setFixedSize(26, 26)
            sw.setToolTip(name)
            sw.setCursor(Qt.PointingHandCursor)
            sw.clicked.connect(lambda _, c=color: self._apply_accent_color(c))
            sw.setProperty("accent_color", color)
            self._accent_swatches.append(sw)
            row2.addWidget(sw)
        self._refresh_accent_swatches()
        lay.addLayout(row2)

        # Font scale
        row3 = QHBoxLayout()
        lbl3 = QLabel("Font Scale")
        lbl3.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; font-weight: 500; background: transparent; border: none;")
        self._font_scale_slider = QSlider(Qt.Horizontal)
        self._font_scale_slider.setRange(80, 140)
        self._font_scale_slider.setValue(int(theme.get_font_scale() * 100))
        self._font_scale_slider.setFixedWidth(160)
        self._font_scale_slider.setStyleSheet(theme.slider())
        self._font_scale_val_lbl = QLabel(f"{theme.get_font_scale():.1f}×")
        self._font_scale_val_lbl.setStyleSheet(
            f"color: {theme.ORANGE}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(11)}; background: transparent; border: none; min-width: 36px;"
        )
        self._font_scale_slider.valueChanged.connect(self._on_font_scale_changed)
        row3.addWidget(lbl3); row3.addStretch()
        row3.addWidget(self._font_scale_slider); row3.addWidget(self._font_scale_val_lbl)
        lay.addLayout(row3)
        return w

    def _settings_language_widget(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w); lay.setContentsMargins(20, 12, 20, 16)
        lbl = QLabel("Interface Language")
        lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; background: transparent; border: none;")
        self._settings_lang_combo = QComboBox()
        self._settings_lang_combo.addItems(["English", "Türkçe"])
        self._settings_lang_combo.setCurrentIndex(0 if self.current_lang == 'en' else 1)
        self._settings_lang_combo.setStyleSheet(theme.combo())
        self._settings_lang_combo.setFixedWidth(180)
        self._settings_lang_combo.currentIndexChanged.connect(
            lambda i: self.change_language(i)
        )
        lay.addWidget(lbl); lay.addStretch(); lay.addWidget(self._settings_lang_combo)
        return w

    def _settings_output_widget(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w); lay.setContentsMargins(20, 12, 20, 16)
        lbl = QLabel("Default Output Folder")
        lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; background: transparent; border: none;")
        self._output_path_lbl = QLabel("—")
        self._output_path_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(11)}; background: transparent; border: none;"
        )
        browse = QPushButton("Browse")
        browse.setStyleSheet(theme.btn_secondary())
        browse.clicked.connect(self._browse_settings_output)
        lay.addWidget(lbl); lay.addStretch()
        lay.addWidget(self._output_path_lbl); lay.addWidget(browse)
        return w

    def _apply_theme_mode(self, mode: str):
        theme.set_theme(mode, theme.get_font_scale(), theme.get_accent())
        self._refresh_all_styles()
        if hasattr(self, '_theme_dark_btn'):
            self._update_theme_mode_btns()

    def _update_theme_mode_btns(self):
        active_s = (
            f"QPushButton {{ background: {theme.ORANGE_SUBTLE}; color: {theme.ORANGE};"
            f" border: 1px solid {theme.ORANGE_DIM}; border-radius: 6px; padding: 0 12px;"
            f" font-size: {theme.fs(12)}; font-weight: 600; }}"
        )
        passive_s = (
            f"QPushButton {{ background: transparent; color: {theme.TEXT_MUTED};"
            f" border: 1px solid {theme.BORDER}; border-radius: 6px; padding: 0 12px;"
            f" font-size: {theme.fs(12)}; }}"
            f" QPushButton:hover {{ background: {theme.BG_HOVER}; color: {theme.TEXT_PRIMARY}; }}"
        )
        is_dark = theme.get_mode() == "dark"
        if hasattr(self, '_theme_dark_btn'):
            self._theme_dark_btn.setStyleSheet(active_s if is_dark else passive_s)
            self._theme_light_btn.setStyleSheet(active_s if not is_dark else passive_s)

    def _apply_accent_color(self, color: str):
        theme.set_theme(theme.get_mode(), theme.get_font_scale(), color)
        self._refresh_all_styles()
        if hasattr(self, '_accent_swatches'):
            self._refresh_accent_swatches()

    def _refresh_accent_swatches(self):
        current = theme.get_accent().lower()
        for sw in self._accent_swatches:
            c = sw.property("accent_color")
            is_active = c.lower() == current
            border = "2px solid rgba(255,255,255,0.75)" if is_active else "2px solid transparent"
            sw.setStyleSheet(
                f"QPushButton {{ background: {c}; border: {border}; border-radius: 6px; }}"
                f" QPushButton:hover {{ border: 2px solid rgba(255,255,255,0.5); }}"
            )

    def _on_font_scale_changed(self, val: int):
        scale = val / 100.0
        if hasattr(self, '_font_scale_val_lbl'):
            self._font_scale_val_lbl.setText(f"{scale:.1f}×")
        theme.set_theme(theme.get_mode(), scale, theme.get_accent())

    def _browse_settings_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder and hasattr(self, '_output_path_lbl'):
            self._output_path_lbl.setText(folder)

    def _toggle_panel(self, panel_name: str):
        panels = {
            'quality': (self.quality_btn, self.quality_panel),
            'caption': (self.caption_btn, self.caption_panel),
            'tags': (self.tags_btn, self.tags_panel)
        }
        btn, panel = panels[panel_name]
        is_checked = btn.isChecked()
        smooth_expand(panel, expand=is_checked, duration=220)
    
    def switch_page(self, index: int):
        old_idx = self.page_stack.currentIndex()
        for i, btn in enumerate(self._nav_buttons):
            btn.setStyleSheet(self._page_btn_style(i == index))
        animate_page_switch(self.page_stack, old_idx, index, duration=220)
        if hasattr(self, '_nav_indicator'):
            self._nav_indicator.move_under(self._nav_buttons[index])
    
    # ── helpers for bento card frames ──────────────────────────────────

    def _bento_card(self, title: str = None) -> QFrame:
        """Dark card with optional title label, returns (card, body_layout)."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {theme.BG_CARD};
                border: 1px solid {theme.BORDER};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(10)
        if title:
            hdr = QLabel(title)
            hdr.setStyleSheet(
                f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; font-weight: 600;"
                f" background: transparent; border: none; letter-spacing: -0.01em;"
            )
            lay.addWidget(hdr)
        return card

    def _row_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)};"
            f" font-family: {theme.FONT_MONO}; background: transparent; border: none;"
        )
        return lbl

    def setup_video_page(self):
        """Setup video processing page — bento grid layout."""
        from PyQt5.QtWidgets import QGridLayout
        from src.ui.animations import ProgressGlowBar

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.video_page.setLayout(root)

        # ── Hidden widgets kept for backward-compat (refresh_styles etc.) ──
        self.title_label = QLabel(get_text('title', self.current_lang))
        self.title_label.hide()
        self.subtitle_label = QLabel(get_text('subtitle', self.current_lang))
        self.subtitle_label.hide()
        self.settings_group = QGroupBox()
        self.settings_group.hide()

        # Hidden collapsible buttons (referenced in _refresh_all_styles / _toggle_panel)
        self.quality_btn  = QPushButton(); self.quality_btn.setCheckable(True);  self.quality_btn.hide()
        self.caption_btn  = QPushButton(); self.caption_btn.setCheckable(True);  self.caption_btn.hide()
        self.tags_btn     = QPushButton(); self.tags_btn.setCheckable(True);     self.tags_btn.hide()

        # ══════════════════════════════════════════════════
        #  BENTO ROW:  left (scroll) ║ right column
        # ══════════════════════════════════════════════════
        bento_row = QHBoxLayout()
        bento_row.setContentsMargins(20, 16, 20, 0)
        bento_row.setSpacing(16)

        # ─── LEFT COLUMN (stretch=2) ───
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("background: transparent; border: none;")

        left_inner = QWidget()
        left_inner.setStyleSheet("background: transparent;")
        left_lay = QVBoxLayout(left_inner)
        left_lay.setContentsMargins(0, 0, 0, 16)
        left_lay.setSpacing(12)

        # -- Drop Zone --
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.on_files_dropped)
        self.drop_zone.setMinimumHeight(110)
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())

        browse_row = QHBoxLayout()
        browse_row.setSpacing(8)
        browse_row.addWidget(self.drop_zone, stretch=1)
        self.browse_btn = QPushButton(get_text('browse_btn', self.current_lang))
        self.browse_btn.setStyleSheet(theme.btn_browse())
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self.browse_video)
        self.browse_btn.setFixedWidth(140)
        browse_row.addWidget(self.browse_btn, alignment=Qt.AlignTop)
        left_lay.addLayout(browse_row)
        self.update_drop_zone_text()

        # -- 2-col settings bento --
        bento_2col = QHBoxLayout()
        bento_2col.setSpacing(10)

        # Extraction Config card
        ext_card = self._bento_card("⚙  Extraction Config")
        ext_lay = ext_card.layout()

        # Frame interval
        iv_row = QHBoxLayout()
        self.interval_label = self._row_label(get_text('frame_interval', self.current_lang))
        iv_row.addWidget(self.interval_label)
        iv_row.addStretch()
        self.interval_value_label = QLabel("30")
        self.interval_value_label.setStyleSheet(theme.label_value())
        iv_row.addWidget(self.interval_value_label)
        ext_lay.addLayout(iv_row)
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setMinimum(1); self.interval_slider.setMaximum(120)
        self.interval_slider.setValue(30); self.interval_slider.setStyleSheet(theme.slider())
        self.interval_help = QLabel("ℹ️"); self.interval_help.setStyleSheet(theme.info_icon()); self.interval_help.hide()
        self.interval_slider.valueChanged.connect(lambda v: self.interval_value_label.setText(str(v)))
        ext_lay.addWidget(self.interval_slider)

        # Trim
        trim_row = QHBoxLayout(); trim_row.setSpacing(6)
        trim_lbl = self._row_label(get_text('trim_label', self.current_lang))
        self.trim_label = trim_lbl; self.trim_help = QLabel(); self.trim_help.hide()
        self.trim_start_spin = QSpinBox(); self.trim_start_spin.setRange(0, 600); self.trim_start_spin.setSuffix(" s"); self.trim_start_spin.setStyleSheet(theme.spinbox_compact())
        self.trim_end_spin = QSpinBox(); self.trim_end_spin.setRange(0, 600); self.trim_end_spin.setSuffix(" s"); self.trim_end_spin.setStyleSheet(theme.spinbox_compact())
        self.trim_start_lbl = self._row_label(get_text('trim_start', self.current_lang))
        self.trim_end_lbl = self._row_label(get_text('trim_end', self.current_lang))
        trim_row.addWidget(trim_lbl); trim_row.addStretch()
        trim_row.addWidget(self.trim_start_lbl); trim_row.addWidget(self.trim_start_spin)
        trim_row.addWidget(self.trim_end_lbl); trim_row.addWidget(self.trim_end_spin)
        ext_lay.addLayout(trim_row)
        ext_lay.addStretch()

        bento_2col.addWidget(ext_card)

        # Processing Options card
        proc_card = self._bento_card("🎛  Processing Options")
        proc_lay = proc_card.layout()

        # Aspect ratio
        ratio_row = QHBoxLayout()
        self.ratio_label = self._row_label(get_text('output_format', self.current_lang))
        self.ratio_help = QLabel(); self.ratio_help.hide()
        self.ratio_combo = QComboBox(); self.ratio_combo.addItems(['9:16', '3:4', '1:1', '4:5', '16:9', '4:3']); self.ratio_combo.setStyleSheet(theme.combo_compact())
        ratio_row.addWidget(self.ratio_label); ratio_row.addStretch(); ratio_row.addWidget(self.ratio_combo)
        proc_lay.addLayout(ratio_row)

        # Confidence + Ensemble toggle
        conf_row = QHBoxLayout()
        self.conf_label = self._row_label(get_text('confidence', self.current_lang))
        self.conf_help = QLabel(); self.conf_help.hide()
        self.conf_spinbox = QSpinBox(); self.conf_spinbox.setRange(10, 95); self.conf_spinbox.setValue(50); self.conf_spinbox.setSuffix("%"); self.conf_spinbox.setStyleSheet(theme.spinbox_compact())
        conf_row.addWidget(self.conf_label); conf_row.addStretch(); conf_row.addWidget(self.conf_spinbox)
        proc_lay.addLayout(conf_row)

        # Ensemble checkbox
        ens_row = QHBoxLayout()
        self.ensemble_cb = QCheckBox(get_text('ensemble_mode', self.current_lang))
        self.ensemble_cb.setChecked(False); self.ensemble_cb.setStyleSheet(theme.checkbox_frame())
        self.ensemble_help = QLabel(); self.ensemble_help.hide()
        ens_row.addWidget(self.ensemble_cb); ens_row.addStretch()
        proc_lay.addLayout(ens_row)

        # Ensemble group (hidden initially)
        self.ensemble_group = QGroupBox(); self.ensemble_group.setStyleSheet(theme.panel_group())
        ensemble_layout = QVBoxLayout()
        models_layout = QHBoxLayout()
        self.models_label = self._row_label(get_text('active_models', self.current_lang))
        self.yolo_cb = QCheckBox("YOLOv8"); self.yolo_cb.setChecked(True); self.yolo_cb.setEnabled(False); self.yolo_cb.setStyleSheet(theme.label_default())
        models_layout.addWidget(self.models_label); models_layout.addWidget(self.yolo_cb); models_layout.addStretch()
        voting_layout = QHBoxLayout()
        self.voting_label = self._row_label(get_text('voting_threshold', self.current_lang))
        self.voting_help = QLabel(); self.voting_help.hide()
        self.voting_spinbox = QSpinBox(); self.voting_spinbox.setRange(1, 1); self.voting_spinbox.setValue(1); self.voting_spinbox.setStyleSheet(theme.spinbox_compact())
        voting_layout.addWidget(self.voting_label); voting_layout.addWidget(self.voting_spinbox); voting_layout.addStretch()
        ensemble_layout.addLayout(models_layout); ensemble_layout.addLayout(voting_layout)
        self.ensemble_group.setLayout(ensemble_layout)
        self.ensemble_group.setVisible(False)
        proc_lay.addWidget(self.ensemble_group)
        self.ensemble_cb.toggled.connect(lambda c: smooth_expand(self.ensemble_group, expand=c, duration=200))

        # Skip subtitle + turbo
        opts_row = QHBoxLayout()
        self.skip_subtitle_cb = QCheckBox(get_text('skip_subtitle', self.current_lang)); self.skip_subtitle_cb.setChecked(True); self.skip_subtitle_cb.setStyleSheet(theme.checkbox_frame())
        self.skip_help = QLabel(); self.skip_help.hide()
        self.turbo_cb = QCheckBox(get_text('turbo_mode', self.current_lang)); self.turbo_cb.setChecked(True); self.turbo_cb.setStyleSheet(theme.checkbox_frame())
        self.turbo_help = QLabel(); self.turbo_help.hide()
        opts_row.addWidget(self.skip_subtitle_cb); opts_row.addWidget(self.turbo_cb); opts_row.addStretch()
        proc_lay.addLayout(opts_row)

        # Min padding (hidden row — kept for build_config compat)
        self.padding_label = self._row_label(get_text('min_padding', self.current_lang)); self.padding_label.hide()
        self.padding_help = QLabel(); self.padding_help.hide()
        self.padding_spinbox = QSpinBox(); self.padding_spinbox.setRange(100, 1000); self.padding_spinbox.setValue(500); self.padding_spinbox.setSingleStep(50); self.padding_spinbox.hide()
        proc_lay.addStretch()

        bento_2col.addWidget(proc_card)
        left_lay.addLayout(bento_2col)

        # -- Accordion panels (self-contained) --
        self.quality_panel = QualitySettingsPanel(self.current_lang)
        self.caption_panel = CaptioningSettingsPanel(self.current_lang)
        self.tags_panel    = TagSettingsPanel(self.current_lang)
        for panel in (self.quality_panel, self.caption_panel, self.tags_panel):
            left_lay.addWidget(panel)

        left_lay.addStretch()
        left_scroll.setWidget(left_inner)
        bento_row.addWidget(left_scroll, stretch=2)

        # ─── RIGHT COLUMN (stretch=1) ───
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(10)

        # Stat cards
        stats_row = QHBoxLayout(); stats_row.setSpacing(8)
        self._stat_cards = {}
        for key, icon, label_key in (
            ('queued',    '📋', 'stat_queued'),
            ('extracted', '⚡', 'stat_extracted'),
            ('saved',     '💾', 'stat_saved'),
        ):
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER};"
                f" border-radius: 10px; }}"
            )
            cl = QVBoxLayout(card); cl.setContentsMargins(10, 10, 10, 10); cl.setAlignment(Qt.AlignCenter)
            val_lbl = QLabel("0")
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet(f"background: transparent; border: none; color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(20)}; font-weight: 700;")
            desc_lbl = QLabel(get_text(label_key, self.current_lang))
            desc_lbl.setAlignment(Qt.AlignCenter)
            desc_lbl.setStyleSheet(f"background: transparent; border: none; color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; font-family: {theme.FONT_MONO};")
            cl.addWidget(val_lbl); cl.addWidget(desc_lbl)
            stats_row.addWidget(card, stretch=1)
            self._stat_cards[key] = {'frame': card, 'value': val_lbl, 'desc': desc_lbl, 'label_key': label_key}
        right_col.addLayout(stats_row)

        # Action buttons card
        action_card = self._bento_card()
        act_lay = action_card.layout()
        act_lay.setSpacing(6)
        top_btns = QHBoxLayout(); top_btns.setSpacing(6)
        self.process_btn = QPushButton(get_text('start_btn', self.current_lang)); self.process_btn.setEnabled(False); self.process_btn.setStyleSheet(theme.btn_action_start()); self.process_btn.clicked.connect(self.start_processing)
        self.pause_btn   = QPushButton(get_text('pause_btn', self.current_lang));  self.pause_btn.setEnabled(False);  self.pause_btn.setStyleSheet(theme.btn_action_pause()); self.pause_btn.clicked.connect(self.toggle_pause)
        top_btns.addWidget(self.process_btn); top_btns.addWidget(self.pause_btn)
        bot_btns = QHBoxLayout(); bot_btns.setSpacing(6)
        self.skip_btn = QPushButton(get_text('skip_btn', self.current_lang)); self.skip_btn.setEnabled(False); self.skip_btn.setStyleSheet(theme.btn_action_skip()); self.skip_btn.clicked.connect(self.skip_current_video)
        self.stop_btn = QPushButton(get_text('stop_btn', self.current_lang)); self.stop_btn.setEnabled(False); self.stop_btn.setStyleSheet(theme.btn_action_stop()); self.stop_btn.clicked.connect(self.stop_processing)
        bot_btns.addWidget(self.skip_btn); bot_btns.addWidget(self.stop_btn)
        self.open_output_btn = QPushButton(get_text('open_output_btn', self.current_lang)); self.open_output_btn.setStyleSheet(theme.btn_secondary()); self.open_output_btn.clicked.connect(self.open_output_folder)
        act_lay.addLayout(top_btns); act_lay.addLayout(bot_btns); act_lay.addWidget(self.open_output_btn)
        right_col.addWidget(action_card)

        # Hover lift + ripple
        for _btn in (self.process_btn, self.pause_btn, self.skip_btn, self.stop_btn, self.open_output_btn, self.browse_btn):
            HoverLift(_btn, lift_px=2); RippleButton(_btn)

        # Progress card
        prog_card = self._bento_card()
        prog_lay = prog_card.layout(); prog_lay.setSpacing(6)
        prog_info = QHBoxLayout()
        self._prog_file_lbl = QLabel("Idle")
        self._prog_file_lbl.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; font-family: {theme.FONT_MONO}; background: transparent; border: none;")
        self._prog_pct_lbl = QLabel("0%")
        self._prog_pct_lbl.setStyleSheet(f"color: {theme.ORANGE}; font-size: {theme.fs(11)}; font-family: {theme.FONT_MONO}; background: transparent; border: none;")
        prog_info.addWidget(self._prog_file_lbl); prog_info.addStretch(); prog_info.addWidget(self._prog_pct_lbl)
        prog_lay.addLayout(prog_info)
        self.progress_bar = ProgressGlowBar()
        prog_lay.addWidget(self.progress_bar)
        self._progress_glow = ProgressGlow(self.progress_bar, color=theme.get_accent())
        right_col.addWidget(prog_card)

        # Thumbnail grid card
        thumb_card = self._bento_card("Latest Frames")
        thumb_lay = thumb_card.layout(); thumb_lay.setSpacing(6)
        self.preview_label = QLabel("Latest Frames"); self.preview_label.hide()
        self.log_label = QLabel(); self.log_label.hide()
        self._preview_frame = QFrame()
        self._preview_frame.setStyleSheet("background: transparent; border: none;")
        self._preview_grid = QGridLayout(self._preview_frame)
        self._preview_grid.setSpacing(4); self._preview_grid.setContentsMargins(0, 0, 0, 0)
        self._PREVIEW_COLS = 4; self._PREVIEW_ROWS = 2
        self._preview_labels = []
        for r in range(self._PREVIEW_ROWS):
            for c in range(self._PREVIEW_COLS):
                lbl = QLabel(); lbl.setFixedSize(60, 60); lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(f"background: {theme.BG_SURFACE}; border: 1px solid {theme.BORDER}; border-radius: 4px;")
                self._preview_grid.addWidget(lbl, r, c); self._preview_labels.append(lbl)
        thumb_lay.addWidget(self._preview_frame)
        right_col.addWidget(thumb_card)
        right_col.addStretch()

        right_widget = QWidget(); right_widget.setStyleSheet("background: transparent;")
        right_widget.setLayout(right_col)
        bento_row.addWidget(right_widget, stretch=1)

        root.addLayout(bento_row, stretch=1)
        self.video_page.setMinimumHeight(640)

        # ── Log terminal (full width) ──
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(160)
        self.log_text.setStyleSheet(theme.log_area())
        log_wrap = QWidget(); log_wrap.setStyleSheet("background: transparent;")
        log_wl = QVBoxLayout(log_wrap); log_wl.setContentsMargins(20, 8, 20, 12)
        log_wl.addWidget(self.log_text)
        root.addWidget(log_wrap)

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
    
    _VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

    def on_files_dropped(self, file_paths: List[str]):
        """Handle file / folder / .txt list selection.

        Folder drop → recursively walk and collect all video files.
        .txt drop   → read line-by-line as video paths.
        Otherwise   → treat each item as a direct video path.
        """
        resolved: List[str] = []
        for p in file_paths:
            pp = Path(p)
            if pp.is_dir():
                # Recursive video walk
                found = sorted(
                    str(f) for f in pp.rglob('*')
                    if f.is_file() and f.suffix.lower() in self._VIDEO_EXTENSIONS
                )
                resolved.extend(found)
                if found:
                    self.log(f"📂 Folder scanned: {pp.name} ({len(found)} videos)")
                else:
                    self.log(f"⚠️ No videos found in {pp.name}")
            elif pp.suffix.lower() == '.txt' and pp.is_file():
                try:
                    lines = pp.read_text(encoding='utf-8').splitlines()
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        lp = Path(line)
                        if lp.is_file() and lp.suffix.lower() in self._VIDEO_EXTENSIONS:
                            resolved.append(str(lp))
                    self.log(f"📋 Loaded {len(resolved)} paths from {pp.name}")
                except Exception as e:
                    self.log(f"❌ Failed to read list file {pp.name}: {e}")
            else:
                resolved.append(str(pp))

        if not resolved:
            self.log("⚠️ No valid video files found")
            return

        prev_count = len(getattr(self, 'video_paths', []) or [])
        self.video_paths = resolved
        self.process_btn.setEnabled(True)

        if len(resolved) == 1:
            self.log(get_text('log_loaded', self.current_lang).format(Path(resolved[0]).name))
            self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(1))
        else:
            names = ', '.join([Path(f).name for f in resolved[:3]])
            if len(resolved) > 3:
                names += f' ... (+{len(resolved)-3} more)'
            self.log(get_text('log_loaded', self.current_lang).format(names))
            self.log(get_text('log_batch_mode', self.current_lang).format(len(resolved)))
            self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(len(resolved)))

        # Pop the drop zone on successful file drop
        scale_pop(self.drop_zone, factor=1.03, duration=260)

        # Animate the queued count + nav badge
        try:
            cur = int(self._stat_cards['queued']['value'].text())
        except (ValueError, KeyError):
            cur = prev_count
        count_up(self._stat_cards['queued']['value'], cur, len(resolved))
        self._update_video_badge(len(resolved))

        # Toast notification
        ToastNotification(
            get_text('toast_files_added', self.current_lang).format(len(resolved)),
            icon='📥', duration_ms=2400, accent=theme.get_accent(),
            parent=self.centralWidget(),
        )
    
    def _cleanup_processing_thread(self):
        """Disconnect signals and schedule deletion of the current processing thread."""
        if self.processing_thread is not None:
            t = self.processing_thread
            self.processing_thread = None  # clear ref first so stale signals are ignored
            try:
                t.progress_update.disconnect(self.on_progress)
                t.log_message.disconnect(self.log)
                t.processing_finished.disconnect(self.on_finished)
                t.error.disconnect(self.on_error)
                t.frame_saved.disconnect(self._on_preview_frame)
            except (TypeError, RuntimeError):
                pass
            if not t.isRunning():
                t.deleteLater()
            else:
                # Thread still alive (in final cleanup after emitting signal).
                # Connect QThread's built-in finished() — NOT our custom signal —
                # so deleteLater() fires safely when run() fully returns.
                # Qt keeps the C++ object alive until then.
                try:
                    t.finished.connect(t.deleteLater)
                except (TypeError, RuntimeError):
                    pass

    def toggle_pause(self):
        """Toggle pause/resume on the processing thread."""
        if self.processing_thread and self.processing_thread.isRunning():
            now_paused = self.processing_thread.toggle_pause()
            if now_paused:
                self.pause_btn.setText(get_text('resume_btn', self.current_lang))
                self.log(get_text('log_paused', self.current_lang))
                self._set_status('paused')
                if hasattr(self, '_progress_glow'):
                    self._progress_glow.stop()
                ToastNotification(
                    get_text('toast_paused', self.current_lang),
                    icon='⏸', duration_ms=1800, accent='#F0C040',
                    parent=self.centralWidget(),
                )
            else:
                self.pause_btn.setText(get_text('pause_btn', self.current_lang))
                self.log(get_text('log_resumed', self.current_lang))
                self._set_status('processing')
                if hasattr(self, '_progress_glow'):
                    self._progress_glow.start()
                ToastNotification(
                    get_text('toast_resumed', self.current_lang),
                    icon='▶', duration_ms=1600, accent=theme.get_accent(),
                    parent=self.centralWidget(),
                )

    def stop_processing(self):
        """Stop video processing. Immediately re-enables the Start button so
        the user can restart with the same file without reselecting."""
        try:
            self._stop_processing_impl()
        except Exception as _e:
            import traceback, datetime
            with open('crash_log.txt', 'a', encoding='utf-8', errors='replace') as _f:
                _f.write(f'\n[{datetime.datetime.now()}] stop_processing CRASH:\n')
                traceback.print_exc(file=_f)

    def _stop_processing_impl(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.log(get_text('log_stopping', self.current_lang))
            self.processing_thread.stop()
            self.stop_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText(get_text('pause_btn', self.current_lang))
            self.drop_zone.setEnabled(True)
            # Re-enable Start immediately if we still have video paths so the
            # user can retry the same files without reselecting.
            if self.video_paths:
                self.process_btn.setEnabled(True)
            # Reset visual state
            self._set_status('idle')
            if hasattr(self, '_progress_glow'):
                self._progress_glow.stop()
            ToastNotification(
                get_text('toast_stopped', self.current_lang),
                icon='⏹', duration_ms=2000, accent='#888',
                parent=self.centralWidget(),
            )

    def skip_current_video(self):
        """Skip the currently processing video and move on to the next one."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.log(get_text('log_skipping_current', self.current_lang))
            self.processing_thread.skip_current_video()
            # Briefly disable the skip button to debounce rapid clicks —
            # it's re-enabled once the next video actually starts. The
            # simplest approximation: leave it disabled until the user
            # sees the progress bar advance again (we'll just re-enable
            # it on the next progress update).
            self.skip_btn.setEnabled(False)
    
    def start_processing(self):
        """Start video processing. Always begins from scratch — cancelling
        a previous run and pressing Start again restarts the full pipeline."""
        if not self.video_paths:
            self.log(get_text('log_no_file', self.current_lang))
            shake_widget(self.drop_zone)
            return

        # If a previous thread is still alive (stop pending), signal it and
        # clean up non-blockingly. _cleanup_processing_thread() wires up
        # QThread.finished → deleteLater so the old thread self-destructs
        # when its run() returns — no blocking wait on the main thread.
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
        self._cleanup_processing_thread()

        # Reset progress state for a fresh run
        self.progress_bar.setValue(0)
        self._reset_stat('extracted')
        self._reset_stat('saved')

        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText(get_text('pause_btn', self.current_lang))
        # Skip only makes sense when there's more than one video in the
        # batch — otherwise it's just a funny-looking Stop.
        self.skip_btn.setEnabled(len(self.video_paths) > 1)
        self.drop_zone.setEnabled(False)
        # Clear the preview thumbnails for a fresh run.
        for lbl in self._preview_labels:
            lbl.clear()

        # Visual state — status dot, progress glow, toast
        self._set_status('processing')
        if hasattr(self, '_progress_glow'):
            self._progress_glow.start()
        ToastNotification(
            get_text('toast_processing_started', self.current_lang),
            icon='🚀', duration_ms=2000, accent=theme.get_accent(),
            parent=self.centralWidget(),
        )

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
        models_to_use = ['yolo']
        voting_threshold = 1
        if use_ensemble:
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
            'start_skip_seconds': float(self.trim_start_spin.value()),
            'end_skip_seconds': float(self.trim_end_spin.value()),
            'quality_settings': quality_settings,
            'caption_settings': caption_settings,
            'tag_settings': tag_settings,
            # Resource settings from the drawer
            'resource_settings': dict(self._resource_cfg),
        }

        # Start processing thread (models load in background)
        self.processing_thread = ProcessingThread(config)
        self.processing_thread.progress_update.connect(self.on_progress)
        self.processing_thread.log_message.connect(self.log)
        self.processing_thread.processing_finished.connect(self.on_finished)
        self.processing_thread.error.connect(self.on_error)
        self.processing_thread.frame_saved.connect(self._on_preview_frame)
        self.processing_thread.start()
    
    def _on_preview_frame(self, path: str):
        """Slot connected to ProcessingThread.frame_saved.

        Shifts the preview grid left and places the new thumbnail at the
        rightmost position. The image is down-sampled to 96x96 in the
        main thread; since JPEGs are small (typically <100KB) and we fire
        at most once per saved frame the overhead is negligible."""
        from PyQt5.QtGui import QPixmap
        try:
            pix = QPixmap(path)
            if pix.isNull():
                return
            thumb = pix.scaled(
                96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        except Exception:
            return
        # Shift left: move every label's pixmap one slot to the left, then
        # set the last slot to the new thumbnail.
        for i in range(len(self._preview_labels) - 1):
            pm = self._preview_labels[i + 1].pixmap()
            if pm and not pm.isNull():
                self._preview_labels[i].setPixmap(pm)
            else:
                self._preview_labels[i].clear()
        self._preview_labels[-1].setPixmap(thumb)
        fade_in(self._preview_labels[-1], duration=160)

    def on_progress(self, progress: float, stats: dict):
        """Update progress"""
        try:
            self._on_progress_impl(progress, stats)
        except Exception as _e:
            import traceback, datetime
            with open('crash_log.txt', 'a', encoding='utf-8', errors='replace') as _f:
                _f.write(f'\n[{datetime.datetime.now()}] on_progress CRASH:\n')
                traceback.print_exc(file=_f)

    def _on_progress_impl(self, progress: float, stats: dict):
        progress_smooth(self.progress_bar, int(progress), duration=160)
        # The skip button gets disabled while the "skip" request is
        # in-flight; as soon as we see a progress tick again it means
        # the next video has started and the button is safe to re-arm.
        if (self.processing_thread and self.processing_thread.isRunning()
                and len(self.video_paths) > 1
                and not self.skip_btn.isEnabled()):
            self.skip_btn.setEnabled(True)

        # Live stat cards — extracted = frames pulled from video,
        # saved = frames kept after filtering
        self._bump_stat('extracted', int(stats.get('processed_frames', 0)))
        self._bump_stat('saved', int(stats.get('saved_frames', 0)))

        # Update log with stats
        processed = stats.get('processed_frames', 0)
        if processed % 10 == 0:
            self.log(get_text('log_progress', self.current_lang).format(
                progress,
                stats.get('saved_frames', 0),
                stats.get('person_frames', 0),
                stats.get('animal_frames', 0),
                stats.get('object_frames', 0),
            ))
    
    def on_finished(self, stats: dict):
        """Processing finished"""
        try:
            self._on_finished_impl(stats)
        except Exception as _e:
            import traceback, datetime
            with open('crash_log.txt', 'a', encoding='utf-8', errors='replace') as _f:
                _f.write(f'\n[{datetime.datetime.now()}] on_finished CRASH:\n')
                traceback.print_exc(file=_f)

    def _on_finished_impl(self, stats: dict):
        # Guard against stale signal from an old / already-cleaned-up thread.
        # After stop_processing() re-enables the Start button the user may have
        # already pressed Start again, reassigning self.processing_thread.
        # sender() lets us verify the signal came from the current thread.
        if self.processing_thread is None:
            return
        sender = self.sender()
        if sender is not None and sender is not self.processing_thread:
            return

        # If stopped by user, just clean up — buttons were already re-enabled
        # in stop_processing().
        if stats.get('stopped'):
            self._cleanup_processing_thread()
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
            self.log(get_text('log_total', self.current_lang).format(stats.get('saved_frames', 0)))
            self.log(get_text('log_persons', self.current_lang).format(stats.get('person_frames', 0)))
            self.log(get_text('log_animals', self.current_lang).format(stats.get('animal_frames', 0)))
            self.log(get_text('log_objects', self.current_lang).format(stats.get('object_frames', 0)))
            self.log(get_text('log_skipped_text', self.current_lang).format(stats.get('skipped_text', 0)))
            self.log(get_text('log_skipped_none', self.current_lang).format(stats.get('skipped_no_detection', 0)))
            # V2.0 stats
            if stats.get('skipped_quality', 0) > 0:
                self.log(f"🔍 Skipped (quality): {stats['skipped_quality']}")
            captioned = stats.get('captioned_frames', 0)
            saved = stats.get('saved_frames', 0)
            self.log(f"📝 Captioned: {captioned}/{saved} frames")
            if stats.get('overlay_crops', 0) > 0:
                self.log(f"🛡️ Overlay-aware crops: {stats['overlay_crops']}")
        
        self.log("="*50)

        # Final stat card sync — lock in the canonical totals
        final_saved = int(
            stats.get('total_frames_saved', stats.get('saved_frames', 0))
        )
        final_processed = int(
            stats.get('total_frames_processed', stats.get('processed_frames', 0))
        )
        if final_processed:
            self._bump_stat('extracted', final_processed)
        self._bump_stat('saved', final_saved)

        # Status + ambient animations
        self._set_status('done')
        if hasattr(self, '_progress_glow'):
            self._progress_glow.stop()
        ToastNotification(
            get_text('toast_processing_done', self.current_lang).format(final_saved),
            icon='✅', duration_ms=3000, accent=theme.get_accent(),
            parent=self.centralWidget(),
        )
        # Queue badge: processing finished, the queue is empty again
        self.video_paths = []
        self._update_video_badge(0)
        self._reset_stat('queued')

        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText(get_text('pause_btn', self.current_lang))
        self.drop_zone.setEnabled(True)
        self._cleanup_processing_thread()

    def on_error(self, error_msg: str):
        """Handle error"""
        try:
            self._on_error_impl(error_msg)
        except Exception as _e:
            import traceback, datetime
            with open('crash_log.txt', 'a', encoding='utf-8', errors='replace') as _f:
                _f.write(f'\n[{datetime.datetime.now()}] on_error CRASH:\n')
                traceback.print_exc(file=_f)

    def _on_error_impl(self, error_msg: str):
        self.log(get_text('log_error', self.current_lang).format(error_msg))
        self._set_status('error')
        if hasattr(self, '_progress_glow'):
            self._progress_glow.stop()
        glitch_effect(self.progress_bar)
        ToastNotification(
            get_text('toast_processing_error', self.current_lang).format(error_msg),
            icon='⚠', duration_ms=4000, accent=theme.RED,
            parent=self.centralWidget(),
        )
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText(get_text('pause_btn', self.current_lang))
        self.drop_zone.setEnabled(True)
        self._cleanup_processing_thread()

    def closeEvent(self, event):
        """Ensure all threads stop when window is closed"""
        # Stop processing thread
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.safe_wait(5000)

        # Stop captioning thread (from caption studio page)
        if hasattr(self, 'caption_studio_page'):
            studio = self.caption_studio_page
            gen_tab = getattr(studio, 'generate_tab', None)
            if gen_tab:
                ct = getattr(gen_tab, 'captioning_thread', None)
                if ct and ct.isRunning():
                    ct.stop()
                    ct.wait(5000)
                gen_tab._safe_delete_thread()

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
        
        # Sidebar nav buttons (icon + label)
        _nav_labels = [
            ("  🎬  ", 'page_video_processing'),
            ("  🏷  ", 'page_caption_studio'),
            ("  👥  ", 'page_character_sort'),
            ("  📊  ", 'page_tag_frequency'),
        ]
        for btn, (icon, key) in zip(self._nav_buttons, _nav_labels):
            btn.setText(f"{icon}{get_text(key, self.current_lang)}")

        self.res_settings_btn.setText("⚙")
        self.res_settings_btn.setToolTip(get_text('res_menu_tooltip', self.current_lang))
        if hasattr(self, '_resource_drawer'):
            self._resource_drawer.update_language(self.current_lang)
        if hasattr(self, '_topbar_monitor'):
            self._topbar_monitor.update_language(self.current_lang)

        # Update caption studio page language
        if hasattr(self, 'caption_studio_page'):
            self.caption_studio_page.update_language(self.current_lang)

        # Update character sort page language
        if hasattr(self, 'char_sort_page'):
            self.char_sort_page.update_language(self.current_lang)

        # Update tag frequency page language
        if hasattr(self, 'tag_freq_page'):
            self.tag_freq_page.update_language(self.current_lang)

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
        self.pause_btn.setText(get_text('pause_btn', self.current_lang))
        self.pause_btn.setToolTip(get_text('pause_btn_tooltip', self.current_lang))
        self.skip_btn.setText(get_text('skip_btn', self.current_lang))
        self.skip_btn.setToolTip(get_text('skip_btn_tooltip', self.current_lang))
        self.stop_btn.setText(get_text('stop_btn', self.current_lang))
        self.open_output_btn.setText(get_text('open_output_btn', self.current_lang))
        self.trim_label.setText(get_text('trim_label', self.current_lang))
        self.trim_help.setToolTip(get_text('trim_tooltip', self.current_lang))
        self.trim_start_lbl.setText(get_text('trim_start', self.current_lang))
        self.trim_end_lbl.setText(get_text('trim_end', self.current_lang))
        self.preview_label.setText(get_text('preview_title', self.current_lang))
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

        # Topbar status label — re-translate without changing the current state
        if hasattr(self, '_status_dot') and hasattr(self, '_status_label'):
            state = getattr(self._status_dot, '_state', 'idle')
            key_map = {
                'idle':       'status_idle',
                'processing': 'status_processing',
                'paused':     'status_paused',
                'done':       'status_done',
                'error':      'status_error',
            }
            self._status_label.setText(
                get_text(key_map.get(state, 'status_idle'), self.current_lang)
            )

        # Stat card description labels (Queued / Extracted / Saved)
        if hasattr(self, '_stat_cards'):
            for card in self._stat_cards.values():
                desc = card.get('desc')
                key = card.get('label_key')
                if desc and key:
                    desc.setText(get_text(key, self.current_lang))
    
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
