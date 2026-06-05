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
    ToggleSwitch, Chip, SearchCombo, ThumbnailGrid,
    ProgressSteps, FloatingActionButton,
)
from src.ui.advanced_settings import (
    QualitySettingsPanel,
    CaptioningSettingsPanel,
    TagSettingsPanel,
    UpscaleSettingsPanel,
)
from src.ui.caption_studio_page import CaptionStudioPage
from src.ui.character_sort_page import CharacterSortPage
from src.ui.tag_frequency_page import TagFrequencyPage
from src.ui.review_grid_page import ReviewGridPage
from src.ui.upscale_page import UpscalePage
from src.ui.training_page import TrainingPage
from src.ui.resource_settings import ResourceSettingsDrawer


class _NextStepBanner(QFrame):
    """Dismissable contextual next-step suggestion shown after processing."""
    go_to_page = pyqtSignal(int)   # emits target page index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("nextStepBanner")
        self._build()
        self.hide()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        self._icon = QLabel("✅")
        self._icon.setStyleSheet("font-size: 18px; background: transparent; border: none;")
        self._msg = QLabel()
        self._msg.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: 13px; font-weight: 600; "
            f"background: transparent; border: none;"
        )
        lay.addWidget(self._icon)
        lay.addWidget(self._msg)
        lay.addStretch()
        self._btn_container = QHBoxLayout()
        lay.addLayout(self._btn_container)
        close = QPushButton("✕")
        close.setFixedSize(24, 24)
        close.setStyleSheet(
            "QPushButton { background: transparent; color: #888; border: none; }"
            " QPushButton:hover { color: white; }"
        )
        close.clicked.connect(self.hide)
        lay.addWidget(close)
        self.setStyleSheet(
            "QFrame#nextStepBanner { background: #1a2a1a; border-bottom: 1px solid #2d4a2d; }"
        )

    def show_suggestion(self, message: str, actions: list):
        """actions: list of (label, page_idx) tuples."""
        self._msg.setText(message)
        # Clear old action buttons
        while self._btn_container.count():
            item = self._btn_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for label, page_idx in actions:
            btn = QPushButton(label)
            btn.setStyleSheet(
                "QPushButton { background: #2d4a2d; color: #7ec87e; "
                "border: 1px solid #3d6a3d; border-radius: 4px; padding: 4px 12px; "
                "font-size: 12px; }"
                " QPushButton:hover { background: #3d6a3d; }"
            )
            btn.clicked.connect(lambda _, p=page_idx: (self.go_to_page.emit(p), self.hide()))
            self._btn_container.addWidget(btn)
        self.show()
        QTimer.singleShot(10000, self.hide)  # auto-hide after 10 s


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
        # Throttle frame_saved emits — loading a QPixmap + animation on the
        # UI thread for every saved frame floods it (freeze). Cap to ~6/sec.
        self._last_frame_saved_emit = 0.0
        self._frame_saved_min_interval = 0.16

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

            # Detector — build based on detection_mode
            detection_mode = cfg.get('detection_mode', 'yolo')

            if cfg['use_ensemble'] and detection_mode == 'yolo':
                from src.core.ensemble_detector import EnsembleDetector
                self.log_message.emit("Loading ensemble models...")
                detector = EnsembleDetector(
                    models_to_use=cfg['models_to_use'],
                    confidence_threshold=cfg['confidence'],
                    voting_threshold=cfg['voting_threshold']
                )
                self.log_message.emit(f"Ensemble loaded: {', '.join(cfg['models_to_use'])}")
            elif detection_mode == 'anime':
                from src.core.anime_detector import AnimeDetector
                self.log_message.emit("Loading AnimeDetector (imgutils/cascade)...")
                detector = AnimeDetector(confidence=cfg['confidence'])
                self.log_message.emit(
                    f"AnimeDetector ready (will load backend on first frame)")
            elif detection_mode == 'auto':
                from src.core.detector import ObjectDetector
                from src.core.anime_detector import AnimeDetector, AutoDetector
                self.log_message.emit("Loading Auto detector (YOLO + AnimeDetector)...")
                yolo = ObjectDetector(confidence=cfg['confidence'])
                anime = AnimeDetector(confidence=cfg['confidence'])
                detector = AutoDetector(yolo, anime)
                self.log_message.emit("Auto detector ready")
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

            # Upscaler (V3.x)
            upscaler = None
            us_cfg = cfg.get('upscale_settings', {})
            if us_cfg.get('enabled', False):
                try:
                    from src.core.upscaler import FrameUpscaler
                    upscaler = FrameUpscaler(
                        model_name=us_cfg.get('model', 'RealESRGAN_x4plus_anime_6B'),
                        tile=us_cfg.get('tile', 0),
                        use_gpu=us_cfg.get('use_gpu', True),
                        face_enhance=us_cfg.get('face_enhance', False),
                    )
                    if upscaler.is_available():
                        self.log_message.emit(
                            f"✅ Upscaler ready: {us_cfg.get('model')} "
                            f"(scale={upscaler.get_scale()}×)"
                        )
                    else:
                        self.log_message.emit(
                            "⚠️ Upscaler deps missing — upscale disabled. "
                            "Install: pip install realesrgan basicsr"
                        )
                        upscaler = None
                except Exception as e:
                    self.log_message.emit(f"❌ Upscaler init error: {e}")
                    upscaler = None

            if not self._is_running:
                _stopped = True
                return

            # V3.x NSFW detector — simple on/off, auto backend
            nsfw_detector = None
            if cfg.get('nsfw_settings', {}).get('enabled', False):
                try:
                    import torch
                    from src.core.nsfw_detector import NsfwDetector
                    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                    nsfw_detector = NsfwDetector(backend='auto', threshold=0.70, device=dev)
                    if nsfw_detector.is_available():
                        self.log_message.emit(
                            f"✅ NSFW klasörleme aktif (backend={nsfw_detector._active_backend})"
                        )
                    else:
                        self.log_message.emit("⚠️ NSFW: backend bulunamadi, heuristic ile devam")
                except Exception as exc:
                    self.log_message.emit(f"⚠️ NSFW detector baslatma hatasi: {exc}")
                    nsfw_detector = None

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
                upscaler=upscaler,
                upscale_target=us_cfg.get('target', 'crop'),
                upscale_min_resolution=us_cfg.get('min_resolution', 512),
                upscale_max_resolution=us_cfg.get('max_resolution', 0),
                subtitle_removal=cfg.get('subtitle_removal', False),
                nsfw_detector=nsfw_detector,
                nsfw_uncertain_folder=nsfw_cfg.get('uncertain_folder', True),
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
        load the thumbnail on the main thread. Throttled to avoid flooding
        the UI thread with disk reads + animations (freeze fix)."""
        if not self._is_running:
            return
        import time as _time
        now = _time.monotonic()
        if (now - self._last_frame_saved_emit) >= self._frame_saved_min_interval:
            self._last_frame_saved_emit = now
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

    files_dropped = pyqtSignal(list)
    click_browse = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Active language for self-rendered text. MainWindow keeps this in
        # sync via `drop_zone.lang = self.current_lang` before calling
        # update_drop_zone_text(); defaults to English on first build.
        self.lang = 'en'
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(theme.drop_zone_default())
        self.setTextFormat(Qt.RichText)
        self.setText(
            f"<div style='line-height:1.6;'>"
            f"<div style='font-size:28px;'>☁</div>"
            f"<div style='font-size:14px;font-weight:600;color:#f1dfd4;margin:4px 0 2px;'>"
            f"{get_text('drop_zone_idle_title', self.lang)}</div>"
            f"<div style='font-size:11px;color:#a38c7d;'>{get_text('drop_zone_idle_hint_browse', self.lang)}</div>"
            f"</div>"
        )
        self._pulse = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.click_browse.emit()
        super().mousePressEvent(event)

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
                parts.append(get_text('drop_zone_part_videos', self.lang).format(len(vids)))
            if dirs:
                parts.append(get_text('drop_zone_part_folders', self.lang).format(len(dirs)))
            if txts:
                parts.append(get_text('drop_zone_part_lists', self.lang).format(len(txts)))
            self.setText(get_text('drop_zone_dropped', self.lang).format(', '.join(parts)))
        else:
            self.setText(get_text('drop_zone_invalid', self.lang))


class VideoSmartCropperUI(QMainWindow):
    """Main UI window with page-based navigation"""
    
    def __init__(self):
        super().__init__()
        self.video_paths = []  # Changed to list for batch support
        self.processor = None
        self.processing_thread = None
        self.current_lang = theme.get_lang()  # Load saved language (persisted)

        self.init_ui()
        self._page_status = {}
        QTimer.singleShot(500, self._check_crash_log)
    
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
        ws_label = QLabel(get_text('workspace_label', self.current_lang))
        ws_label.setStyleSheet(theme.sidebar_section_label())
        sidebar_lay.addWidget(ws_label)
        sidebar_lay.addSpacing(4)
        self._sidebar_section_labels = [ws_label]

        self.page_video_btn = QPushButton("Video Harvester")
        self.page_caption_studio_btn = QPushButton(get_text('page_caption_studio', self.current_lang))
        self.page_char_sort_btn = QPushButton(get_text('page_character_sort', self.current_lang))

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

        # Section: LIBRARY
        lib_label = QLabel(get_text('library_label', self.current_lang))
        lib_label.setStyleSheet(theme.sidebar_section_label())
        sidebar_lay.addWidget(lib_label)
        sidebar_lay.addSpacing(4)
        self._sidebar_section_labels.append(lib_label)

        self.page_tag_freq_btn = QPushButton(get_text('page_tag_frequency', self.current_lang))
        self.page_tag_freq_btn.setCursor(Qt.PointingHandCursor)
        self.page_tag_freq_btn.setStyleSheet(self._page_btn_style(False))
        self.page_tag_freq_btn.clicked.connect(lambda: self.switch_page(3))
        sidebar_lay.addWidget(self.page_tag_freq_btn)
        self._nav_buttons.append(self.page_tag_freq_btn)

        self.page_review_btn = QPushButton(get_text('page_review', self.current_lang))
        self.page_review_btn.setCursor(Qt.PointingHandCursor)
        self.page_review_btn.setStyleSheet(self._page_btn_style(False))
        self.page_review_btn.clicked.connect(lambda: self.switch_page(4))
        sidebar_lay.addWidget(self.page_review_btn)
        self._nav_buttons.append(self.page_review_btn)

        # Upscale page lives at stack index 6 (added after Settings@5), so its
        # nav index is 6 — Settings keeps index 5 unchanged. The button widget
        # sits here visually (LIBRARY), but it is appended to _nav_buttons AFTER
        # Settings below so its list index (6) matches switch_page(6).
        self.page_upscale_btn = QPushButton(get_text('page_upscale', self.current_lang))
        self.page_upscale_btn.setCursor(Qt.PointingHandCursor)
        self.page_upscale_btn.setStyleSheet(self._page_btn_style(False))
        self.page_upscale_btn.clicked.connect(lambda: self.switch_page(6))
        sidebar_lay.addWidget(self.page_upscale_btn)

        # Sliding underline indicator for the active nav button
        self._nav_indicator = NavIndicator(self._sidebar, color=theme.get_accent(), height=3)
        QTimer.singleShot(0, lambda: self._nav_indicator.move_under(self._nav_buttons[0]))

        sidebar_lay.addStretch()

        # Settings button (footer)
        self.page_settings_btn = QPushButton("Settings")
        self.page_settings_btn.setCursor(Qt.PointingHandCursor)
        self.page_settings_btn.setStyleSheet(self._page_btn_style(False))
        self.page_settings_btn.clicked.connect(lambda: self.switch_page(5))
        sidebar_lay.addWidget(self.page_settings_btn)
        self._nav_buttons.append(self.page_settings_btn)   # index 5
        # Upscale at index 6 — button created earlier in the LIBRARY section.
        self._nav_buttons.append(self.page_upscale_btn)    # index 6

        # Training page — index 7
        self.page_training_btn = QPushButton(
            "7  Eğitim" if self.current_lang == 'tr' else "7  Training"
        )
        self.page_training_btn.setCursor(Qt.PointingHandCursor)
        self.page_training_btn.setStyleSheet(self._page_btn_style(False))
        self.page_training_btn.clicked.connect(lambda: self.switch_page(7))
        sidebar_lay.addWidget(self.page_training_btn)
        self._nav_buttons.append(self.page_training_btn)   # index 7

        root_layout.addWidget(self._sidebar)

        # ═══════════ RIGHT AREA (topbar + content) ═══════════
        right_area = QVBoxLayout()
        right_area.setContentsMargins(0, 0, 0, 0)
        right_area.setSpacing(0)
        self._main_content_layout = right_area  # used by crash banner insertWidget

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

        # (drawer removed — settings now in sidebar Settings page)

        right_area.addWidget(self._topbar)
        self._topbar_monitor.start()

        # ── Next Step Banner (contextual, hidden by default) ──
        self._next_step_banner = _NextStepBanner()
        self._next_step_banner.go_to_page.connect(self.switch_page)
        right_area.addWidget(self._next_step_banner)

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

        # Review Grid page (index 4)
        self.review_grid_page = ReviewGridPage(self.current_lang)
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setWidget(self.review_grid_page)
        review_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(review_scroll)

        # Settings page (index 5)
        self.settings_page = self._build_settings_page()
        self.page_stack.addWidget(self.settings_page)

        # Upscale page (index 6) — standalone batch upscaler
        self.upscale_page = UpscalePage(self.current_lang)
        upscale_scroll = QScrollArea()
        upscale_scroll.setWidgetResizable(True)
        upscale_scroll.setWidget(self.upscale_page)
        upscale_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(upscale_scroll)

        # Training page (index 7) — Kohya integrated training
        self.training_page = TrainingPage(self.current_lang)
        training_scroll = QScrollArea()
        training_scroll.setWidgetResizable(True)
        training_scroll.setWidget(self.training_page)
        training_scroll.setFrameShape(QFrame.NoFrame)
        self.page_stack.addWidget(training_scroll)

        right_area.addWidget(self.page_stack, stretch=1)
        right_container = QWidget()
        right_container.setLayout(right_area)
        root_layout.addWidget(right_container, stretch=1)

        # Legacy lang_combo kept for backward compat (embed_lang_combo calls)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['English', 'Türkçe'])
        self.lang_combo.setCurrentIndex(0 if self.current_lang == 'en' else 1)
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        from src.ui.resource_settings import load_settings
        self._resource_cfg = load_settings()
        self._apply_hardware_limits(self._resource_cfg)

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
        pass  # drawer removed — settings are in sidebar Settings page

    def _apply_hardware_limits(self, data: dict):
        """Apply CPU/GPU resource limits to the live process so changes take
        effect without an app restart. Subsequent processing runs build a fresh
        processor from self._resource_cfg, and these process-global knobs
        (torch/cv2 thread counts, CUDA memory fraction) update immediately."""
        try:
            threads = int(data.get("cpu_threads", 0) or 0)
            if threads > 0:
                try:
                    import torch
                    torch.set_num_threads(threads)
                except Exception:
                    pass
                try:
                    import cv2
                    cv2.setNumThreads(threads)
                except Exception:
                    pass
                import os as _os
                _os.environ["OMP_NUM_THREADS"] = str(threads)
        except Exception as e:
            self.log(f"[resources] thread limit apply failed: {e}")
        try:
            if data.get("gpu_enabled", True):
                pct = int(data.get("gpu_mem_limit_pct", 0) or 0)
                if 0 < pct <= 100:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.set_per_process_memory_fraction(pct / 100.0, 0)
        except Exception as e:
            self.log(f"[resources] VRAM limit apply failed: {e}")

    def _on_resource_settings_changed(self, data: dict):
        """Slot: user clicked Apply in the resource drawer."""
        self._resource_cfg = data
        self._apply_hardware_limits(data)
        new_mode = data.get("theme_mode", "dark")
        new_scale = data.get("font_scale", 100) / 100.0
        new_accent = data.get("accent", theme.get_accent())
        mode_changed = new_mode != theme.get_mode()
        scale_changed = abs(new_scale - theme.get_font_scale()) > 0.01
        accent_changed = (new_accent or "").lower() != theme.get_accent().lower()
        if mode_changed or scale_changed or accent_changed:
            theme.set_theme(new_mode, new_scale, accent=new_accent)
            # Debounce: schedule refresh after 80ms so rapid swatch clicks don't freeze
            if not hasattr(self, '_style_timer'):
                self._style_timer = QTimer(self)
                self._style_timer.setSingleShot(True)
                self._style_timer.timeout.connect(self._refresh_all_styles)
            self._style_timer.start(80)
        self.log(get_text('res_apply', self.current_lang))

    def _refresh_all_styles(self):
        """Re-apply all stylesheets after a theme change."""
        # Apply global stylesheet to QApplication so all popup/child windows inherit it
        _gs = theme.global_stylesheet()
        self.setStyleSheet(_gs)
        QApplication.instance().setStyleSheet(_gs)

        # Deep-refresh all lhCard frames across all pages in one pass
        _card_ss = (
            f"QFrame {{background:{theme.BG_CARD};border:1px solid {theme.BORDER_LIGHT};"
            f"border-radius:10px;}}"
        )
        for w in self.centralWidget().findChildren(QFrame):
            if w.property("lhCard"):
                w.setStyleSheet(_card_ss)
            elif w.property("lhActionCard"):
                w.setStyleSheet(
                    f"QFrame {{background:{theme.BG_CARD};border:1px solid {theme.BORDER_LIGHT};"
                    f"border-top:2px solid {theme.ORANGE};border-radius:10px;}}"
                )

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
        # (drawer removed)

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
        if hasattr(self, 'upscale_panel') and hasattr(self.upscale_panel, 'refresh_styles'):
            self.upscale_panel.refresh_styles()
        if hasattr(self, 'review_grid_page') and hasattr(self.review_grid_page, 'refresh_styles'):
            self.review_grid_page.refresh_styles()
        if hasattr(self, 'upscale_page') and hasattr(self.upscale_page, 'refresh_styles'):
            self.upscale_page.refresh_styles()

        # Progress steps (video page)
        if hasattr(self, '_progress_steps'):
            self._progress_steps._build()
        if hasattr(self, '_fab'):
            self._fab.apply_theme()

        if hasattr(self, '_video_bento_frames'):
            for frame in self._video_bento_frames:
                frame.setStyleSheet(
                    f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER_LIGHT}; border-radius: 10px; }}"
                )
        # Drop zone (video page) also needs bolder border
        if hasattr(self, 'drop_zone'):
            self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())

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
            'tag_freq_page', 'training_page',
        ):
            page = getattr(self, attr, None)
            if page is not None and hasattr(page, 'refresh_styles'):
                try:
                    page.refresh_styles()
                except Exception as e:
                    self.log(f"[theme] refresh_styles failed for {attr}: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_video_badge') and hasattr(self, 'page_video_btn'):
            btn = self.page_video_btn
            self._video_badge.move(btn.width() - 26, 6)
        self._update_fab_position()

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

        lay.addWidget(self._settings_section("🎨", get_text('settings_sec_appearance', self.current_lang), self._settings_appearance_widget()))
        lay.addWidget(self._settings_section("🌐", get_text('settings_sec_language', self.current_lang), self._settings_language_widget()))
        lay.addWidget(self._settings_section("📁", get_text('settings_sec_output_paths', self.current_lang), self._settings_output_widget()))
        lay.addWidget(self._settings_section("⚡", get_text('settings_sec_gpu', self.current_lang), self._settings_gpu_widget()))
        lay.addWidget(self._settings_section("🚀", get_text('settings_sec_performance', self.current_lang), self._settings_perf_widget()))
        lay.addWidget(self._settings_section("🧵", get_text('settings_sec_cpu', self.current_lang), self._settings_cpu_widget()))
        lay.addWidget(self._settings_section("💾", get_text('settings_sec_memory', self.current_lang), self._settings_memory_widget()))
        lay.addWidget(self._settings_section("🔧", get_text('settings_sec_misc', self.current_lang), self._settings_misc_widget()))
        lay.addStretch()

        # Apply / Reset row (bottom of settings content)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 8, 0, 8)
        reset_btn = QPushButton(get_text('settings_reset', self.current_lang))
        reset_btn.setStyleSheet(theme.btn_secondary())
        reset_btn.clicked.connect(self._settings_reset_defaults)
        apply_btn = QPushButton(get_text('settings_apply', self.current_lang))
        apply_btn.setStyleSheet(theme.btn_primary())
        apply_btn.clicked.connect(self._settings_apply)
        btn_row.addWidget(reset_btn); btn_row.addStretch(); btn_row.addWidget(apply_btn)
        lay.addLayout(btn_row)

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
        lbl = QLabel(get_text('settings_theme_mode', self.current_lang))
        lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; font-weight: 500; background: transparent; border: none;")
        row1.addWidget(lbl); row1.addStretch()
        dark_btn = QPushButton(get_text('theme_dark', self.current_lang))
        light_btn = QPushButton(get_text('theme_light', self.current_lang))
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
        lbl2 = QLabel(get_text('settings_accent_color', self.current_lang))
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
        lbl3 = QLabel(get_text('settings_font_scale', self.current_lang))
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
        lbl = QLabel(get_text('settings_language', self.current_lang))
        lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; background: transparent; border: none;")
        self._settings_lang_combo = QComboBox()
        self._settings_lang_combo.addItems(["English", "Türkçe"])
        self._settings_lang_combo.setCurrentIndex(0 if self.current_lang == 'en' else 1)
        self._settings_lang_combo.setStyleSheet(theme.combo())
        self._settings_lang_combo.setFixedWidth(180)
        self._settings_lang_combo.setMaxVisibleItems(5)
        self._settings_lang_combo.currentIndexChanged.connect(
            lambda i: self.change_language(i)
        )
        lay.addWidget(lbl); lay.addStretch(); lay.addWidget(self._settings_lang_combo)
        return w

    def _settings_output_widget(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w); lay.setContentsMargins(20, 12, 20, 16)
        lbl = QLabel(get_text('settings_output_folder', self.current_lang))
        lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; background: transparent; border: none;")
        self._output_path_lbl = QLabel("—")
        self._output_path_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(11)}; background: transparent; border: none;"
        )
        browse = QPushButton(get_text('settings_browse', self.current_lang))
        browse.setStyleSheet(theme.btn_secondary())
        browse.clicked.connect(self._browse_settings_output)
        lay.addWidget(lbl); lay.addStretch()
        lay.addWidget(self._output_path_lbl); lay.addWidget(browse)
        return w

    # ── Settings section builders (migrated from ResourceSettingsDrawer) ──

    def _settings_row(self, label_text: str, widget, suffix_lbl: QLabel = None) -> QHBoxLayout:
        """Helper: label on left, control + optional value label on right."""
        row = QHBoxLayout(); row.setSpacing(12)
        lbl = QLabel(label_text)
        lbl.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(12)};"
            f" background: transparent; border: none; min-width: 160px;"
        )
        row.addWidget(lbl)
        row.addWidget(widget, stretch=1)
        if suffix_lbl:
            row.addWidget(suffix_lbl)
        return row

    def _settings_slider(self, mn, mx, step, suffix=""):
        """Helper: returns (QSlider, value QLabel)."""
        sl = QSlider(Qt.Horizontal); sl.setMinimum(mn); sl.setMaximum(mx)
        sl.setSingleStep(step); sl.setStyleSheet(theme.slider())
        val = QLabel(f"{mn}{suffix}")
        val.setStyleSheet(
            f"color: {theme.ORANGE}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(11)};"
            f" background: transparent; border: none; min-width: 50px; qproperty-alignment: AlignRight;"
        )
        sl.valueChanged.connect(lambda v, s=suffix, lbl=val: lbl.setText(f"{v}{s}"))
        return sl, val

    def _settings_gpu_widget(self) -> QWidget:
        from src.ui.resource_settings import load_settings
        s = load_settings()
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w); lay.setContentsMargins(20, 12, 20, 14); lay.setSpacing(12)

        self._sg_gpu_cb = ToggleSwitch("Enable GPU Acceleration", checked=s.get("gpu_enabled", True))
        lay.addWidget(self._sg_gpu_cb)

        self._sg_fp16_cb = ToggleSwitch("FP16 Half Precision", checked=s.get("fp16_enabled", True))
        lay.addWidget(self._sg_fp16_cb)

        self._sg_vram_sl, self._sg_vram_lbl = self._settings_slider(10, 100, 5, "%")
        self._sg_vram_sl.setValue(s.get("gpu_mem_limit_pct", 80))
        lay.addLayout(self._settings_row("VRAM Limit", self._sg_vram_sl, self._sg_vram_lbl))
        return w

    def _settings_perf_widget(self) -> QWidget:
        from src.ui.resource_settings import load_settings
        s = load_settings()
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w); lay.setContentsMargins(20, 12, 20, 14); lay.setSpacing(12)

        self._sg_batch_sl, self._sg_batch_lbl = self._settings_slider(1, 32, 1)
        self._sg_batch_sl.setValue(s.get("batch_size", 8))
        lay.addLayout(self._settings_row("Batch Size", self._sg_batch_sl, self._sg_batch_lbl))

        self._sg_prefetch_sl, self._sg_prefetch_lbl = self._settings_slider(1, 120, 1)
        self._sg_prefetch_sl.setValue(s.get("prefetch_frames", 32))
        lay.addLayout(self._settings_row("Prefetch Frames", self._sg_prefetch_sl, self._sg_prefetch_lbl))
        return w

    def _settings_cpu_widget(self) -> QWidget:
        import os
        from src.ui.resource_settings import load_settings
        s = load_settings()
        cpu_count = max(1, os.cpu_count() or 4)
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w); lay.setContentsMargins(20, 12, 20, 14); lay.setSpacing(12)

        self._sg_threads_sl, self._sg_threads_lbl = self._settings_slider(1, cpu_count, 1)
        self._sg_threads_sl.setValue(s.get("cpu_threads", min(4, cpu_count)))
        lay.addLayout(self._settings_row("CPU Threads", self._sg_threads_sl, self._sg_threads_lbl))

        self._sg_workers_sl, self._sg_workers_lbl = self._settings_slider(1, cpu_count, 1)
        self._sg_workers_sl.setValue(s.get("decode_workers", min(2, cpu_count)))
        lay.addLayout(self._settings_row("Decode Workers", self._sg_workers_sl, self._sg_workers_lbl))
        return w

    def _settings_memory_widget(self) -> QWidget:
        from src.ui.resource_settings import load_settings
        s = load_settings()
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w); lay.setContentsMargins(20, 12, 20, 14); lay.setSpacing(12)

        self._sg_ram_sl, self._sg_ram_lbl = self._settings_slider(512, 32768, 256, " MB")
        self._sg_ram_sl.setValue(s.get("ram_limit_mb", 4096))
        lay.addLayout(self._settings_row("RAM Limit", self._sg_ram_sl, self._sg_ram_lbl))
        return w

    def _settings_misc_widget(self) -> QWidget:
        from src.ui.resource_settings import load_settings
        s = load_settings()
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w); lay.setContentsMargins(20, 12, 20, 14); lay.setSpacing(12)

        self._sg_async_cb = ToggleSwitch("Async Save (non-blocking I/O)", checked=s.get("async_save", True))
        lay.addWidget(self._sg_async_cb)

        self._sg_gc_cb = ToggleSwitch("Auto GC (garbage collect after batch)", checked=s.get("auto_gc", True))
        lay.addWidget(self._sg_gc_cb)

        self._sg_jpeg_sl, self._sg_jpeg_lbl = self._settings_slider(50, 100, 5)
        self._sg_jpeg_sl.setValue(s.get("jpeg_quality", 92))
        lay.addLayout(self._settings_row("JPEG Quality", self._sg_jpeg_sl, self._sg_jpeg_lbl))
        return w

    def _settings_apply(self):
        """Collect all settings page values and apply."""
        from src.ui.resource_settings import save_settings
        data = {
            "gpu_enabled":       self._sg_gpu_cb.isChecked(),
            "fp16_enabled":      self._sg_fp16_cb.isChecked(),
            "gpu_mem_limit_pct": self._sg_vram_sl.value(),
            "batch_size":        self._sg_batch_sl.value(),
            "prefetch_frames":   self._sg_prefetch_sl.value(),
            "cpu_threads":       self._sg_threads_sl.value(),
            "decode_workers":    self._sg_workers_sl.value(),
            "ram_limit_mb":      self._sg_ram_sl.value(),
            "async_save":        self._sg_async_cb.isChecked(),
            "auto_gc":           self._sg_gc_cb.isChecked(),
            "jpeg_quality":      self._sg_jpeg_sl.value(),
            "theme_mode":        theme.get_mode(),
            "font_scale":        int(theme.get_font_scale() * 100),
            "accent":            theme.get_accent(),
        }
        save_settings(data)
        self._on_resource_settings_changed(data)

    def _settings_reset_defaults(self):
        """Reset all performance settings to defaults."""
        from src.ui.resource_settings import load_settings, DEFAULT_SETTINGS
        if hasattr(self, '_sg_gpu_cb'):
            self._sg_gpu_cb.setChecked(DEFAULT_SETTINGS.get("gpu_enabled", True))
            self._sg_fp16_cb.setChecked(DEFAULT_SETTINGS.get("fp16_enabled", True))
            self._sg_vram_sl.setValue(DEFAULT_SETTINGS.get("gpu_mem_limit_pct", 80))
            self._sg_batch_sl.setValue(DEFAULT_SETTINGS.get("batch_size", 8))
            self._sg_prefetch_sl.setValue(DEFAULT_SETTINGS.get("prefetch_frames", 32))
            self._sg_threads_sl.setValue(DEFAULT_SETTINGS.get("cpu_threads", 4))
            self._sg_workers_sl.setValue(DEFAULT_SETTINGS.get("decode_workers", 2))
            self._sg_ram_sl.setValue(DEFAULT_SETTINGS.get("ram_limit_mb", 4096))
            self._sg_async_cb.setChecked(DEFAULT_SETTINGS.get("async_save", True))
            self._sg_gc_cb.setChecked(DEFAULT_SETTINGS.get("auto_gc", True))
            self._sg_jpeg_sl.setValue(DEFAULT_SETTINGS.get("jpeg_quality", 92))

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
        # Debounce refresh so rapid swatch clicks don't freeze the UI
        if not hasattr(self, '_style_timer'):
            self._style_timer = QTimer(self)
            self._style_timer.setSingleShot(True)
            self._style_timer.timeout.connect(self._refresh_all_styles)
        self._style_timer.start(80)
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
        # Live-apply font scale without restart (debounced so dragging stays smooth)
        if not hasattr(self, '_style_timer'):
            self._style_timer = QTimer(self)
            self._style_timer.setSingleShot(True)
            self._style_timer.timeout.connect(self._refresh_all_styles)
        self._style_timer.start(80)

    def _browse_settings_output(self):
        folder = QFileDialog.getExistingDirectory(self, get_text('dlg_select_output_folder', self.current_lang))
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
        card.setProperty("lhCard", True)
        card.setStyleSheet(
            f"QFrame {{background:{theme.BG_CARD};border:1px solid {theme.BORDER_LIGHT};"
            f"border-radius:10px;}}"
        )
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

        # ── Progress Steps strip ─────────────────────────────────
        self._progress_steps = ProgressSteps(
            ["Select Files", "Configure", "Process", "Done"],
            current=0,
        )
        self._progress_steps.setContentsMargins(20, 8, 20, 0)
        root.addWidget(self._progress_steps)

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
        self.drop_zone.click_browse.connect(self.browse_video)
        self.drop_zone.setMinimumHeight(110)
        self.drop_zone.setStyleSheet(theme.drop_zone_frame_default())

        browse_row = QHBoxLayout()
        browse_row.setSpacing(8)
        browse_row.addWidget(self.drop_zone, stretch=1)
        self.browse_btn = QPushButton(get_text('browse_btn', self.current_lang))
        self.browse_btn.setStyleSheet(theme.btn_browse())
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self.browse_video)
        self.browse_btn.setMinimumWidth(170)
        browse_row.addWidget(self.browse_btn, alignment=Qt.AlignTop)
        left_lay.addLayout(browse_row)
        self.update_drop_zone_text()

        # -- 2-col settings bento --
        bento_2col = QHBoxLayout()
        bento_2col.setSpacing(10)
        self._video_bento_frames = []

        # Extraction Config card
        ext_card = self._bento_card("⚙  Extraction Config")
        self._video_bento_frames.append(ext_card)
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
        self._video_bento_frames.append(proc_card)
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

        # Detection mode selector (YOLO / Anime / Auto)
        det_row = QHBoxLayout()
        self.detection_mode_label = self._row_label(get_text('detection_mode', self.current_lang))
        self.detection_mode_help = QLabel(); self.detection_mode_help.hide()
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItem(get_text('detection_mode_yolo', self.current_lang), 'yolo')
        self.detection_mode_combo.addItem(get_text('detection_mode_anime', self.current_lang), 'anime')
        self.detection_mode_combo.addItem(get_text('detection_mode_auto', self.current_lang), 'auto')
        self.detection_mode_combo.setStyleSheet(theme.combo_compact())
        self.detection_mode_combo.setToolTip(get_text('detection_mode_tooltip', self.current_lang))
        det_row.addWidget(self.detection_mode_label); det_row.addStretch()
        det_row.addWidget(self.detection_mode_combo)
        proc_lay.addLayout(det_row)

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

        # Subtitle removal toggle (only active when skip_subtitle is ON)
        removal_row = QHBoxLayout()
        removal_row.setContentsMargins(20, 0, 0, 0)  # indent under skip_subtitle
        self.subtitle_removal_cb = QCheckBox(get_text('subtitle_removal', self.current_lang))
        self.subtitle_removal_cb.setChecked(False)
        self.subtitle_removal_cb.setStyleSheet(theme.checkbox_frame())
        self.subtitle_removal_cb.setToolTip(get_text('subtitle_removal_tooltip', self.current_lang))
        self.subtitle_removal_cb.setEnabled(self.skip_subtitle_cb.isChecked())
        self.skip_subtitle_cb.toggled.connect(
            lambda checked: self.subtitle_removal_cb.setEnabled(checked))
        removal_row.addWidget(self.subtitle_removal_cb)
        removal_row.addStretch()
        proc_lay.addLayout(removal_row)

        # NSFW detection — single checkbox, routes frames to sfw/nsfw subfolders
        nsfw_row = QHBoxLayout()
        self.nsfw_cb = QCheckBox(get_text('nsfw_separation', self.current_lang))
        self.nsfw_cb.setChecked(False)
        self.nsfw_cb.setStyleSheet(theme.checkbox_frame())
        self.nsfw_cb.setToolTip(get_text('nsfw_separation_tooltip', self.current_lang))
        nsfw_row.addWidget(self.nsfw_cb)
        nsfw_row.addStretch()
        proc_lay.addLayout(nsfw_row)

        # Min padding (hidden row — kept for build_config compat)
        self.padding_label = self._row_label(get_text('min_padding', self.current_lang)); self.padding_label.hide()
        self.padding_help = QLabel(); self.padding_help.hide()
        self.padding_spinbox = QSpinBox(); self.padding_spinbox.setRange(100, 1000); self.padding_spinbox.setValue(500); self.padding_spinbox.setSingleStep(50); self.padding_spinbox.hide()
        proc_lay.addStretch()

        bento_2col.addWidget(proc_card)
        left_lay.addLayout(bento_2col)

        # -- Accordion panels (self-contained) --
        self.quality_panel  = QualitySettingsPanel(self.current_lang)
        self.caption_panel  = CaptioningSettingsPanel(self.current_lang)
        self.tags_panel     = TagSettingsPanel(self.current_lang)
        self.upscale_panel  = UpscaleSettingsPanel(self.current_lang)
        for panel in (self.quality_panel, self.caption_panel, self.tags_panel,
                      self.upscale_panel):
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
            ('queued',    '#', 'stat_queued'),
            ('extracted', '#', 'stat_extracted'),
            ('saved',     '#', 'stat_saved'),
        ):
            card = QFrame()
            card.setProperty("lhCard", True)
            card.setStyleSheet(
                f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER_LIGHT};"
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

        # Tooltips (translatable — re-applied in update_ui_texts())
        self._apply_video_tooltips()

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
        thumb_card = self._bento_card(get_text('latest_frames', self.current_lang))
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

        # ── Floating Action Button (Start Processing quick-trigger) ──
        self._fab = FloatingActionButton("▶", size=52, parent=self.video_page)
        self._fab.setToolTip(get_text('fab_start_tooltip', self.current_lang))
        self._fab.clicked.connect(self.start_processing)
        self._fab.hide()   # shown when files are loaded
        self._fab.raise_()

    def _update_fab_position(self):
        if hasattr(self, '_fab') and self._fab:
            p = self.video_page
            self._fab.move(p.width() - 72, p.height() - 72)

    def _apply_video_tooltips(self):
        """(Re)apply translatable tooltips to the video page controls.
        Called once on build and again from update_ui_texts() on language switch."""
        lang = self.current_lang
        if hasattr(self, 'browse_btn'):
            self.browse_btn.setToolTip(get_text('browse_btn_tooltip', lang))
        if hasattr(self, 'interval_slider'):
            self.interval_slider.setToolTip(get_text('frame_interval_tooltip', lang))
        if hasattr(self, 'trim_start_spin'):
            self.trim_start_spin.setToolTip(get_text('trim_tooltip', lang))
        if hasattr(self, 'trim_end_spin'):
            self.trim_end_spin.setToolTip(get_text('trim_tooltip', lang))
        if hasattr(self, 'ratio_combo'):
            self.ratio_combo.setToolTip(get_text('output_format_tooltip', lang))
        if hasattr(self, 'conf_spinbox'):
            self.conf_spinbox.setToolTip(get_text('confidence_tooltip', lang))
        if hasattr(self, 'ensemble_cb'):
            self.ensemble_cb.setToolTip(get_text('ensemble_mode_tooltip', lang))
        if hasattr(self, 'skip_subtitle_cb'):
            self.skip_subtitle_cb.setToolTip(get_text('skip_subtitle_tooltip', lang))
        if hasattr(self, 'turbo_cb'):
            self.turbo_cb.setToolTip(get_text('turbo_mode_tooltip', lang))
        if hasattr(self, 'process_btn'):
            self.process_btn.setToolTip(get_text('process_btn_tooltip', lang))
        if hasattr(self, 'pause_btn'):
            self.pause_btn.setToolTip(get_text('pause_btn_tooltip', lang))
        if hasattr(self, 'skip_btn'):
            self.skip_btn.setToolTip(get_text('skip_btn_tooltip', lang))
        if hasattr(self, 'stop_btn'):
            self.stop_btn.setToolTip(get_text('stop_btn_tooltip', lang))
        if hasattr(self, 'open_output_btn'):
            self.open_output_btn.setToolTip(get_text('open_output_btn_tooltip', lang))
        if hasattr(self, '_fab') and self._fab:
            self._fab.setToolTip(get_text('fab_start_tooltip', lang))

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
            get_text('dlg_select_video_files', self.current_lang),
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

        # Advance progress steps to "Configure"
        if hasattr(self, '_progress_steps'):
            self._progress_steps.set_step(1)

        # Show FAB
        if hasattr(self, '_fab'):
            self._update_fab_position()
            self._fab.show()
            self._fab.raise_()

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
    
    def _thread_running(self) -> bool:
        """Safely check whether the processing thread is alive.
        Returns False if the thread is None or its C++ object was deleted."""
        if self.processing_thread is None:
            return False
        try:
            return self.processing_thread.isRunning()
        except RuntimeError:
            return False

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
            try:
                running = t.isRunning()
            except RuntimeError:
                # C++ object already gone — nothing to do.
                return
            if not running:
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
        if self._thread_running():
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
        if self._thread_running():
            self.log(get_text('log_stopping', self.current_lang))
            # Signal the thread to stop — do NOT block the UI with wait()
            self.processing_thread.stop()
            # Immediately update UI so user knows stop was acknowledged
            self.stop_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText(get_text('pause_btn', self.current_lang))
            self.drop_zone.setEnabled(True)
            if self.video_paths:
                self.process_btn.setEnabled(True)
            self._set_status('idle')
            if hasattr(self, '_progress_glow'):
                self._progress_glow.stop()
            if hasattr(self, '_progress_steps'):
                self._progress_steps.set_step(0)
            ToastNotification(
                get_text('toast_stopped', self.current_lang),
                icon='⏹', duration_ms=2000, accent='#888',
                parent=self.centralWidget(),
            )
            # Thread cleans up GPU resources in its own run() → finally block
            # on_finished() will fire via QueuedConnection when done

    def skip_current_video(self):
        """Skip the currently processing video and move on to the next one."""
        if self._thread_running():
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

        # If a previous thread is still alive (stop pending), we MUST wait for
        # it to fully release the GPU before starting a new run — otherwise the
        # old thread's _cleanup() (model release) races the new thread's model
        # loading, causing CUDA conflicts / hangs. A short blocking wait only
        # happens on restart, which is acceptable.
        if self.processing_thread is not None:
            if self._thread_running():
                self.processing_thread.stop()
                # Wait up to 8s for the thread to finish GPU cleanup.
                if not self.processing_thread.safe_wait(8000):
                    self.log("⚠️ Previous run still stopping — please wait a moment and try again.")
                    # Re-enable Start so the user can retry shortly.
                    self.process_btn.setEnabled(True)
                    return
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

        # Advance progress steps to "Process"
        if hasattr(self, '_progress_steps'):
            self._progress_steps.set_step(2)

        # Hide FAB during processing
        if hasattr(self, '_fab'):
            self._fab.hide()

        # Visual state — status dot, progress glow, toast.
        # Wrapped: a cosmetic effect must NEVER prevent the processing thread
        # from starting (a stale QGraphicsEffect once crashed here, leaving
        # progress frozen at 0% with nothing saved).
        try:
            self._set_status('processing')
            if hasattr(self, '_progress_glow'):
                self._progress_glow.start()
            ToastNotification(
                get_text('toast_processing_started', self.current_lang),
                icon='🚀', duration_ms=2000, accent=theme.get_accent(),
                parent=self.centralWidget(),
            )
        except Exception as _e:
            import traceback, datetime
            with open('crash_log.txt', 'a', encoding='utf-8', errors='replace') as _f:
                _f.write(f'\n[{datetime.datetime.now()}] start_processing visual-fx (non-fatal):\n')
                traceback.print_exc(file=_f)

        # Get settings
        frame_interval = self.interval_slider.value()
        skip_text = self.skip_subtitle_cb.isChecked()
        subtitle_removal = self.subtitle_removal_cb.isChecked() and skip_text
        detection_mode = self.detection_mode_combo.currentData() or 'yolo'
        confidence = self.conf_spinbox.value() / 100.0
        min_padding = self.padding_spinbox.value()
        aspect_ratio = self.ratio_combo.currentText()
        use_ensemble = self.ensemble_cb.isChecked()
        use_turbo = self.turbo_cb.isChecked()

        # V2.0 Settings
        quality_settings = self.quality_panel.get_settings()
        caption_settings = self.caption_panel.get_settings()
        tag_settings = self.tags_panel.get_settings()
        # V3.x
        upscale_settings = self.upscale_panel.get_settings()
        nsfw_settings   = {'enabled': self.nsfw_cb.isChecked() if hasattr(self, 'nsfw_cb') else False}

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
            'subtitle_removal': subtitle_removal,
            'detection_mode': detection_mode,
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
            'upscale_settings': upscale_settings,
            'nsfw_settings': nsfw_settings,
            # Resource settings from the drawer
            'resource_settings': dict(self._resource_cfg),
        }

        # Start processing thread (models load in background)
        # QueuedConnection ensures GPU-thread signals are marshalled safely to UI thread
        self.processing_thread = ProcessingThread(config)
        self.processing_thread.progress_update.connect(
            self.on_progress, Qt.QueuedConnection)
        self.processing_thread.log_message.connect(
            self.log, Qt.QueuedConnection)
        self.processing_thread.processing_finished.connect(
            self.on_finished, Qt.QueuedConnection)
        self.processing_thread.error.connect(
            self.on_error, Qt.QueuedConnection)
        self.processing_thread.frame_saved.connect(
            self._on_preview_frame, Qt.QueuedConnection)
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
        # Guard the thread access — the C++ object may have been deleted
        # if a stale signal arrives after cleanup (RuntimeError).
        try:
            thread_alive = (self.processing_thread is not None
                            and self.processing_thread.isRunning())
        except RuntimeError:
            thread_alive = False
        if (thread_alive and len(self.video_paths) > 1
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

        # Advance progress steps to "Done"
        if hasattr(self, '_progress_steps'):
            self._progress_steps.set_step(3)

        # Re-show FAB for next run
        if hasattr(self, '_fab') and getattr(self, 'video_paths', []):
            self._update_fab_position()
            self._fab.show()
            self._fab.raise_()

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

        # P3: mark Video page done
        self._mark_page_done(0)

        # P4: show next-step banner
        ok_count = int(stats.get('total_frames_saved', stats.get('saved_frames', 0)))
        if hasattr(self, '_next_step_banner'):
            self._next_step_banner.show_suggestion(
                f"✅ {ok_count} frame kaydedildi",
                [("→ Caption Studio", 1), ("→ Karakter Sırala", 2)],
            )

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
        """Ensure all threads stop when window is closed.

        Every step is guarded so that NOTHING can prevent the final
        ``os._exit(0)`` from firing — otherwise an exception here (e.g. a
        already-deleted QThread C++ object raising RuntimeError) would leave
        the Python process alive in the background after the window closed.
        """
        # Stop processing thread
        try:
            if self.processing_thread is not None and self.processing_thread.isRunning():
                self.processing_thread.stop()
                self.processing_thread.safe_wait(5000)
        except (RuntimeError, Exception):
            pass

        # Stop captioning thread (from caption studio page)
        try:
            if hasattr(self, 'caption_studio_page'):
                studio = self.caption_studio_page
                gen_tab = getattr(studio, 'generate_tab', None)
                if gen_tab:
                    ct = getattr(gen_tab, 'captioning_thread', None)
                    if ct and ct.isRunning():
                        ct.stop()
                        ct.wait(5000)
                    gen_tab._safe_delete_thread()
        except (RuntimeError, Exception):
            pass

        try:
            event.accept()
        except Exception:
            pass
        # Force process exit so no threads/child procs linger. os._exit is a
        # hard kill of THIS process; reached unconditionally via the guards
        # above so the app can never get stuck "running in the background".
        import os
        os._exit(0)
    
    def change_language(self, index):
        """Change UI language"""
        self.current_lang = 'en' if index == 0 else 'tr'
        theme.set_lang(self.current_lang)  # Persist selection
        # Keep both language combos in sync without re-triggering this handler.
        target_idx = 0 if self.current_lang == 'en' else 1
        for combo in (getattr(self, 'lang_combo', None),
                      getattr(self, '_settings_lang_combo', None)):
            if combo is not None and combo.currentIndex() != target_idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(target_idx)
                combo.blockSignals(False)
        self.update_ui_texts()
    
    def update_ui_texts(self):
        """Update all UI texts with current language"""
        self.setWindowTitle(get_text('app_title', self.current_lang))
        
        # Sidebar nav buttons — update section labels too
        for i, lbl in enumerate(self._sidebar_section_labels):
            lbl.setText(get_text('workspace_label' if i == 0 else 'library_label', self.current_lang))

        _nav_labels = [
            'page_video_processing',
            'page_caption_studio',
            'page_character_sort',
            'page_tag_frequency',
        ]
        _step_nums = ['1', '2', '3', '4']
        _ps = getattr(self, '_page_status', {})
        for (btn, key), step in zip(zip(self._nav_buttons, _nav_labels), _step_nums):
            idx = self._nav_buttons.index(btn)
            base = f"{step}  {get_text(key, self.current_lang)}"
            if _ps.get(idx) == 'done':
                base += '  ✅'
            btn.setText(base)

        if hasattr(self, 'page_review_btn'):
            review_base = f"5  {get_text('page_review', self.current_lang)}"
            if _ps.get(4) == 'done':
                review_base += '  ✅'
            self.page_review_btn.setText(review_base)
        if hasattr(self, 'page_upscale_btn'):
            upscale_base = f"6  {get_text('page_upscale', self.current_lang)}"
            if _ps.get(6) == 'done':
                upscale_base += '  ✅'
            self.page_upscale_btn.setText(upscale_base)

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

        # Update review grid page language
        if hasattr(self, 'review_grid_page'):
            self.review_grid_page.update_language(self.current_lang)

        # Update upscale page language
        if hasattr(self, 'upscale_page'):
            self.upscale_page.update_language(self.current_lang)

        # Update training page language
        if hasattr(self, 'training_page'):
            self.training_page.update_language(self.current_lang)

        # Settings + Training nav buttons (not covered by the loop above)
        if hasattr(self, 'page_settings_btn'):
            self.page_settings_btn.setText(get_text('page_settings', self.current_lang))
        if hasattr(self, 'page_training_btn'):
            self.page_training_btn.setText(
                f"7  {get_text('page_training_nav', self.current_lang)}"
            )

        # Video page elements
        self.title_label.setText(get_text('title', self.current_lang))
        self.subtitle_label.setText(get_text('subtitle', self.current_lang))
        self.browse_btn.setText(get_text('browse_btn', self.current_lang))
        self.settings_group.setTitle(get_text('settings_title', self.current_lang))
        self.interval_label.setText(get_text('frame_interval', self.current_lang))
        self.interval_help.setToolTip(get_text('frame_interval_tooltip', self.current_lang))
        self.ratio_label.setText(get_text('output_format', self.current_lang))
        self.ratio_help.setToolTip(get_text('output_format_tooltip', self.current_lang))
        if hasattr(self, 'detection_mode_label'):
            self.detection_mode_label.setText(get_text('detection_mode', self.current_lang))
            self.detection_mode_combo.setItemText(0, get_text('detection_mode_yolo', self.current_lang))
            self.detection_mode_combo.setItemText(1, get_text('detection_mode_anime', self.current_lang))
            self.detection_mode_combo.setItemText(2, get_text('detection_mode_auto', self.current_lang))
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
        if hasattr(self, 'subtitle_removal_cb'):
            self.subtitle_removal_cb.setText(get_text('subtitle_removal', self.current_lang))
            self.subtitle_removal_cb.setToolTip(get_text('subtitle_removal_tooltip', self.current_lang))
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

        # Re-apply translatable tooltips on the video page controls
        self._apply_video_tooltips()

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
        if hasattr(self, 'upscale_panel'):
            self.upscale_panel.update_language(self.current_lang)
        if hasattr(self, 'nsfw_cb'):
            self.nsfw_cb.setText(get_text('nsfw_separation', self.current_lang))
            self.nsfw_cb.setToolTip(get_text('nsfw_separation_tooltip', self.current_lang))

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

    # ── P2: Startup crash recovery ───────────────────────────────────────────

    def _check_crash_log(self):
        import datetime
        log_path = Path(__file__).resolve().parents[2] / "crash_log.txt"
        if not log_path.exists():
            return
        try:
            content = log_path.read_text(encoding='utf-8', errors='ignore')
            log_lines = content.strip().splitlines()
            last_ts = None
            last_error = ""
            for i, line in enumerate(log_lines):
                try:
                    ts = datetime.datetime.strptime(line.strip(), "%Y-%m-%d %H:%M:%S")
                    last_ts = ts
                    last_error = "\n".join(log_lines[i + 1:i + 5])
                except ValueError:
                    pass
            if last_ts is None:
                return
            age = (datetime.datetime.now() - last_ts).total_seconds()
            if age > 86400:   # older than 24 h — skip
                return
            is_oom = any(w in last_error for w in ('CUDA out of memory', 'OOM', 'RuntimeError'))
            self._show_crash_banner(last_error[:120], is_oom)
        except Exception:
            pass

    def _show_crash_banner(self, error_snippet: str, is_oom: bool):
        # Topbar altına kırmızı/turuncu dismissable banner ekle
        banner = QFrame(self)
        banner.setObjectName("crashBanner")
        banner_lay = QHBoxLayout(banner)
        banner_lay.setContentsMargins(16, 8, 16, 8)

        icon = "💾" if is_oom else "⚠️"
        msg = (
            "Son çalışmada bellek hatası (OOM) oluştu. Güvenli Mod öneriliyor."
            if is_oom
            else f"Son çalışmada hata: {error_snippet}"
        )
        lbl = QLabel(f"{icon}  {msg}")
        lbl.setStyleSheet(
            f"color: {'#ff6b6b' if is_oom else '#ffa94d'}; font-size: {theme.fs(11)}; "
            f"font-weight: 600; background: transparent; border: none;"
        )
        banner_lay.addWidget(lbl)
        banner_lay.addStretch()

        if is_oom:
            apply_btn = QPushButton("Güvenli Mod Uygula")
            apply_btn.setStyleSheet(
                f"QPushButton {{ background: #ff6b6b22; color: #ff6b6b; "
                f"border: 1px solid #ff6b6b44; border-radius: 4px; padding: 4px 12px; "
                f"font-size: {theme.fs(11)}; }}"
                f" QPushButton:hover {{ background: #ff6b6b44; }}"
            )
            apply_btn.clicked.connect(lambda: self._apply_safe_mode(banner))
            banner_lay.addWidget(apply_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {theme.TEXT_MUTED}; "
            f"border: none; font-size: {theme.fs(12)}; }}"
            f" QPushButton:hover {{ color: {theme.TEXT_PRIMARY}; }}"
        )
        close_btn.clicked.connect(banner.hide)
        banner_lay.addWidget(close_btn)

        banner.setStyleSheet(
            f"QFrame#crashBanner {{ "
            f"background: {'#ff6b6b11' if is_oom else '#ffa94d11'}; "
            f"border-bottom: 1px solid {'#ff6b6b33' if is_oom else '#ffa94d33'}; "
            f"border-top: none; border-left: none; border-right: none; }}"
        )

        if hasattr(self, '_main_content_layout'):
            self._main_content_layout.insertWidget(1, banner)
        elif self.centralWidget() and self.centralWidget().layout():
            self.centralWidget().layout().insertWidget(1, banner)

    def _apply_safe_mode(self, banner: QFrame):
        from src.ui.resource_settings import load_settings, save_settings
        s = load_settings()
        s['gpu_mem_limit_pct'] = 60
        s['auto_gc'] = True
        save_settings(s)
        banner.hide()
        try:
            ToastNotification(
                "✅ Güvenli Mod uygulandı: VRAM %60 cap + Auto GC",
                icon='✅', duration_ms=3000, accent=theme.get_accent(),
                parent=self.centralWidget(),
            )
        except Exception:
            pass

    # ── P3: Sidebar step-done markers ────────────────────────────────────────

    def _mark_page_done(self, page_idx: int):
        # Sidebar butonuna ✅ işareti ekle
        self._page_status[page_idx] = 'done'
        if page_idx < len(self._nav_buttons):
            btn = self._nav_buttons[page_idx]
            current_text = btn.text()
            if '✅' not in current_text:
                btn.setText(current_text + '  ✅')

    def update_drop_zone_text(self):
        """Update drop zone text"""
        self.drop_zone.setTextFormat(Qt.RichText)
        lang = self.current_lang
        self.drop_zone.lang = lang  # keep DropZone's self-rendered text in sync
        if not self.video_paths:
            self.drop_zone.setText(
                f"<div style='line-height:1.6;'>"
                f"<div style='font-size:28px;'>☁</div>"
                f"<div style='font-size:14px;font-weight:600;color:#f1dfd4;margin:4px 0 2px;'>"
                f"{get_text('drop_zone_idle_title', lang)}</div>"
                f"<div style='font-size:11px;color:#a38c7d;'>{get_text('drop_zone_idle_hint', lang)}</div>"
                f"</div>"
            )
        elif len(self.video_paths) == 1:
            p = self.video_paths[0]
            short = p if len(p) <= 44 else "..." + p[-41:]
            self.drop_zone.setText(
                f"<div style='line-height:1.6;'>"
                f"<div style='font-size:20px;color:#e8832a;'>✓</div>"
                f"<div style='font-size:13px;font-weight:600;color:#f1dfd4;margin:2px 0;'>"
                f"{get_text('drop_zone_one_loaded', lang)}</div>"
                f"<div style='font-size:10px;color:#a38c7d;font-family:monospace;'>{short}</div>"
                f"</div>"
            )
        else:
            self.drop_zone.setText(
                f"<div style='line-height:1.6;'>"
                f"<div style='font-size:20px;color:#e8832a;'>✓</div>"
                f"<div style='font-size:13px;font-weight:600;color:#f1dfd4;margin:2px 0;'>"
                f"{get_text('drop_zone_n_loaded', lang).format(len(self.video_paths))}</div>"
                f"<div style='font-size:10px;color:#a38c7d;'>{get_text('drop_zone_more_hint', lang)}</div>"
                f"</div>"
            )



def create_app():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    _assets = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets'))
    for _name in ('icon.png', 'icon.ico'):
        _p = os.path.join(_assets, _name)
        if os.path.exists(_p):
            app.setWindowIcon(QIcon(_p))
            break
    app.setStyle('Fusion')
    window = VideoSmartCropperUI()
    window.show()
    return app, window
