"""
Character Sort Page for LoRA-Harvester v3.0
Identifies characters by face and organises frames into per-character folders.
Works on any existing image folder — fully independent from video processing.
"""

import os
import shutil
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QPushButton, QProgressBar, QFileDialog, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent
from typing import Dict, List, Optional
from src.ui.translations import get_text
from src.ui import theme


# ─────────────────────────────────────────────────────────────────────────────
# Background thread
# ─────────────────────────────────────────────────────────────────────────────

class CharacterSortThread(QThread):
    """Runs CharacterRecognizer.sort_directory() in a background thread."""

    progress = pyqtSignal(int, int, str)   # current, total, message
    sort_finished = pyqtSignal(dict)       # stats dict (renamed to avoid shadowing QThread.finished)
    error = pyqtSignal(str)
    log_msg = pyqtSignal(str)              # intermediate log messages

    def __init__(
        self,
        input_dir: Optional[str],
        output_dir: Optional[str],
        reference_dir: Optional[str],
        settings: Dict,
        input_files: Optional[List[str]] = None,
        mode: str = 'real',   # 'real' → InsightFace   |   'anime' → ResNet+cascade
        parent=None,
    ):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.reference_dir = reference_dir
        self.settings = settings
        self.input_files = input_files
        self.mode = mode
        self._running = True
        self._temp_dir: Optional[str] = None

    def run(self):
        try:
            s = self.settings

            # ── temp dir for individual files ─────────────────────────────────
            actual_input = self.input_dir
            if self.input_files:
                self._temp_dir = tempfile.mkdtemp(prefix='lora_charsort_')
                self.log_msg.emit(f"📋 Copying {len(self.input_files)} files to temp dir...")
                for src in self.input_files:
                    if self._running:
                        dst = os.path.join(self._temp_dir, os.path.basename(src))
                        if os.path.exists(dst):
                            stem = Path(src).stem
                            suffix = Path(src).suffix
                            dst = os.path.join(self._temp_dir, f"{stem}_{id(src)}{suffix}")
                        shutil.copy2(src, dst)
                actual_input = self._temp_dir

            cluster_eps = 99.0 if s.get('no_cluster') else s.get('cluster_eps', 0.6)
            cluster_min = 99999 if s.get('no_cluster') else s.get('cluster_min', 2)

            def progress_cb(current, total, msg):
                if self._running:
                    self.progress.emit(current, total, msg)

            # ── select recognizer based on mode ───────────────────────────────
            if self.mode == 'anime':
                from src.core.anime_character_recognizer import AnimeCharacterRecognizer
                self.log_msg.emit("🎌 Anime modu: lbpcascade_animeface + ResNet-18")
                recognizer = AnimeCharacterRecognizer(
                    similarity_threshold=s.get('threshold', 0.55),
                    match_margin=s.get('match_margin', 0.05),
                    cluster_eps=cluster_eps if cluster_eps < 90 else 0.55,
                    cluster_min_samples=cluster_min if cluster_min < 9999 else 2,
                    use_gpu=s.get('use_gpu', True),
                    progress_callback=progress_cb,
                    num_workers=s.get('num_workers', 4),
                    use_cache=s.get('use_cache', True),
                    cache_path=s.get('cache_path'),
                )
            else:
                from src.core.character_recognizer import CharacterRecognizer
                self.log_msg.emit("🎭 Gerçek yüz modu: InsightFace " + s.get('model', 'buffalo_l'))
                recognizer = CharacterRecognizer(
                    reference_dir=self.reference_dir if self.reference_dir else None,
                    similarity_threshold=s.get('threshold', 0.45),
                    match_margin=s.get('match_margin', 0.05),
                    cluster_eps=cluster_eps,
                    cluster_min_samples=cluster_min,
                    use_gpu=s.get('use_gpu', True),
                    model_name=s.get('model', 'buffalo_l'),
                    progress_callback=progress_cb,
                    num_workers=s.get('num_workers', 4),
                    use_cache=s.get('use_cache', True),
                    cache_path=s.get('cache_path'),
                )

            if self.reference_dir:
                self.log_msg.emit(f"📚 Loading references from {self.reference_dir} ...")
                counts = recognizer.load_references(self.reference_dir)
                for name, cnt in counts.items():
                    self.log_msg.emit(f"   {name}: {cnt} reference(s)")

            self.log_msg.emit("🔄 Scanning images...")

            try:
                stats = recognizer.sort_directory(
                    input_dir=actual_input,
                    output_dir=self.output_dir if self.output_dir else None,
                    copy=s.get('copy_files', False),
                    recursive=s.get('recursive', True),
                    max_characters=s.get('max_characters', 6),
                    max_per_character=s.get('max_per_character', 0),
                )
            finally:
                recognizer.close_cache()
                self._cleanup_temp()

            if self._running:
                self.sort_finished.emit(stats)

        except ImportError as e:
            self._cleanup_temp()
            self.error.emit(
                f"Eksik bağımlılık: {e}\n"
                "Kur: pip install insightface scikit-learn onnxruntime torchvision"
            )
        except Exception as e:
            import traceback
            self._cleanup_temp()
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _cleanup_temp(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None

    def stop(self):
        self._running = False
        self.requestInterruption()


# ─────────────────────────────────────────────────────────────────────────────
# Drop Zone Frame — proper PyQt5 subclass for folder drag & drop
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class DropZoneFrame(QFrame):
    """
    QFrame subclass for folder OR image-file drag-and-drop.
    Emits folder_dropped(str) for a single folder,
    or files_dropped(list[str]) for one or more image files.
    """

    folder_dropped = pyqtSignal(str)
    files_dropped = pyqtSignal(list)

    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(57)
        self.setStyleSheet(theme.drop_zone_frame_default())

        inner = QHBoxLayout(self)
        inner.setContentsMargins(10, 2, 10, 2)

        self._lbl = QLabel(placeholder)
        self._lbl.setStyleSheet(theme.label_transparent())
        self._lbl.setObjectName("placeholder")
        inner.addWidget(self._lbl)

    def get_label(self) -> QLabel:
        return self._lbl

    def _classify_urls(self, urls):
        """Return ('folder', path) or ('files', [paths]) or None."""
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if not paths:
            return None
        if len(paths) == 1 and os.path.isdir(paths[0]):
            return ('folder', paths[0])
        image_paths = [p for p in paths if Path(p).suffix.lower() in _IMAGE_EXTS]
        if image_paths:
            return ('files', image_paths)
        return None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            result = self._classify_urls(event.mimeData().urls())
            if result:
                event.acceptProposedAction()
                self.setStyleSheet(theme.drop_zone_frame_active())
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(theme.drop_zone_frame_default())

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(theme.drop_zone_frame_default())
        if event.mimeData().hasUrls():
            result = self._classify_urls(event.mimeData().urls())
            if result:
                event.acceptProposedAction()
                kind, payload = result
                if kind == 'folder':
                    self.folder_dropped.emit(payload)
                else:
                    self.files_dropped.emit(payload)
                return
        event.ignore()


# ─────────────────────────────────────────────────────────────────────────────
# Page widget
# ─────────────────────────────────────────────────────────────────────────────

class CharacterSortPage(QWidget):
    """
    GUI page for face-based character recognition and folder sorting.
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    # ── preset definitions ────────────────────────────────────────────────────
    PRESETS = {
        'real': {
            'mode':        'real',
            'model':       'buffalo_l',
            'threshold':   0.45,
            'cluster_eps': 0.60,
            'cluster_min': 2,
            'no_cluster':  False,
        },
        'anime': {
            'mode':        'anime',
            'model':       'anime',       # display label only
            'threshold':   0.55,
            'cluster_eps': 0.55,
            'cluster_min': 2,
            'no_cluster':  False,
        },
    }

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self.input_folder: Optional[str] = None
        self.input_files: Optional[List[str]] = None
        self.ref_folder: Optional[str] = None
        self.output_folder: Optional[str] = None
        self._thread: Optional[CharacterSortThread] = None
        self._mode: str = 'real'           # 'real' | 'anime'
        self.init_ui()

    # ─── UI construction ──────────────────────────────────────────────────────

    def init_ui(self):
        from PyQt5.QtWidgets import QScrollArea, QGridLayout
        root = QVBoxLayout()
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(16)
        self.setLayout(root)

        # ── Hidden compat widgets (referenced by update_language / refresh_styles / _apply_preset) ──
        self.title_lbl    = QLabel(get_text('char_sort_title',    self.lang)); self.title_lbl.hide()
        self.subtitle_lbl = QLabel(get_text('char_sort_subtitle', self.lang)); self.subtitle_lbl.hide()
        self.step1_title  = QLabel(); self.step1_title.hide()
        self.step2_title  = QLabel(); self.step2_title.hide()
        self.step3_title  = QLabel(); self.step3_title.hide()
        self.input_lbl    = QLabel(); self.input_lbl.hide()
        self.ref_lbl      = QLabel(); self.ref_lbl.hide()
        self.out_lbl      = QLabel(); self.out_lbl.hide()
        self.browse_input_btn = QPushButton(); self.browse_input_btn.hide()
        self.browse_files_btn = QPushButton(); self.browse_files_btn.hide()
        self.browse_ref_btn   = QPushButton(); self.browse_ref_btn.hide()
        self.browse_out_btn   = QPushButton(); self.browse_out_btn.hide()
        self.clear_ref_btn    = QPushButton(); self.clear_ref_btn.hide()
        self.clear_out_btn    = QPushButton(); self.clear_out_btn.hide()
        self.input_count_lbl  = QLabel(); self.input_count_lbl.hide()
        self.ref_status_lbl   = QLabel(); self.ref_status_lbl.hide()
        self.out_status_lbl   = QLabel(); self.out_status_lbl.hide()
        self.out_drop = DropZoneFrame(""); self.out_drop.hide()
        self.out_drop.folder_dropped.connect(self._set_output)
        # Hidden settings widgets used by _get_settings / _apply_preset
        self.thresh_spin = QDoubleSpinBox(); self.thresh_spin.setRange(0.20, 0.80); self.thresh_spin.setValue(0.45); self.thresh_spin.setSingleStep(0.05); self.thresh_spin.hide()
        self.eps_spin    = QDoubleSpinBox(); self.eps_spin.setRange(0.20, 1.50); self.eps_spin.setValue(0.60); self.eps_spin.setSingleStep(0.05); self.eps_spin.hide()
        self.min_spin    = QSpinBox(); self.min_spin.setRange(1, 20); self.min_spin.setValue(2); self.min_spin.hide()
        self.no_cluster_cb = QCheckBox(); self.no_cluster_cb.hide()
        self.copy_cb       = QCheckBox(); self.copy_cb.hide()
        self.recursive_cb  = QCheckBox(); self.recursive_cb.setChecked(True); self.recursive_cb.hide()
        self.gpu_cb        = QCheckBox(); self.gpu_cb.setChecked(True); self.gpu_cb.hide()
        self.max_char_slider   = QSlider(Qt.Horizontal); self.max_char_slider.setMinimum(1); self.max_char_slider.setMaximum(6); self.max_char_slider.setValue(6); self.max_char_slider.hide()
        self.max_char_value_lbl = QLabel("6"); self.max_char_value_lbl.hide()
        self.topn_spin     = QSpinBox(); self.topn_spin.setMinimum(0); self.topn_spin.setMaximum(9999); self.topn_spin.setValue(0); self.topn_spin.hide()
        # Hidden labels for update_language / refresh_styles
        self.model_lbl      = QLabel(); self.model_lbl.hide()
        self.thresh_lbl     = QLabel(); self.thresh_lbl.hide()
        self.thresh_lbl_info= QLabel(); self.thresh_lbl_info.hide()
        self.eps_lbl        = QLabel(); self.eps_lbl.hide()
        self.eps_info       = QLabel(); self.eps_info.hide()
        self.min_lbl        = QLabel(); self.min_lbl.hide()
        self.min_info       = QLabel(); self.min_info.hide()
        self.max_char_lbl   = QLabel(); self.max_char_lbl.hide()
        self.max_char_info  = QLabel(); self.max_char_info.hide()
        self._ticks_lbl     = QLabel(); self._ticks_lbl.hide()
        self.topn_lbl       = QLabel(); self.topn_lbl.hide()
        self.topn_info      = QLabel(); self.topn_info.hide()
        # Hidden preset bar widgets for _apply_preset / update_language / refresh_styles
        self._preset_type_lbl  = QLabel(); self._preset_type_lbl.hide()
        self._preset_info_lbl  = QLabel(); self._preset_info_lbl.hide()
        self._preset_real_btn  = QPushButton(); self._preset_real_btn.setCheckable(True); self._preset_real_btn.setChecked(True); self._preset_real_btn.hide()
        self._preset_anime_btn = QPushButton(); self._preset_anime_btn.setCheckable(True); self._preset_anime_btn.hide()

        # ── TOP ROW: Drop zones (left 8) + Settings card (right 4) ──────────
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # LEFT — two drop zone cards side by side
        drops_lay = QHBoxLayout()
        drops_lay.setSpacing(12)

        # Source Images drop zone
        self.input_drop = DropZoneFrame(get_text('char_drag_input', self.lang))
        self.input_drop.setMinimumHeight(160)
        self.input_drop.setStyleSheet(self._drop_zone_card_style())
        self.input_drop.folder_dropped.connect(self._set_input)
        self.input_drop.files_dropped.connect(self._set_input_files)
        # Rebuild DropZoneFrame layout to match mockup card design
        for i in reversed(range(self.input_drop.layout().count())):
            w = self.input_drop.layout().itemAt(i).widget()
            if w:
                w.setParent(None)
        self.input_drop.layout().deleteLater()
        src_lay = QVBoxLayout(self.input_drop)
        src_lay.setAlignment(Qt.AlignCenter); src_lay.setSpacing(6); src_lay.setContentsMargins(16, 12, 16, 12)
        src_icon = QLabel("📁"); src_icon.setAlignment(Qt.AlignCenter)
        src_icon.setStyleSheet("font-size: 28px; background: transparent; border: none;")
        src_title_lbl = QLabel("Source Images"); src_title_lbl.setAlignment(Qt.AlignCenter)
        src_title_lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(15)}; font-weight: 600; background: transparent; border: none;")
        src_desc = QLabel("Drag & drop folder containing images to cluster")
        src_desc.setAlignment(Qt.AlignCenter); src_desc.setWordWrap(True)
        src_desc.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; background: transparent; border: none;")
        self._src_path_lbl = QLabel("No folder selected")
        self._src_path_lbl.setAlignment(Qt.AlignCenter)
        self._src_path_lbl.setStyleSheet(
            f"background: {theme.BG_SURFACE}; border: 1px solid {theme.BORDER}; border-radius: 3px;"
            f" color: {theme.TEXT_SECONDARY}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(10)}; padding: 2px 8px;"
        )
        src_lay.addWidget(src_icon); src_lay.addWidget(src_title_lbl)
        src_lay.addWidget(src_desc); src_lay.addWidget(self._src_path_lbl)
        self.input_drop.mousePressEvent = lambda e: self._browse_input()
        self.input_drop.setCursor(Qt.PointingHandCursor)
        drops_lay.addWidget(self.input_drop)

        # Reference Faces drop zone
        self.ref_drop = DropZoneFrame(get_text('char_drag_ref', self.lang))
        self.ref_drop.setMinimumHeight(160)
        self.ref_drop.setStyleSheet(self._drop_zone_card_style())
        self.ref_drop.folder_dropped.connect(self._set_ref)
        for i in reversed(range(self.ref_drop.layout().count())):
            w = self.ref_drop.layout().itemAt(i).widget()
            if w:
                w.setParent(None)
        self.ref_drop.layout().deleteLater()
        ref_lay = QVBoxLayout(self.ref_drop)
        ref_lay.setAlignment(Qt.AlignCenter); ref_lay.setSpacing(6); ref_lay.setContentsMargins(16, 12, 16, 12)
        ref_icon = QLabel("👤"); ref_icon.setAlignment(Qt.AlignCenter)
        ref_icon.setStyleSheet("font-size: 28px; background: transparent; border: none;")
        ref_title_lbl = QLabel("Reference Faces (Optional)"); ref_title_lbl.setAlignment(Qt.AlignCenter)
        ref_title_lbl.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(15)}; font-weight: 600; background: transparent; border: none;")
        ref_desc = QLabel("Drag & drop folder with known character subfolders")
        ref_desc.setAlignment(Qt.AlignCenter); ref_desc.setWordWrap(True)
        ref_desc.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; background: transparent; border: none;")
        self._ref_path_lbl = QLabel("No reference folder loaded")
        self._ref_path_lbl.setAlignment(Qt.AlignCenter)
        self._ref_path_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; font-style: italic; background: transparent; border: none;"
        )
        ref_lay.addWidget(ref_icon); ref_lay.addWidget(ref_title_lbl)
        ref_lay.addWidget(ref_desc); ref_lay.addWidget(self._ref_path_lbl)
        self.ref_drop.mousePressEvent = lambda e: self._browse_ref()
        self.ref_drop.setCursor(Qt.PointingHandCursor)
        drops_lay.addWidget(self.ref_drop)

        drops_w = QWidget(); drops_w.setStyleSheet("background: transparent;")
        drops_w.setLayout(drops_lay)
        top_row.addWidget(drops_w, stretch=8)

        # RIGHT — Clustering Settings card
        sc = QFrame()
        sc.setStyleSheet(
            f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER}; border-radius: 10px; }}"
        )
        sc_lay = QVBoxLayout(sc)
        sc_lay.setContentsMargins(16, 14, 16, 14)
        sc_lay.setSpacing(10)

        sc_hdr = QHBoxLayout()
        sc_title_lbl = QLabel("⚙  Clustering Settings")
        sc_title_lbl.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(14)}; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        sc_hdr.addWidget(sc_title_lbl); sc_hdr.addStretch()
        sc_lay.addLayout(sc_hdr)

        # InsightFace Model
        sc_lay.addWidget(self._settings_label("INSIGHTFACE MODEL"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("buffalo_l (ResNet50) - Balanced",          "buffalo_l")
        self.model_combo.addItem("antelopev2 (ResNet100) - High Accuracy",   "antelopev2")
        self.model_combo.setStyleSheet(theme.combo())
        sc_lay.addWidget(self.model_combo)

        # Similarity Threshold slider
        th_hdr = QHBoxLayout()
        th_hdr.addWidget(self._settings_label("SIMILARITY THRESHOLD"))
        th_hdr.addStretch()
        self._thresh_val_lbl = QLabel("0.65")
        self._thresh_val_lbl.setStyleSheet(
            f"color: {theme.ORANGE}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(11)};"
            f" background: transparent; border: none;"
        )
        th_hdr.addWidget(self._thresh_val_lbl)
        sc_lay.addLayout(th_hdr)
        self._thresh_sl = QSlider(Qt.Horizontal)
        self._thresh_sl.setMinimum(10); self._thresh_sl.setMaximum(90); self._thresh_sl.setValue(65)
        self._thresh_sl.setStyleSheet(self._slider_style())
        self._thresh_sl.valueChanged.connect(lambda v: self._thresh_val_lbl.setText(f"{v/100:.2f}"))
        sc_lay.addWidget(self._thresh_sl)

        # Max Faces + Domain side by side
        mid_row = QHBoxLayout(); mid_row.setSpacing(12)
        max_col = QVBoxLayout()
        max_col.addWidget(self._settings_label("MAX FACES/IMG"))
        self._max_faces_spin = QSpinBox()
        self._max_faces_spin.setRange(1, 10); self._max_faces_spin.setValue(3)
        self._max_faces_spin.setStyleSheet(theme.spinbox())
        max_col.addWidget(self._max_faces_spin)
        mid_row.addLayout(max_col)
        domain_col = QVBoxLayout()
        domain_col.addWidget(self._settings_label("DOMAIN"))
        domain_toggle = QFrame()
        domain_toggle.setStyleSheet(
            f"QFrame {{ background: {theme.BG_DEEPEST}; border: 1px solid {theme.BORDER}; border-radius: 4px; }}"
        )
        dtl = QHBoxLayout(domain_toggle); dtl.setContentsMargins(2, 2, 2, 2); dtl.setSpacing(0)
        self._domain_real_btn  = QPushButton("Real")
        self._domain_anime_btn = QPushButton("Anime")
        self._domain_real_btn.setStyleSheet(self._domain_btn_style(True))
        self._domain_anime_btn.setStyleSheet(self._domain_btn_style(False))
        self._domain_real_btn.clicked.connect(lambda: self._set_domain('real'))
        self._domain_anime_btn.clicked.connect(lambda: self._set_domain('anime'))
        dtl.addWidget(self._domain_real_btn); dtl.addWidget(self._domain_anime_btn)
        domain_col.addWidget(domain_toggle)
        mid_row.addLayout(domain_col)
        sc_lay.addLayout(mid_row)

        # DBSCAN Min Samples
        db_row = QHBoxLayout()
        db_row.addWidget(self._settings_label("DBSCAN MIN SAMPLES"))
        db_row.addStretch()
        self._dbscan_spin = QSpinBox()
        self._dbscan_spin.setRange(2, 20); self._dbscan_spin.setValue(5)
        self._dbscan_spin.setFixedWidth(64)
        self._dbscan_spin.setStyleSheet(theme.spinbox())
        db_row.addWidget(self._dbscan_spin)
        sc_lay.addLayout(db_row)

        sc_lay.addStretch()

        div = QFrame(); div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"background: {theme.BORDER}; border: none;"); div.setFixedHeight(1)
        sc_lay.addWidget(div)

        self.start_btn = QPushButton("▶  Start Clustering")
        self.start_btn.setEnabled(False)
        self.start_btn.setFixedHeight(36)
        self.start_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.ORANGE}; color: #1a1a1a; font-weight: 700;"
            f" font-size: {theme.fs(13)}; border: none; border-radius: 6px; }}"
            f"QPushButton:hover {{ background: {theme.ORANGE_LIGHT}; }}"
            f"QPushButton:disabled {{ background: {theme.BG_SURFACE}; color: {theme.TEXT_MUTED}; }}"
        )
        self.start_btn.clicked.connect(self._start)
        sc_lay.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.stop_btn.clicked.connect(self._stop)
        sc_lay.addWidget(self.stop_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(theme.progress_bar()); self.progress_bar.setValue(0)
        sc_lay.addWidget(self.progress_bar)

        top_row.addWidget(sc, stretch=4)
        root.addLayout(top_row)

        # ── CLUSTER RESULTS ──────────────────────────────────────────────────
        res_hdr = QHBoxLayout()
        self._results_title = QLabel("Cluster Results")
        self._results_title.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(18)}; font-weight: 700;"
            f" letter-spacing: -0.02em; background: transparent; border: none;"
        )
        self._cluster_count_badge = QLabel("0 Total")
        self._cluster_count_badge.setStyleSheet(
            f"background: {theme.BG_SURFACE}; color: {theme.TEXT_SECONDARY};"
            f" font-family: {theme.FONT_MONO}; font-size: {theme.fs(11)};"
            f" border-radius: 10px; padding: 2px 10px; border: none;"
        )
        res_hdr.addWidget(self._results_title); res_hdr.addWidget(self._cluster_count_badge); res_hdr.addStretch()

        # View toggle buttons (grid / list)
        _toggle_style_active = (
            f"QPushButton {{ background: {theme.BG_SURFACE}; border: 1px solid {theme.BORDER};"
            f" border-radius: 4px; color: {theme.TEXT_PRIMARY}; padding: 4px 8px; font-size: 14px; }}"
        )
        _toggle_style_inactive = (
            f"QPushButton {{ background: transparent; border: 1px solid {theme.BORDER};"
            f" border-radius: 4px; color: {theme.TEXT_MUTED}; padding: 4px 8px; font-size: 14px; }}"
            f"QPushButton:hover {{ color: {theme.TEXT_SECONDARY}; border-color: {theme.BORDER_LIGHT}; }}"
        )
        self._view_grid_btn = QPushButton("⊞")
        self._view_grid_btn.setFixedSize(30, 26)
        self._view_grid_btn.setStyleSheet(_toggle_style_active)
        self._view_list_btn = QPushButton("☰")
        self._view_list_btn.setFixedSize(30, 26)
        self._view_list_btn.setStyleSheet(_toggle_style_inactive)
        self._view_grid_btn.clicked.connect(lambda: [
            self._view_grid_btn.setStyleSheet(_toggle_style_active),
            self._view_list_btn.setStyleSheet(_toggle_style_inactive),
        ])
        self._view_list_btn.clicked.connect(lambda: [
            self._view_list_btn.setStyleSheet(_toggle_style_active),
            self._view_grid_btn.setStyleSheet(_toggle_style_inactive),
        ])
        res_hdr.addWidget(self._view_grid_btn)
        res_hdr.addWidget(self._view_list_btn)
        root.addLayout(res_hdr)

        hline = QFrame(); hline.setFrameShape(QFrame.HLine)
        hline.setStyleSheet(f"background: {theme.BORDER}; border: none;"); hline.setFixedHeight(1)
        root.addWidget(hline)

        cluster_scroll = QScrollArea()
        cluster_scroll.setWidgetResizable(True); cluster_scroll.setMinimumHeight(180)
        cluster_scroll.setStyleSheet("background: transparent; border: none;")
        self._cluster_inner = QWidget(); self._cluster_inner.setStyleSheet("background: transparent;")
        self._cluster_grid_lay = QGridLayout(self._cluster_inner); self._cluster_grid_lay.setSpacing(10)
        self._cluster_grid_lay.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._cluster_placeholder = QLabel("Run clustering to see results here.")
        self._cluster_placeholder.setAlignment(Qt.AlignCenter)
        self._cluster_placeholder.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(13)}; background: transparent; border: none;"
        )
        self._cluster_grid_lay.addWidget(self._cluster_placeholder, 0, 0)
        cluster_scroll.setWidget(self._cluster_inner)
        root.addWidget(cluster_scroll, stretch=1)

        # ── LOG TERMINAL ─────────────────────────────────────────────────────
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(160)
        self.log_text.setStyleSheet(theme.log_area())
        root.addWidget(self.log_text)

    # ── New UI helper methods ──────────────────────────────────────────────────

    def _settings_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-family: {theme.FONT_MONO}; font-size: {theme.fs(10)};"
            f" font-weight: 500; letter-spacing: 0.06em; background: transparent; border: none;"
        )
        return lbl

    def _drop_zone_card_style(self) -> str:
        return (
            f"QFrame {{ background: {theme.BG_SURFACE}; border: 1px dashed {theme.BORDER};"
            f" border-radius: 8px; }}"
            f"QFrame:hover {{ border-color: {theme.ORANGE}; background: {theme.BG_CARD}; }}"
        )

    def _slider_style(self) -> str:
        return (
            f"QSlider::groove:horizontal {{ height: 4px; background: {theme.BORDER}; border-radius: 2px; }}"
            f"QSlider::handle:horizontal {{ width: 14px; height: 14px; background: {theme.ORANGE};"
            f" border-radius: 7px; margin: -5px 0; }}"
            f"QSlider::sub-page:horizontal {{ background: {theme.ORANGE}; border-radius: 2px; }}"
        )

    def _domain_btn_style(self, active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background: {theme.ORANGE}22; color: {theme.ORANGE};"
                f" border: 1px solid {theme.ORANGE}44; border-radius: 3px;"
                f" font-size: {theme.fs(11)}; padding: 3px 10px; }}"
            )
        return (
            f"QPushButton {{ background: transparent; color: {theme.TEXT_MUTED};"
            f" border: none; border-radius: 3px; font-size: {theme.fs(11)}; padding: 3px 10px; }}"
            f"QPushButton:hover {{ color: {theme.TEXT_SECONDARY}; }}"
        )

    def _set_domain(self, mode: str):
        self._mode = mode
        is_real = (mode == 'real')
        self._domain_real_btn.setStyleSheet(self._domain_btn_style(is_real))
        self._domain_anime_btn.setStyleSheet(self._domain_btn_style(not is_real))
        self._preset_real_btn.setChecked(is_real)
        self._preset_anime_btn.setChecked(not is_real)
        preset = self.PRESETS.get(mode, self.PRESETS['real'])
        self.thresh_spin.setValue(preset['threshold'])
        self.eps_spin.setValue(preset['cluster_eps'])
        self.min_spin.setValue(preset['cluster_min'])
        self._thresh_sl.setValue(int(preset['threshold'] * 100))
        self._dbscan_spin.setValue(preset['cluster_min'])
        self.model_combo.setEnabled(is_real)

    # ── Preset bar ────────────────────────────────────────────────────────────

    def _build_preset_bar(self) -> QFrame:
        """Two large preset buttons: Gerçek Yüz and Anime."""
        frame = QFrame()
        frame.setStyleSheet(theme.card_frame())
        lay = QHBoxLayout(frame)
        lay.setSpacing(12)

        self._preset_type_lbl = QLabel("Model Tipi:")
        self._preset_type_lbl.setStyleSheet(theme.label_default())
        self._preset_type_lbl.setFixedWidth(80)
        lay.addWidget(self._preset_type_lbl)

        self._preset_real_btn = QPushButton("🎭  Gerçek Yüz")
        self._preset_real_btn.setCheckable(True)
        self._preset_real_btn.setChecked(True)
        self._preset_real_btn.setFixedHeight(38)
        self._preset_real_btn.setStyleSheet(self._preset_btn_style(active=True))
        self._preset_real_btn.clicked.connect(lambda: self._apply_preset('real'))
        lay.addWidget(self._preset_real_btn)

        self._preset_anime_btn = QPushButton("🎌  Anime")
        self._preset_anime_btn.setCheckable(True)
        self._preset_anime_btn.setChecked(False)
        self._preset_anime_btn.setFixedHeight(38)
        self._preset_anime_btn.setStyleSheet(self._preset_btn_style(active=False))
        self._preset_anime_btn.clicked.connect(lambda: self._apply_preset('anime'))
        lay.addWidget(self._preset_anime_btn)

        # Info label — changes based on active preset
        self._preset_info_lbl = QLabel("InsightFace buffalo_l — gerçek insan yüzleri için")
        self._preset_info_lbl.setStyleSheet(theme.label_muted())
        lay.addWidget(self._preset_info_lbl, stretch=1)

        return frame

    @staticmethod
    def _preset_btn_style(active: bool) -> str:
        accent = theme.get_accent() if hasattr(theme, 'get_accent') else theme.ORANGE_LIGHT
        if active:
            return (
                f"QPushButton {{ background:{accent}; color:#fff; border:none; "
                f"border-radius:6px; font-weight:bold; padding:4px 16px; }}"
                f"QPushButton:hover {{ background:{accent}; }}"
            )
        return (
            f"QPushButton {{ background:transparent; color:{theme.TEXT_MUTED}; "
            f"border:1px solid {theme.TEXT_MUTED}; border-radius:6px; padding:4px 16px; }}"
            f"QPushButton:hover {{ border-color:{accent}; color:{accent}; }}"
        )

    def _apply_preset(self, key: str):
        """Apply a preset: update mode, toggle button styles, apply settings."""
        preset = self.PRESETS.get(key, self.PRESETS['real'])
        self._mode = preset['mode']

        # Toggle button styles
        self._preset_real_btn.setChecked(key == 'real')
        self._preset_anime_btn.setChecked(key == 'anime')
        self._preset_real_btn.setStyleSheet(self._preset_btn_style(key == 'real'))
        self._preset_anime_btn.setStyleSheet(self._preset_btn_style(key == 'anime'))

        # Update info label
        if key == 'anime':
            self._preset_info_lbl.setText(
                "lbpcascade_animeface + ResNet-18 — anime/çizgi film karakterleri için"
            )
            # Anime preset: hide real-face model combo, show note
            self.model_combo.setEnabled(False)
            self.model_combo.setToolTip("Anime modunda model seçimi kullanılmaz")
        else:
            self._preset_info_lbl.setText(
                "InsightFace buffalo_l — gerçek insan yüzleri için"
            )
            self.model_combo.setEnabled(True)
            self.model_combo.setToolTip(get_text('char_model_tooltip', self.lang))

        # Apply numerical settings
        self.thresh_spin.setValue(preset['threshold'])
        self.eps_spin.setValue(preset['cluster_eps'])
        self.min_spin.setValue(preset['cluster_min'])
        self.no_cluster_cb.setChecked(preset['no_cluster'])

        self.log(f"✅ Preset uygulandı: {'🎌 Anime' if key == 'anime' else '🎭 Gerçek Yüz'}")

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def _build_folder_step(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(theme.card_frame())
        lay = QVBoxLayout(frame)

        self.step1_title = QLabel(get_text('step1_select_folder', self.lang))
        self.step1_title.setFont(QFont('Inter', 14, QFont.Bold))
        self.step1_title.setStyleSheet(theme.label_section())
        lay.addWidget(self.step1_title)

        # Input folder row
        self.input_drop = DropZoneFrame(get_text('char_drag_input', self.lang))
        self.input_drop.folder_dropped.connect(self._set_input)
        self.input_drop.files_dropped.connect(self._set_input_files)

        self.input_lbl = QLabel(get_text('char_input_folder', self.lang))
        self.input_lbl.setStyleSheet(theme.label_default())

        self.browse_input_btn = QPushButton(get_text('char_browse_input', self.lang))
        self.browse_input_btn.setStyleSheet(theme.btn_browse())
        self.browse_input_btn.clicked.connect(self._browse_input)

        self.browse_files_btn = QPushButton("🖼️ Görseller")
        self.browse_files_btn.setStyleSheet(theme.btn_browse())
        self.browse_files_btn.setToolTip("Birden fazla görsel dosyası seç")
        self.browse_files_btn.clicked.connect(self._browse_input_files)

        row_in = QHBoxLayout()
        row_in.addWidget(self.input_lbl)
        row_in.addWidget(self.input_drop, stretch=1)
        row_in.addWidget(self.browse_input_btn)
        row_in.addWidget(self.browse_files_btn)
        lay.addLayout(row_in)

        self.input_count_lbl = QLabel("")
        self.input_count_lbl.setStyleSheet(theme.label_success())
        lay.addWidget(self.input_count_lbl)

        # Reference folder row (optional)
        self.ref_drop = DropZoneFrame(get_text('char_drag_ref', self.lang))
        self.ref_drop.folder_dropped.connect(self._set_ref)

        self.ref_lbl = QLabel(get_text('char_ref_folder', self.lang))
        self.ref_lbl.setStyleSheet(theme.label_default())

        self.browse_ref_btn = QPushButton(get_text('char_browse_ref', self.lang))
        self.browse_ref_btn.setStyleSheet(theme.btn_browse())
        self.browse_ref_btn.clicked.connect(self._browse_ref)

        self.clear_ref_btn = QPushButton("🗑️")
        self.clear_ref_btn.setFixedWidth(44)
        self.clear_ref_btn.setStyleSheet(theme.btn_danger())
        self.clear_ref_btn.setToolTip(get_text('char_clear_ref', self.lang))
        self.clear_ref_btn.clicked.connect(self._clear_ref)
        self.clear_ref_btn.setVisible(False)

        row_ref = QHBoxLayout()
        row_ref.addWidget(self.ref_lbl)
        row_ref.addWidget(self.ref_drop, stretch=1)
        row_ref.addWidget(self.browse_ref_btn)
        row_ref.addWidget(self.clear_ref_btn)
        lay.addLayout(row_ref)

        self.ref_status_lbl = QLabel(get_text('char_ref_optional', self.lang))
        self.ref_status_lbl.setStyleSheet(theme.label_muted())
        lay.addWidget(self.ref_status_lbl)

        # Output folder row (optional)
        self.out_drop = DropZoneFrame(get_text('char_drag_output', self.lang))
        self.out_drop.folder_dropped.connect(self._set_output)

        self.out_lbl = QLabel(get_text('char_output_folder', self.lang))
        self.out_lbl.setStyleSheet(theme.label_default())

        self.browse_out_btn = QPushButton(get_text('char_browse_output', self.lang))
        self.browse_out_btn.setStyleSheet(theme.btn_browse())
        self.browse_out_btn.clicked.connect(self._browse_output)

        self.clear_out_btn = QPushButton("🗑️")
        self.clear_out_btn.setFixedWidth(44)
        self.clear_out_btn.setStyleSheet(theme.btn_danger())
        self.clear_out_btn.setToolTip(get_text('char_clear_output', self.lang))
        self.clear_out_btn.clicked.connect(self._clear_output)
        self.clear_out_btn.setVisible(False)

        row_out = QHBoxLayout()
        row_out.addWidget(self.out_lbl)
        row_out.addWidget(self.out_drop, stretch=1)
        row_out.addWidget(self.browse_out_btn)
        row_out.addWidget(self.clear_out_btn)
        lay.addLayout(row_out)

        self.out_status_lbl = QLabel(get_text('char_output_default', self.lang))
        self.out_status_lbl.setStyleSheet(theme.label_muted())
        lay.addWidget(self.out_status_lbl)

        return frame

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def _build_settings_step(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(theme.card_frame())
        lay = QVBoxLayout(frame)
        lay.setSpacing(8)

        self.step2_title = QLabel(get_text('step2_settings', self.lang))
        self.step2_title.setFont(QFont('Inter', 14, QFont.Bold))
        self.step2_title.setStyleSheet(theme.label_section())
        lay.addWidget(self.step2_title)

        # Row 1: model + threshold
        row1 = QHBoxLayout()

        self.model_lbl = QLabel(get_text('char_model', self.lang))
        self.model_lbl.setStyleSheet(theme.label_frame())
        self.model_combo = QComboBox()
        self.model_combo.addItems(['buffalo_l', 'buffalo_m', 'buffalo_s', 'antelopev2'])
        self.model_combo.setStyleSheet(theme.combo())
        self.model_combo.setToolTip(get_text('char_model_tooltip', self.lang))

        self.thresh_lbl = QLabel(get_text('char_threshold', self.lang))
        self.thresh_lbl.setStyleSheet(theme.label_frame())
        self.thresh_lbl_info = QLabel("ℹ️")
        self.thresh_lbl_info.setStyleSheet(theme.info_icon_frame())
        self.thresh_lbl_info.setToolTip(get_text('char_threshold_tooltip', self.lang))
        self.thresh_lbl_info.setCursor(Qt.WhatsThisCursor)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.20, 0.80)
        self.thresh_spin.setValue(0.45)
        self.thresh_spin.setSingleStep(0.05)
        self.thresh_spin.setStyleSheet(theme.spinbox())

        row1.addWidget(self.model_lbl)
        row1.addWidget(self.model_combo)
        row1.addSpacing(20)
        row1.addWidget(self.thresh_lbl)
        row1.addWidget(self.thresh_lbl_info)
        row1.addWidget(self.thresh_spin)
        row1.addStretch()
        lay.addLayout(row1)

        # Row 2: cluster eps + min
        row2 = QHBoxLayout()

        self.eps_lbl = QLabel(get_text('char_cluster_eps', self.lang))
        self.eps_lbl.setStyleSheet(theme.label_frame())
        self.eps_info = QLabel("ℹ️")
        self.eps_info.setStyleSheet(theme.info_icon_frame())
        self.eps_info.setToolTip(get_text('char_cluster_eps_tooltip', self.lang))
        self.eps_info.setCursor(Qt.WhatsThisCursor)
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.20, 1.50)
        self.eps_spin.setValue(0.60)
        self.eps_spin.setSingleStep(0.05)
        self.eps_spin.setStyleSheet(theme.spinbox())

        self.min_lbl = QLabel(get_text('char_cluster_min', self.lang))
        self.min_lbl.setStyleSheet(theme.label_frame())
        self.min_info = QLabel("ℹ️")
        self.min_info.setStyleSheet(theme.info_icon_frame())
        self.min_info.setToolTip(get_text('char_cluster_min_tooltip', self.lang))
        self.min_info.setCursor(Qt.WhatsThisCursor)
        self.min_spin = QSpinBox()
        self.min_spin.setRange(1, 20)
        self.min_spin.setValue(2)
        self.min_spin.setStyleSheet(theme.spinbox())

        row2.addWidget(self.eps_lbl)
        row2.addWidget(self.eps_info)
        row2.addWidget(self.eps_spin)
        row2.addSpacing(20)
        row2.addWidget(self.min_lbl)
        row2.addWidget(self.min_info)
        row2.addWidget(self.min_spin)
        row2.addStretch()
        lay.addLayout(row2)

        # Row 2b: max characters slider
        row_max = QHBoxLayout()

        self.max_char_lbl = QLabel(get_text('char_max_characters', self.lang))
        self.max_char_lbl.setStyleSheet(theme.label_frame())
        self.max_char_info = QLabel("ℹ️")
        self.max_char_info.setStyleSheet(theme.info_icon_frame())
        self.max_char_info.setToolTip(get_text('char_max_characters_tooltip', self.lang))
        self.max_char_info.setCursor(Qt.WhatsThisCursor)

        self.max_char_slider = QSlider(Qt.Horizontal)
        self.max_char_slider.setMinimum(1)
        self.max_char_slider.setMaximum(6)
        self.max_char_slider.setValue(6)
        self.max_char_slider.setTickPosition(QSlider.TicksBelow)
        self.max_char_slider.setTickInterval(1)
        self.max_char_slider.setFixedWidth(180)
        self.max_char_slider.setStyleSheet(theme.slider())

        self.max_char_value_lbl = QLabel("6")
        self.max_char_value_lbl.setFixedWidth(20)
        self.max_char_value_lbl.setStyleSheet(theme.label_value())
        self.max_char_slider.valueChanged.connect(
            lambda v: self.max_char_value_lbl.setText(str(v))
        )

        # Tick labels
        ticks_lbl = QLabel("1  2  3  4  5  6")
        ticks_lbl.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)};")
        self._ticks_lbl = ticks_lbl

        row_max.addWidget(self.max_char_lbl)
        row_max.addWidget(self.max_char_info)
        row_max.addWidget(self.max_char_slider)
        row_max.addWidget(self.max_char_value_lbl)
        row_max.addSpacing(10)
        row_max.addWidget(ticks_lbl)
        row_max.addStretch()
        lay.addLayout(row_max)

        # Row 2b: max per character (quality-ranked top-N trim)
        row_topn = QHBoxLayout()
        self.topn_lbl = QLabel(get_text('char_max_per_char', self.lang))
        self.topn_lbl.setStyleSheet(theme.label_frame())
        self.topn_info = QLabel("ℹ️")
        self.topn_info.setStyleSheet(theme.info_icon_frame())
        self.topn_info.setToolTip(get_text('char_max_per_char_tooltip', self.lang))
        self.topn_info.setCursor(Qt.WhatsThisCursor)
        self.topn_spin = QSpinBox()
        self.topn_spin.setMinimum(0)
        self.topn_spin.setMaximum(9999)
        self.topn_spin.setValue(0)
        self.topn_spin.setSpecialValueText("off")
        self.topn_spin.setStyleSheet(theme.spinbox() if hasattr(theme, 'spinbox') else "")
        row_topn.addWidget(self.topn_lbl)
        row_topn.addWidget(self.topn_info)
        row_topn.addWidget(self.topn_spin)
        row_topn.addStretch()
        lay.addLayout(row_topn)

        # Row 3: checkboxes
        row3 = QHBoxLayout()

        self.no_cluster_cb = QCheckBox(get_text('char_no_cluster', self.lang))
        self.no_cluster_cb.setToolTip(get_text('char_no_cluster_tooltip', self.lang))
        self.no_cluster_cb.toggled.connect(self._on_no_cluster_toggled)

        self.copy_cb = QCheckBox(get_text('char_copy_files', self.lang))
        self.copy_cb.setToolTip(get_text('char_copy_tooltip', self.lang))

        self.recursive_cb = QCheckBox(get_text('recursive_search', self.lang))
        self.recursive_cb.setChecked(True)
        self.recursive_cb.setToolTip(get_text('recursive_tooltip', self.lang))

        self.gpu_cb = QCheckBox(get_text('char_use_gpu', self.lang))
        self.gpu_cb.setChecked(True)
        self.gpu_cb.setToolTip(get_text('char_use_gpu_tooltip', self.lang))

        row3.addWidget(self.no_cluster_cb)
        row3.addWidget(self.copy_cb)
        row3.addWidget(self.recursive_cb)
        row3.addWidget(self.gpu_cb)
        row3.addStretch()
        lay.addLayout(row3)

        return frame

    # ── Step 3 ────────────────────────────────────────────────────────────────

    def _build_start_step(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(theme.card_frame())
        lay = QVBoxLayout(frame)

        self.step3_title = QLabel(get_text('step3_start', self.lang))
        self.step3_title.setFont(QFont('Inter', 14, QFont.Bold))
        self.step3_title.setStyleSheet(theme.label_section())
        lay.addWidget(self.step3_title)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton(get_text('char_start_btn', self.lang))
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(theme.btn_primary())
        self.start_btn.clicked.connect(self._start)

        self.stop_btn = QPushButton(get_text('stop_btn', self.lang))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.stop_btn.clicked.connect(self._stop)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(theme.progress_bar())
        lay.addWidget(self.progress_bar)

        return frame

    # ─── Folder selection ─────────────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, get_text('char_browse_input', self.lang))
        if folder:
            self._set_input(folder)

    def _browse_input_files(self):
        exts = "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
        files, _ = QFileDialog.getOpenFileNames(self, "Görsel dosyaları seç", "", exts)
        if files:
            self._set_input_files(files)

    def _browse_ref(self):
        folder = QFileDialog.getExistingDirectory(self, get_text('char_browse_ref', self.lang))
        if folder:
            self._set_ref(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, get_text('char_browse_output', self.lang))
        if folder:
            self._set_output(folder)

    def _set_input(self, path: str):
        self.input_folder = path
        self.input_files = None
        display = path if len(path) <= 42 else "..." + path[-39:]
        if hasattr(self, '_src_path_lbl'):
            self._src_path_lbl.setText(display)
            self._src_path_lbl.setStyleSheet(
                f"background: {theme.BG_SURFACE}; border: 1px solid {theme.ORANGE}44;"
                f" border-radius: 3px; color: {theme.ORANGE_LIGHT};"
                f" font-family: {theme.FONT_MONO}; font-size: {theme.fs(10)}; padding: 2px 8px;"
            )
            self.input_drop.setStyleSheet(
                f"QFrame {{ background: {theme.BG_SURFACE}; border: 1px dashed {theme.ORANGE};"
                f" border-radius: 8px; }}"
            )
        self.input_drop.setToolTip(path)
        count = self._count_images_recursive(path)
        self.input_count_lbl.setText(get_text('images_found', self.lang).format(count))
        self.start_btn.setEnabled(True)
        self.log(f"📂 Input: {path}  ({count} images)")

    def _set_input_files(self, files: List[str]):
        self.input_files = files
        self.input_folder = None
        display = f"{len(files)} image(s) selected"
        if hasattr(self, '_src_path_lbl'):
            self._src_path_lbl.setText(display)
            self._src_path_lbl.setStyleSheet(
                f"background: {theme.BG_SURFACE}; border: 1px solid {theme.ORANGE}44;"
                f" border-radius: 3px; color: {theme.ORANGE_LIGHT};"
                f" font-family: {theme.FONT_MONO}; font-size: {theme.fs(10)}; padding: 2px 8px;"
            )
        self.input_count_lbl.setText(get_text('images_found', self.lang).format(len(files)))
        self.start_btn.setEnabled(True)
        self.log(f"🖼️ {len(files)} görsel seçildi")

    def _set_ref(self, path: str):
        self.ref_folder = path
        display = path if len(path) <= 42 else "..." + path[-39:]
        if hasattr(self, '_ref_path_lbl'):
            self._ref_path_lbl.setText(display)
            self._ref_path_lbl.setStyleSheet(
                f"color: {theme.ORANGE_LIGHT}; font-size: {theme.fs(10)};"
                f" font-style: normal; background: transparent; border: none;"
            )
            self.ref_drop.setStyleSheet(
                f"QFrame {{ background: {theme.BG_SURFACE}; border: 1px dashed {theme.ORANGE};"
                f" border-radius: 8px; }}"
            )
        self.ref_drop.setToolTip(path)
        char_dirs = [d for d in Path(path).iterdir() if d.is_dir()]
        self.ref_status_lbl.setText(
            get_text('char_ref_loaded', self.lang).format(len(char_dirs))
        )
        self.log(f"📚 References: {path}  ({len(char_dirs)} character folder(s))")

    def _set_output(self, path: str):
        self.output_folder = path
        lbl = self.out_drop.get_label()
        display = path if len(path) <= 50 else "..." + path[-47:]
        lbl.setText(display)
        lbl.setStyleSheet(theme.label_success())
        self.out_drop.setStyleSheet(theme.drop_zone_frame_success())
        self.out_drop.setToolTip(path)
        self.clear_out_btn.setVisible(True)
        self.out_status_lbl.setText(get_text('char_output_set', self.lang))
        self.out_status_lbl.setStyleSheet(theme.label_success())
        self.log(f"📁 Output: {path}")

    def _clear_ref(self):
        self.ref_folder = None
        lbl = self.ref_drop.get_label()
        lbl.setText(get_text('char_drag_ref', self.lang))
        lbl.setStyleSheet(theme.label_transparent())
        self.ref_drop.setStyleSheet(theme.drop_zone_frame_default())
        self.ref_drop.setToolTip("")
        self.clear_ref_btn.setVisible(False)
        self.ref_status_lbl.setText(get_text('char_ref_optional', self.lang))
        self.ref_status_lbl.setStyleSheet(theme.label_muted())

    def _clear_output(self):
        self.output_folder = None
        lbl = self.out_drop.get_label()
        lbl.setText(get_text('char_drag_output', self.lang))
        lbl.setStyleSheet(theme.label_transparent())
        self.out_drop.setStyleSheet(theme.drop_zone_frame_default())
        self.out_drop.setToolTip("")
        self.clear_out_btn.setVisible(False)
        self.out_status_lbl.setText(get_text('char_output_default', self.lang))
        self.out_status_lbl.setStyleSheet(theme.label_muted())

    # ─── Settings helpers ─────────────────────────────────────────────────────

    def _on_no_cluster_toggled(self, checked: bool):
        self.eps_spin.setEnabled(not checked)
        self.min_spin.setEnabled(not checked)

    def _get_settings(self) -> Dict:
        # Read from new visible widgets when available, fall back to hidden compat ones
        thresh = getattr(self, '_thresh_sl', self.thresh_spin).value()
        if hasattr(self, '_thresh_sl'):
            thresh = thresh / 100.0  # slider is 10-90 → 0.10-0.90
        else:
            thresh = self.thresh_spin.value()
        cluster_min = getattr(self, '_dbscan_spin', self.min_spin).value()
        model_key = self.model_combo.currentData() or self.model_combo.currentText()
        return {
            'model': model_key,
            'threshold': thresh,
            'cluster_eps': self.eps_spin.value(),
            'cluster_min': cluster_min,
            'no_cluster': self.no_cluster_cb.isChecked(),
            'copy_files': self.copy_cb.isChecked(),
            'recursive': self.recursive_cb.isChecked(),
            'use_gpu': self.gpu_cb.isChecked(),
            'max_characters': self.max_char_slider.value(),
            'max_per_character': self.topn_spin.value(),
        }

    # ─── Image counting ───────────────────────────────────────────────────────

    def _count_images_recursive(self, folder: str) -> int:
        """Count all images in folder and all subfolders."""
        count = 0
        for f in Path(folder).rglob("*"):
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                count += 1
        return count

    # ─── Processing ───────────────────────────────────────────────────────────

    def _start(self):
        if not self.input_folder and not self.input_files:
            return

        self.log(f"\n{'='*40}")
        mode_label = "🎌 Anime" if self._mode == 'anime' else "🎭 Gerçek Yüz"
        self.log(f"{mode_label}  —  {get_text('char_start_log', self.lang)}")
        settings = self._get_settings()
        self.log(f"   Mod      : {mode_label}")
        self.log(f"   Model    : {'lbpcascade + ResNet-18' if self._mode == 'anime' else settings['model']}")
        self.log(f"   Threshold: {settings['threshold']:.2f}")
        cluster_info = 'disabled' if settings['no_cluster'] else f"eps={settings['cluster_eps']:.2f}  min={settings['cluster_min']}"
        self.log(f"   Cluster  : {cluster_info}")
        self.log(f"   Action   : {'Copy' if settings['copy_files'] else 'Move'}")
        self.log(f"   GPU      : {'Yes' if settings['use_gpu'] else 'No'}")
        self.log(f"   Max chars: {settings['max_characters']}")
        if self.ref_folder:
            self.log(f"   References: {self.ref_folder}")
        if self.input_files:
            out = self.output_folder or "görsel yanında _sorted klasörü"
        else:
            out = self.output_folder or str(Path(self.input_folder) / '_sorted')
        self.log(f"   Output   : {out}")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_input_btn.setEnabled(False)
        self.browse_files_btn.setEnabled(False)
        self.browse_ref_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")

        # Clean up any leftover thread
        self._cleanup_thread()

        self._thread = CharacterSortThread(
            input_dir=self.input_folder,
            output_dir=self.output_folder,
            reference_dir=self.ref_folder,
            settings=settings,
            input_files=self.input_files,
            mode=self._mode,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.sort_finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.log_msg.connect(self.log)
        self._thread.start()

    def _cleanup_thread(self):
        """Disconnect signals and schedule deletion of the current thread."""
        if self._thread is not None:
            t = self._thread
            self._thread = None
            try:
                t.progress.disconnect(self._on_progress)
                t.sort_finished.disconnect(self._on_finished)
                t.error.disconnect(self._on_error)
                t.log_msg.disconnect(self.log)
            except (TypeError, RuntimeError):
                pass
            if not t.isRunning():
                t.deleteLater()
            else:
                # sort_directory() cannot be interrupted — thread will finish
                # on its own. Connect QThread's built-in finished() (not our
                # custom sort_finished) so deleteLater fires when run() returns.
                # Qt keeps the C++ object alive until then; Python wrapper stays
                # alive because C++ is alive.
                try:
                    t.finished.connect(t.deleteLater)
                except (TypeError, RuntimeError):
                    pass

    def _stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self.log("⏹️  " + get_text('log_stopping', self.lang))
            # sort_directory() runs to completion — don't block the main thread.
            # _cleanup_thread() below wires up QThread.finished → deleteLater
            # so the object is safely deleted when the thread naturally exits.
        self._cleanup_thread()
        self._reset_buttons()

    def _reset_buttons(self):
        self.start_btn.setEnabled(bool(self.input_folder or self.input_files))
        self.stop_btn.setEnabled(False)
        self.browse_input_btn.setEnabled(True)
        self.browse_files_btn.setEnabled(True)
        self.browse_ref_btn.setEnabled(True)

    # ─── Thread callbacks ─────────────────────────────────────────────────────

    def _on_progress(self, current: int, total: int, msg: str):
        pct = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{current}/{total} — {msg[:40]}")

    def _on_finished(self, stats: Dict):
        self.progress_bar.setValue(100)
        total = sum(stats.values())
        self.log(f"\n{'='*40}")
        self.log(f"✅ {get_text('char_complete', self.lang).format(total)}")
        known = sum(v for k, v in stats.items() if not k.startswith(('character_', 'unknown', 'no_face', 'multi_face')))
        clustered = sum(v for k, v in stats.items() if k.startswith('character_'))
        if known:
            self.log(f"   {get_text('char_matched', self.lang)}: {known}")
        if clustered:
            self.log(f"   {get_text('char_clustered', self.lang)}: {clustered}")
        for name, count in sorted(stats.items()):
            self.log(f"   📁 {name}: {count}")
        self._cleanup_thread()
        self._reset_buttons()
        self._populate_cluster_grid(stats)

    def _populate_cluster_grid(self, stats: Dict):
        """Populate the cluster results grid with folder cards."""
        from PyQt5.QtWidgets import QGridLayout
        # Clear existing content
        while self._cluster_grid_lay.count():
            item = self._cluster_grid_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not stats:
            self._cluster_placeholder = QLabel("No clusters found.")
            self._cluster_placeholder.setAlignment(Qt.AlignCenter)
            self._cluster_placeholder.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent; border: none;")
            self._cluster_grid_lay.addWidget(self._cluster_placeholder, 0, 0)
            self._cluster_count_badge.setText("0 Total")
            return

        cols = 5
        self._cluster_count_badge.setText(f"{len(stats)} Total")
        for idx, (name, count) in enumerate(sorted(stats.items())):
            row, col = divmod(idx, cols)
            card = self._make_cluster_card(name, count)
            self._cluster_grid_lay.addWidget(card, row, col)

    def _make_cluster_card(self, name: str, count: int, confidence: float = 0.0) -> QFrame:
        is_noise = name in ('unknown', 'no_face', 'multi_face') or name.startswith('unknown')
        is_named = not is_noise and not name.startswith('cluster_')
        if is_noise:
            border_color = "#7f1d1d"
            name_color   = "#f87171"
        elif is_named:
            border_color = f"{theme.ORANGE}55"
            name_color   = theme.ORANGE
        else:
            border_color = theme.BORDER
            name_color   = theme.TEXT_PRIMARY

        card = QFrame()
        card.setFixedWidth(160)
        card.setStyleSheet(
            f"QFrame {{ background: {theme.BG_SURFACE}; border: 1px solid {border_color};"
            f" border-radius: 8px; overflow: hidden; }}"
        )
        cl = QVBoxLayout(card); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)

        # Mosaic top (3×2 grid of tinted rectangles)
        mosaic = QFrame(); mosaic.setFixedHeight(100)
        mosaic.setStyleSheet(f"background: {theme.BG_DEEPEST}; border: none;")
        from PyQt5.QtWidgets import QGridLayout as _GL
        mg = _GL(mosaic); mg.setSpacing(1); mg.setContentsMargins(1, 1, 1, 1)
        _shades = ["#27272a","#3f3f46","#27272a","#1c1c1e","#27272a","#3f3f46"]
        if is_noise:
            _shades = ["#1c0a0a","#27272a","#1c0a0a","#27272a","#1c0a0a","#27272a"]
        for r in range(2):
            for c in range(3):
                i = r * 3 + c
                cell = QLabel()
                cell.setStyleSheet(f"background:{_shades[i]};border:none;")
                mg.addWidget(cell, r, c)
        # image count badge (top-right overlay)
        badge_row = QHBoxLayout(); badge_row.addStretch()
        img_badge = QLabel(f"  {count}  ")
        img_badge.setStyleSheet(
            f"background: rgba(0,0,0,0.7); color: {theme.TEXT_SECONDARY};"
            f" font-family: {theme.FONT_MONO}; font-size: 9px; border-radius: 3px; border: none;"
        )
        badge_row.addWidget(img_badge)
        mg.addLayout(badge_row, 0, 2, Qt.AlignTop | Qt.AlignRight)
        cl.addWidget(mosaic)

        # Info section
        info = QFrame()
        info.setStyleSheet(
            f"QFrame {{ background: {theme.BG_SURFACE}; border: none;"
            f" border-top: 1px solid {theme.BORDER}; border-bottom-left-radius: 8px;"
            f" border-bottom-right-radius: 8px; }}"
        )
        il = QVBoxLayout(info); il.setContentsMargins(10, 8, 10, 8); il.setSpacing(2)
        name_row = QHBoxLayout(); name_row.setSpacing(4)
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"color: {name_color}; font-weight: 600; font-size: {theme.fs(11)};"
            f" background: transparent; border: none;"
        )
        name_lbl.setWordWrap(False)
        name_row.addWidget(name_lbl, stretch=1)
        edit_icon = QLabel("✏" if not is_noise else "🗑")
        edit_icon.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 10px; background: transparent; border: none;")
        name_row.addWidget(edit_icon)
        il.addLayout(name_row)
        conf_lbl = QLabel("Outliers" if is_noise else f"Confidence: {confidence:.2f}" if confidence else f"{count} images")
        conf_lbl.setStyleSheet(
            f"color: {'#7f1d1d' if is_noise else theme.TEXT_MUTED}; font-size: {theme.fs(9)};"
            f" font-family: {theme.FONT_MONO}; background: transparent; border: none;"
        )
        il.addWidget(conf_lbl)
        cl.addWidget(info)
        return card

    def _on_error(self, msg: str):
        self.log(f"❌ {get_text('log_error', self.lang).format(msg)}")
        self._cleanup_thread()
        self._reset_buttons()

    # ─── Logging ──────────────────────────────────────────────────────────────

    def log(self, message: str):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    # ─── Language update ──────────────────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self.title_lbl.setText(get_text('char_sort_title', lang))
        self.subtitle_lbl.setText(get_text('char_sort_subtitle', lang))
        self.step1_title.setText(get_text('step1_select_folder', lang))
        self.step2_title.setText(get_text('step2_settings', lang))
        self.step3_title.setText(get_text('step3_start', lang))

        # Folder row labels
        self.input_lbl.setText(get_text('char_input_folder', lang))
        self.ref_lbl.setText(get_text('char_ref_folder', lang))
        self.out_lbl.setText(get_text('char_output_folder', lang))

        # Browse buttons
        self.browse_input_btn.setText(get_text('char_browse_input', lang))
        self.browse_ref_btn.setText(get_text('char_browse_ref', lang))
        self.browse_out_btn.setText(get_text('char_browse_output', lang))
        self.clear_ref_btn.setToolTip(get_text('char_clear_ref', lang))
        self.clear_out_btn.setToolTip(get_text('char_clear_output', lang))

        # Action buttons
        self.start_btn.setText(get_text('char_start_btn', lang))
        self.stop_btn.setText(get_text('stop_btn', lang))

        # Settings labels + tooltips
        self.model_lbl.setText(get_text('char_model', lang))
        self.model_combo.setToolTip(get_text('char_model_tooltip', lang))
        self.thresh_lbl.setText(get_text('char_threshold', lang))
        self.thresh_lbl_info.setToolTip(get_text('char_threshold_tooltip', lang))
        self.eps_lbl.setText(get_text('char_cluster_eps', lang))
        self.eps_info.setToolTip(get_text('char_cluster_eps_tooltip', lang))
        self.min_lbl.setText(get_text('char_cluster_min', lang))
        self.min_info.setToolTip(get_text('char_cluster_min_tooltip', lang))
        self.max_char_lbl.setText(get_text('char_max_characters', lang))
        self.max_char_info.setToolTip(get_text('char_max_characters_tooltip', lang))
        self.topn_lbl.setText(get_text('char_max_per_char', lang))
        self.topn_info.setToolTip(get_text('char_max_per_char_tooltip', lang))

        # Checkboxes + tooltips
        self.no_cluster_cb.setText(get_text('char_no_cluster', lang))
        self.no_cluster_cb.setToolTip(get_text('char_no_cluster_tooltip', lang))
        self.copy_cb.setText(get_text('char_copy_files', lang))
        self.copy_cb.setToolTip(get_text('char_copy_tooltip', lang))
        self.recursive_cb.setText(get_text('recursive_search', lang))
        self.recursive_cb.setToolTip(get_text('recursive_tooltip', lang))
        self.gpu_cb.setText(get_text('char_use_gpu', lang))
        self.gpu_cb.setToolTip(get_text('char_use_gpu_tooltip', lang))

        # Preset bar — re-sync info text with current mode and language
        self._preset_type_lbl.setText("Model Tipi:" if lang == 'tr' else "Model Type:")
        if self._mode == 'anime':
            self._preset_info_lbl.setText(
                "lbpcascade_animeface + ResNet-18 — anime/çizgi film karakterleri için"
                if lang == 'tr' else
                "lbpcascade_animeface + ResNet-18 — for anime/cartoon characters"
            )
        else:
            self._preset_info_lbl.setText(
                "InsightFace buffalo_l — gerçek insan yüzleri için"
                if lang == 'tr' else
                "InsightFace buffalo_l — for real human faces"
            )

        # Status labels (only if not already set by user selection)
        if not self.ref_folder:
            self.ref_status_lbl.setText(get_text('char_ref_optional', lang))
        if not self.output_folder:
            self.out_status_lbl.setText(get_text('char_output_default', lang))

        # Drop zone placeholders (only if no folder selected)
        if not self.input_folder:
            self.input_drop.get_label().setText(get_text('char_drag_input', lang))
        if not self.ref_folder:
            self.ref_drop.get_label().setText(get_text('char_drag_ref', lang))
        if not self.output_folder:
            self.out_drop.get_label().setText(get_text('char_drag_output', lang))

    # ─── Theme ────────────────────────────────────────────────────────────────

    def refresh_styles(self):
        """Re-apply all stylesheets after a theme change."""
        # Title + subtitle
        self.title_lbl.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self.subtitle_lbl.setStyleSheet(theme.label_muted())
        self.log_text.setStyleSheet(theme.log_area())

        # Preset buttons — re-apply active/inactive styles after theme change
        self._preset_type_lbl.setStyleSheet(theme.label_default())
        self._preset_real_btn.setStyleSheet(self._preset_btn_style(self._mode == 'real'))
        self._preset_anime_btn.setStyleSheet(self._preset_btn_style(self._mode == 'anime'))
        self._preset_info_lbl.setStyleSheet(theme.label_muted())

        # Step cards (the three QFrame children built via _build_*_step)
        for frame in self.findChildren(QFrame):
            ss = frame.styleSheet()
            if "border-radius: 10px" in ss:
                frame.setStyleSheet(theme.card_frame())

        # Step titles
        self.step1_title.setStyleSheet(theme.label_section())
        self.step2_title.setStyleSheet(theme.label_section())
        self.step3_title.setStyleSheet(theme.label_section())

        # Folder rows
        self.input_lbl.setStyleSheet(theme.label_default())
        self.browse_input_btn.setStyleSheet(theme.btn_browse())
        self.browse_files_btn.setStyleSheet(theme.btn_browse())
        self.ref_lbl.setStyleSheet(theme.label_default())
        self.browse_ref_btn.setStyleSheet(theme.btn_browse())
        self.clear_ref_btn.setStyleSheet(theme.btn_danger())
        self.out_lbl.setStyleSheet(theme.label_default())
        self.browse_out_btn.setStyleSheet(theme.btn_browse())
        self.clear_out_btn.setStyleSheet(theme.btn_danger())

        # Drop zones — preserve state
        if self.input_folder:
            self.input_drop.setStyleSheet(theme.drop_zone_frame_success())
            self.input_drop.get_label().setStyleSheet(theme.label_success())
        else:
            self.input_drop.setStyleSheet(theme.drop_zone_frame_default())
            self.input_drop.get_label().setStyleSheet(theme.label_transparent())

        if self.ref_folder:
            self.ref_drop.setStyleSheet(theme.drop_zone_frame_success())
            self.ref_drop.get_label().setStyleSheet(theme.label_success())
            self.ref_status_lbl.setStyleSheet(theme.label_success())
        else:
            self.ref_drop.setStyleSheet(theme.drop_zone_frame_default())
            self.ref_drop.get_label().setStyleSheet(theme.label_transparent())
            self.ref_status_lbl.setStyleSheet(theme.label_muted())

        if self.output_folder:
            self.out_drop.setStyleSheet(theme.drop_zone_frame_success())
            self.out_drop.get_label().setStyleSheet(theme.label_success())
            self.out_status_lbl.setStyleSheet(theme.label_success())
        else:
            self.out_drop.setStyleSheet(theme.drop_zone_frame_default())
            self.out_drop.get_label().setStyleSheet(theme.label_transparent())
            self.out_status_lbl.setStyleSheet(theme.label_muted())

        self.input_count_lbl.setStyleSheet(theme.label_success())

        # Settings row labels, spinboxes, combos
        self.model_lbl.setStyleSheet(theme.label_frame())
        self.model_combo.setStyleSheet(theme.combo())
        self.thresh_lbl.setStyleSheet(theme.label_frame())
        self.thresh_lbl_info.setStyleSheet(theme.info_icon_frame())
        self.thresh_spin.setStyleSheet(theme.spinbox())
        self.eps_lbl.setStyleSheet(theme.label_frame())
        self.eps_info.setStyleSheet(theme.info_icon_frame())
        self.eps_spin.setStyleSheet(theme.spinbox())
        self.min_lbl.setStyleSheet(theme.label_frame())
        self.min_info.setStyleSheet(theme.info_icon_frame())
        self.min_spin.setStyleSheet(theme.spinbox())
        self.max_char_lbl.setStyleSheet(theme.label_frame())
        self.max_char_info.setStyleSheet(theme.info_icon_frame())
        self.max_char_slider.setStyleSheet(theme.slider())
        self.max_char_value_lbl.setStyleSheet(theme.label_value())
        self._ticks_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)};"
        )
        self.topn_lbl.setStyleSheet(theme.label_frame())
        self.topn_info.setStyleSheet(theme.info_icon_frame())
        self.topn_spin.setStyleSheet(theme.spinbox())

        # Step 3
        self.start_btn.setStyleSheet(theme.btn_primary())
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.progress_bar.setStyleSheet(theme.progress_bar())
