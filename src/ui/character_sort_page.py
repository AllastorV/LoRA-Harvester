"""
Character Sort Page for LoRA-Harvester v3.0
Identifies characters by face and organises frames into per-character folders.
Works on any existing image folder — fully independent from video processing.
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QPushButton, QProgressBar, QFileDialog, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent
from typing import Dict, Optional
from src.ui.translations import get_text
from src.ui import theme


# ─────────────────────────────────────────────────────────────────────────────
# Background thread
# ─────────────────────────────────────────────────────────────────────────────

class CharacterSortThread(QThread):
    """Runs CharacterRecognizer.sort_directory() in a background thread."""

    progress = pyqtSignal(int, int, str)   # current, total, message
    finished = pyqtSignal(dict)            # stats dict
    error = pyqtSignal(str)
    log_msg = pyqtSignal(str)              # intermediate log messages

    def __init__(
        self,
        input_dir: str,
        output_dir: Optional[str],
        reference_dir: Optional[str],
        settings: Dict,
        parent=None,
    ):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.reference_dir = reference_dir
        self.settings = settings
        self._running = True

    def run(self):
        try:
            from src.core.character_recognizer import CharacterRecognizer

            s = self.settings

            cluster_eps = 99.0 if s.get('no_cluster') else s.get('cluster_eps', 0.6)
            cluster_min = 99999 if s.get('no_cluster') else s.get('cluster_min', 2)

            def progress_cb(current, total, msg):
                if self._running:
                    self.progress.emit(current, total, msg)

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
                    input_dir=self.input_dir,
                    output_dir=self.output_dir if self.output_dir else None,
                    copy=s.get('copy_files', False),
                    recursive=s.get('recursive', False),
                    max_characters=s.get('max_characters', 6),
                    max_per_character=s.get('max_per_character', 0),
                )
            finally:
                # Always release the SQLite cache so WAL is flushed before
                # this worker thread terminates.
                recognizer.close_cache()

            if self._running:
                self.finished.emit(stats)

        except ImportError as e:
            self.error.emit(
                f"Missing dependency: {e}\n"
                "Install with: pip install insightface scikit-learn onnxruntime"
            )
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def stop(self):
        self._running = False
        self.requestInterruption()


# ─────────────────────────────────────────────────────────────────────────────
# Drop Zone Frame — proper PyQt5 subclass for folder drag & drop
# ─────────────────────────────────────────────────────────────────────────────

class DropZoneFrame(QFrame):
    """
    A QFrame subclass that correctly handles folder drag-and-drop.
    Using a proper subclass instead of lambda monkey-patching is required
    because PyQt5 dispatches events via C++ virtual methods, not Python
    attribute lookup.
    """

    folder_dropped = pyqtSignal(str)

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

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile() and os.path.isdir(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self.setStyleSheet(theme.drop_zone_frame_active())
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(theme.drop_zone_frame_default())

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(theme.drop_zone_frame_default())
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    self.folder_dropped.emit(path)
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

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self.input_folder: Optional[str] = None
        self.ref_folder: Optional[str] = None
        self.output_folder: Optional[str] = None
        self._thread: Optional[CharacterSortThread] = None
        self.init_ui()

    # ─── UI construction ──────────────────────────────────────────────────────

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(30, 20, 30, 20)
        self.setLayout(layout)

        # Tooltip style handled by global theme

        # Title
        self.title_lbl = QLabel(get_text('char_sort_title', self.lang))
        self.title_lbl.setFont(QFont('Arial', 20, QFont.Bold))
        self.title_lbl.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_lbl)

        self.subtitle_lbl = QLabel(get_text('char_sort_subtitle', self.lang))
        self.subtitle_lbl.setAlignment(Qt.AlignCenter)
        self.subtitle_lbl.setStyleSheet(theme.label_muted())
        layout.addWidget(self.subtitle_lbl)

        # ── Step 1: Folders ────────────────────────────────────────────────
        layout.addWidget(self._build_folder_step())

        # ── Step 2: Settings ───────────────────────────────────────────────
        layout.addWidget(self._build_settings_step())

        # ── Step 3: Start ──────────────────────────────────────────────────
        layout.addWidget(self._build_start_step())

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(105)
        self.log_text.setMaximumHeight(225)
        self.log_text.setStyleSheet(theme.log_area())
        layout.addWidget(self.log_text)

        layout.addStretch()

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def _build_folder_step(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(theme.card_frame())
        lay = QVBoxLayout(frame)

        self.step1_title = QLabel(get_text('step1_select_folder', self.lang))
        self.step1_title.setFont(QFont('Arial', 14, QFont.Bold))
        self.step1_title.setStyleSheet(theme.label_section())
        lay.addWidget(self.step1_title)

        # Input folder row
        self.input_drop = DropZoneFrame(get_text('char_drag_input', self.lang))
        self.input_drop.folder_dropped.connect(self._set_input)

        self.input_lbl = QLabel(get_text('char_input_folder', self.lang))
        self.input_lbl.setStyleSheet(theme.label_default())

        self.browse_input_btn = QPushButton(get_text('char_browse_input', self.lang))
        self.browse_input_btn.setStyleSheet(theme.btn_browse())
        self.browse_input_btn.clicked.connect(self._browse_input)

        row_in = QHBoxLayout()
        row_in.addWidget(self.input_lbl)
        row_in.addWidget(self.input_drop, stretch=1)
        row_in.addWidget(self.browse_input_btn)
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
        self.step2_title.setFont(QFont('Arial', 14, QFont.Bold))
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
        self.step3_title.setFont(QFont('Arial', 14, QFont.Bold))
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
        lbl = self.input_drop.get_label()
        display = path if len(path) <= 50 else "..." + path[-47:]
        lbl.setText(display)
        lbl.setStyleSheet(theme.label_success())
        self.input_drop.setStyleSheet(theme.drop_zone_frame_success())
        self.input_drop.setToolTip(path)
        count = self._count_images(path)
        self.input_count_lbl.setText(get_text('images_found', self.lang).format(count))
        # Enable start if folder is selected — images may be in subdirectories (recursive mode)
        self.start_btn.setEnabled(True)
        self.log(f"📂 Input: {path}  ({count} images)")

    def _set_ref(self, path: str):
        self.ref_folder = path
        lbl = self.ref_drop.get_label()
        display = path if len(path) <= 50 else "..." + path[-47:]
        lbl.setText(display)
        lbl.setStyleSheet(theme.label_success())
        self.ref_drop.setStyleSheet(theme.drop_zone_frame_success())
        self.ref_drop.setToolTip(path)
        self.clear_ref_btn.setVisible(True)
        char_dirs = [d for d in Path(path).iterdir() if d.is_dir()]
        self.ref_status_lbl.setText(
            get_text('char_ref_loaded', self.lang).format(len(char_dirs))
        )
        self.ref_status_lbl.setStyleSheet(theme.label_success())
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
        return {
            'model': self.model_combo.currentText(),
            'threshold': self.thresh_spin.value(),
            'cluster_eps': self.eps_spin.value(),
            'cluster_min': self.min_spin.value(),
            'no_cluster': self.no_cluster_cb.isChecked(),
            'copy_files': self.copy_cb.isChecked(),
            'recursive': self.recursive_cb.isChecked(),
            'use_gpu': self.gpu_cb.isChecked(),
            'max_characters': self.max_char_slider.value(),
            'max_per_character': self.topn_spin.value(),
        }

    # ─── Image counting ───────────────────────────────────────────────────────

    def _count_images(self, folder: str) -> int:
        count = 0
        p = Path(folder)
        pattern = "**/*" if self.recursive_cb.isChecked() else "*"
        for f in p.glob(pattern):
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                count += 1
        return count

    # ─── Processing ───────────────────────────────────────────────────────────

    def _start(self):
        if not self.input_folder:
            return

        self.log(f"\n{'='*40}")
        self.log(f"🎭 {get_text('char_start_log', self.lang)}")
        settings = self._get_settings()
        self.log(f"   Model    : {settings['model']}")
        self.log(f"   Threshold: {settings['threshold']:.2f}")
        cluster_info = 'disabled' if settings['no_cluster'] else f"eps={settings['cluster_eps']:.2f}  min={settings['cluster_min']}"
        self.log(f"   Cluster  : {cluster_info}")
        self.log(f"   Action   : {'Copy' if settings['copy_files'] else 'Move'}")
        self.log(f"   GPU      : {'Yes' if settings['use_gpu'] else 'No'}")
        self.log(f"   Max chars: {settings['max_characters']}")
        if self.ref_folder:
            self.log(f"   References: {self.ref_folder}")
        out = self.output_folder or str(Path(self.input_folder) / '_sorted')
        self.log(f"   Output   : {out}")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_input_btn.setEnabled(False)
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
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.log_msg.connect(self.log)
        self._thread.start()

    def _cleanup_thread(self):
        """Disconnect signals and schedule deletion of the current thread"""
        if self._thread is not None:
            try:
                self._thread.progress.disconnect(self._on_progress)
                self._thread.finished.disconnect(self._on_finished)
                self._thread.error.disconnect(self._on_error)
                self._thread.log_msg.disconnect(self.log)
            except (TypeError, RuntimeError):
                pass
            self._thread.deleteLater()
            self._thread = None

    def _stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(5000)
            self.log("⏹️  " + get_text('log_stopping', self.lang))
        self._cleanup_thread()
        self._reset_buttons()

    def _reset_buttons(self):
        self.start_btn.setEnabled(bool(self.input_folder))
        self.stop_btn.setEnabled(False)
        self.browse_input_btn.setEnabled(True)
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
