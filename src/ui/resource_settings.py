"""
Resource & Performance Settings Drawer for LoRA-Harvester.
Slide-in panel to tune GPU, CPU, memory, and batch settings.
"""

import json
import multiprocessing
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QCheckBox, QFrame, QScrollArea,
)
from PyQt5.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, pyqtSignal, QRect,
)
from PyQt5.QtGui import QFont
from src.ui.translations import get_text
from src.ui import theme


# Persistent settings path (next to the executable / repo root)
_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "resource_settings.json"

# Hardware-detection helpers
_CPU_COUNT = multiprocessing.cpu_count() or 4

def _detect_gpu_mem_gb() -> float:
    """Return GPU memory in GB, or 0 if no CUDA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except Exception:
        pass
    return 0.0


# ─── Default values ────────────────────────────────────────────────────────

def _defaults() -> dict:
    gpu_gb = _detect_gpu_mem_gb()
    return {
        # GPU
        "gpu_enabled": gpu_gb > 0,
        "fp16_enabled": True,
        "gpu_mem_limit_pct": 80,          # % of total VRAM to use
        # Batch / throughput
        "batch_size": 4 if gpu_gb < 6 else 8,
        "prefetch_frames": 30,
        # CPU / threading
        "cpu_threads": min(_CPU_COUNT, 4),
        "decode_workers": min(_CPU_COUNT, 4),
        # Memory
        "ram_limit_mb": 4096,
        # Misc perf
        "async_save": True,
        "jpeg_quality": 95,
        "auto_gc": True,
    }


# ─── Persistence helpers ───────────────────────────────────────────────────

def load_settings() -> dict:
    defaults = _defaults()
    if _SETTINGS_PATH.exists():
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge: saved values override defaults; unknown keys are dropped
            for k in defaults:
                if k in saved and type(saved[k]) is type(defaults[k]):
                    defaults[k] = saved[k]
        except Exception:
            pass
    return defaults


def save_settings(data: dict):
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Drawer Widget
# ═══════════════════════════════════════════════════════════════════════════

class ResourceSettingsDrawer(QFrame):
    """
    Slide-in settings drawer.  Lives as a child of the main window
    and animates from the right edge.
    """

    closed = pyqtSignal()           # emitted when the drawer is closed
    settings_changed = pyqtSignal(dict)  # emitted when "Apply" is clicked

    DRAWER_WIDTH = 380

    def __init__(self, lang: str = "en", parent=None):
        super().__init__(parent)
        self.lang = lang
        self._settings = load_settings()
        self._gpu_mem_gb = _detect_gpu_mem_gb()

        # Visual
        self.setFixedWidth(self.DRAWER_WIDTH)
        self.setStyleSheet(f"""
            ResourceSettingsDrawer {{
                background-color: {theme.BG_PANEL};
                border-left: 2px solid {theme.BORDER_ACCENT};
            }}
        """)
        self.setFrameShape(QFrame.StyledPanel)

        # Animation
        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._is_open = False

        self._build_ui()
        self._load_values()
        self.hide()

    # ─── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(6)

        # Header
        header = QHBoxLayout()
        self._title = QLabel(get_text("res_title", self.lang))
        self._title.setFont(QFont("Arial", 16, QFont.Bold))
        self._title.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        header.addWidget(self._title)
        header.addStretch()
        self._close_btn = QPushButton("✕")
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {theme.TEXT_SECONDARY};
                font-size: 18px; border: none; border-radius: 4px;
            }}
            QPushButton:hover {{ background: {theme.BG_HOVER}; color: {theme.RED}; }}
        """)
        self._close_btn.clicked.connect(self.close_drawer)
        header.addWidget(self._close_btn)
        root.addLayout(header)

        # Subtitle
        self._subtitle = QLabel(get_text("res_subtitle", self.lang))
        self._subtitle.setStyleSheet(theme.label_muted())
        self._subtitle.setWordWrap(True)
        root.addWidget(self._subtitle)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"background: transparent;")
        inner = QWidget()
        self._lay = QVBoxLayout(inner)
        self._lay.setSpacing(4)
        self._lay.setContentsMargins(0, 8, 0, 0)

        # ── GPU Section ──────────────────────────────────────────
        self._add_section("res_section_gpu")

        self.gpu_cb = self._add_checkbox("res_gpu_enabled", "gpu_enabled")
        self.fp16_cb = self._add_checkbox("res_fp16", "fp16_enabled")

        self.vram_slider, self.vram_val = self._add_slider(
            "res_gpu_mem_limit", 10, 100, 5, "%",
        )

        # ── Batch / Throughput Section ───────────────────────────
        self._add_section("res_section_batch")

        self.batch_slider, self.batch_val = self._add_slider(
            "res_batch_size", 1, 32, 1,
        )
        self.prefetch_slider, self.prefetch_val = self._add_slider(
            "res_prefetch_frames", 1, 120, 1,
        )

        # ── CPU / Threading Section ──────────────────────────────
        self._add_section("res_section_cpu")

        self.threads_slider, self.threads_val = self._add_slider(
            "res_cpu_threads", 1, _CPU_COUNT, 1,
        )
        self.workers_slider, self.workers_val = self._add_slider(
            "res_decode_workers", 1, _CPU_COUNT, 1,
        )

        # ── Memory Section ───────────────────────────────────────
        self._add_section("res_section_memory")

        self.ram_slider, self.ram_val = self._add_slider(
            "res_ram_limit", 512, 32768, 256, "MB",
        )

        # ── Misc Performance ─────────────────────────────────────
        self._add_section("res_section_misc")

        self.async_cb = self._add_checkbox("res_async_save", "async_save")
        self.gc_cb = self._add_checkbox("res_auto_gc", "auto_gc")
        self.jpeg_slider, self.jpeg_val = self._add_slider(
            "res_jpeg_quality", 50, 100, 5,
        )

        self._lay.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # ── Bottom buttons ───────────────────────────────────────
        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton(get_text("res_reset", self.lang))
        self._reset_btn.setStyleSheet(theme.btn_secondary())
        self._reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self._reset_btn)

        self._apply_btn = QPushButton(get_text("res_apply", self.lang))
        self._apply_btn.setStyleSheet(theme.btn_primary())
        self._apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(self._apply_btn)
        root.addLayout(btn_row)

        self.setLayout(root)

    # ─── Builder helpers ─────────────────────────────────────────────────

    def _add_section(self, key: str):
        lbl = QLabel(get_text(key, self.lang))
        lbl.setFont(QFont("Arial", 11, QFont.Bold))
        lbl.setStyleSheet(
            f"color: {theme.ORANGE}; margin-top: 10px; margin-bottom: 2px;"
        )
        self._lay.addWidget(lbl)
        # Store for language update
        if not hasattr(self, "_section_labels"):
            self._section_labels = []
        self._section_labels.append((lbl, key))

    def _add_checkbox(self, text_key: str, setting_key: str) -> QCheckBox:
        cb = QCheckBox(get_text(text_key, self.lang))
        cb.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; padding: 3px 0;")
        self._lay.addWidget(cb)
        # Store mapping for language update
        if not hasattr(self, "_cb_keys"):
            self._cb_keys = []
        self._cb_keys.append((cb, text_key))
        return cb

    def _add_slider(self, text_key, mn, mx, step, suffix=""):
        row = QHBoxLayout()
        lbl = QLabel(get_text(text_key, self.lang))
        lbl.setStyleSheet(theme.label_default())
        lbl.setMinimumWidth(130)
        row.addWidget(lbl)

        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(mn)
        sl.setMaximum(mx)
        sl.setSingleStep(step)
        sl.setStyleSheet(theme.slider())
        row.addWidget(sl, stretch=1)

        val = QLabel("")
        val.setMinimumWidth(55)
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val.setStyleSheet(theme.label_value())
        row.addWidget(val)

        # Live value update
        def _update(v, _val=val, _suf=suffix):
            _val.setText(f"{v}{_suf}")
        sl.valueChanged.connect(_update)

        self._lay.addLayout(row)

        # Store for language update
        if not hasattr(self, "_slider_keys"):
            self._slider_keys = []
        self._slider_keys.append((lbl, text_key))

        return sl, val

    # ─── Value load / save ───────────────────────────────────────────────

    def _load_values(self):
        s = self._settings
        self.gpu_cb.setChecked(s["gpu_enabled"])
        self.fp16_cb.setChecked(s["fp16_enabled"])
        self.vram_slider.setValue(s["gpu_mem_limit_pct"])
        self.batch_slider.setValue(s["batch_size"])
        self.prefetch_slider.setValue(s["prefetch_frames"])
        self.threads_slider.setValue(s["cpu_threads"])
        self.workers_slider.setValue(s["decode_workers"])
        self.ram_slider.setValue(s["ram_limit_mb"])
        self.async_cb.setChecked(s["async_save"])
        self.gc_cb.setChecked(s["auto_gc"])
        self.jpeg_slider.setValue(s["jpeg_quality"])

    def _collect_values(self) -> dict:
        return {
            "gpu_enabled": self.gpu_cb.isChecked(),
            "fp16_enabled": self.fp16_cb.isChecked(),
            "gpu_mem_limit_pct": self.vram_slider.value(),
            "batch_size": self.batch_slider.value(),
            "prefetch_frames": self.prefetch_slider.value(),
            "cpu_threads": self.threads_slider.value(),
            "decode_workers": self.workers_slider.value(),
            "ram_limit_mb": self.ram_slider.value(),
            "async_save": self.async_cb.isChecked(),
            "auto_gc": self.gc_cb.isChecked(),
            "jpeg_quality": self.jpeg_slider.value(),
        }

    def _apply(self):
        data = self._collect_values()
        self._settings = data
        save_settings(data)
        self.settings_changed.emit(data)
        self.close_drawer()

    def _reset_defaults(self):
        self._settings = _defaults()
        self._load_values()

    # ─── Slide animation ─────────────────────────────────────────────────

    def open_drawer(self):
        if self._is_open:
            return
        parent = self.parent()
        if parent is None:
            return
        ph = parent.height()
        pw = parent.width()
        # Start off-screen right, animate to right-aligned
        start = QRect(pw, 0, self.DRAWER_WIDTH, ph)
        end = QRect(pw - self.DRAWER_WIDTH, 0, self.DRAWER_WIDTH, ph)
        self.setGeometry(start)
        self.show()
        self.raise_()
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()
        self._is_open = True

    def close_drawer(self):
        if not self._is_open:
            return
        parent = self.parent()
        if parent is None:
            self.hide()
            self._is_open = False
            self.closed.emit()
            return
        ph = parent.height()
        pw = parent.width()
        start = QRect(pw - self.DRAWER_WIDTH, 0, self.DRAWER_WIDTH, ph)
        end = QRect(pw, 0, self.DRAWER_WIDTH, ph)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.finished.connect(self._on_close_done)
        self._anim.start()

    def _on_close_done(self):
        self._anim.finished.disconnect(self._on_close_done)
        self.hide()
        self._is_open = False
        self.closed.emit()

    def toggle(self):
        if self._is_open:
            self.close_drawer()
        else:
            self.open_drawer()

    def get_settings(self) -> dict:
        """Return the current in-memory settings dict."""
        return dict(self._settings)

    # ─── Language update ─────────────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self._title.setText(get_text("res_title", lang))
        self._subtitle.setText(get_text("res_subtitle", lang))
        self._reset_btn.setText(get_text("res_reset", lang))
        self._apply_btn.setText(get_text("res_apply", lang))
        for lbl, key in self._section_labels:
            lbl.setText(get_text(key, lang))
        for cb, key in self._cb_keys:
            cb.setText(get_text(key, lang))
        for lbl, key in self._slider_keys:
            lbl.setText(get_text(key, lang))
