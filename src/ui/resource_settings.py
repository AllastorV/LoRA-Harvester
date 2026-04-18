"""
Resource & Performance Settings Drawer for LoRA-Harvester.
Slide-in panel to tune GPU, CPU, memory, and batch settings.
Includes a live System Monitor widget (CPU, RAM, GPU, VRAM).
"""

import json
import multiprocessing
import subprocess
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QCheckBox, QFrame, QScrollArea, QProgressBar,
    QGraphicsOpacityEffect,
)
from PyQt5.QtCore import (
    Qt, QPropertyAnimation, QAbstractAnimation, QEasingCurve, pyqtSignal,
    QRect, QTimer,
)
from PyQt5.QtGui import QFont, QColor
from src.ui.translations import get_text
from src.ui import theme

# Windows: suppress the black console window that flashes on every subprocess.run().
_NO_WINDOW_KW = {}
if sys.platform == "win32":
    _NO_WINDOW_KW["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


# Persistent settings path (next to the executable / repo root)
_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "resource_settings.json"

# Hardware-detection helpers
_CPU_COUNT = multiprocessing.cpu_count() or 4

def _detect_gpu_mem_gb() -> float:
    """Return GPU memory in GB, or 0 if no CUDA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
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
        # Theme / UI
        "theme_mode": theme.get_mode(),       # "dark" or "light"
        "font_scale": int(theme.get_font_scale() * 100),  # 80..140
        "accent": theme.get_accent(),         # hex color "#rrggbb"
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
#  System Monitor Widget
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_bytes(n: float) -> str:
    if n >= 1024:
        return f"{n / 1024:.1f} TB"
    return f"{n:.1f} GB"


class _UsageBar(QFrame):
    """A single resource gauge: label + animated progress bar + value text."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)

        top = QHBoxLayout()
        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(11)}; "
            f"font-weight: bold; border: none; background: transparent;"
        )
        self._value = QLabel("—")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._value.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent;"
        )
        top.addWidget(self._label)
        top.addStretch()
        top.addWidget(self._value)
        lay.addLayout(top)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._apply_bar_style()
        lay.addWidget(self._bar)

        self._anim = QPropertyAnimation(self._bar, b"value", self)
        self._anim.setDuration(400)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _apply_bar_style(self):
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {theme.BG_DARK};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {self._color};
                border-radius: 3px;
            }}
        """)

    def set_value(self, pct: float, text: str):
        target = max(0, min(1000, int(pct * 10)))
        if self._anim.state() == QAbstractAnimation.Running:
            self._anim.stop()
        self._anim.setStartValue(self._bar.value())
        self._anim.setEndValue(target)
        self._anim.start()
        self._value.setText(text)

    def set_label(self, text: str):
        self._label.setText(text)

    def refresh_styles(self):
        self._label.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(11)}; "
            f"font-weight: bold; border: none; background: transparent;"
        )
        self._value.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent;"
        )
        self._apply_bar_style()


class SystemMonitorWidget(QFrame):
    """Live system usage preview: CPU, RAM, GPU, VRAM with animated bars."""

    def __init__(self, lang: str = "en", parent=None):
        super().__init__(parent)
        self.lang = lang
        self._gpu_available = False
        self._apply_frame_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        self._title_lbl = QLabel(get_text("sys_monitor_title", lang))
        self._title_lbl.setFont(QFont("Arial", 11, QFont.Bold))
        self._title_lbl.setStyleSheet(
            f"color: {theme.ORANGE}; border: none; background: transparent;"
        )
        lay.addWidget(self._title_lbl)

        self._cpu_bar = _UsageBar(get_text("sys_cpu", lang), theme.ORANGE, self)
        self._ram_bar = _UsageBar(get_text("sys_ram", lang), "#5B9BD5", self)
        self._gpu_bar = _UsageBar(get_text("sys_gpu", lang), theme.GREEN, self)
        self._vram_bar = _UsageBar(get_text("sys_vram", lang), theme.YELLOW, self)

        lay.addWidget(self._cpu_bar)
        lay.addWidget(self._ram_bar)
        lay.addWidget(self._gpu_bar)
        lay.addWidget(self._vram_bar)

        self._gpu_na_label = QLabel(get_text("sys_gpu_not_available", lang))
        self._gpu_na_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent;"
        )
        self._gpu_na_label.setAlignment(Qt.AlignCenter)
        self._gpu_na_label.hide()
        lay.addWidget(self._gpu_na_label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.setInterval(2000)

        self._detect_gpu()
        self._poll()

    def _apply_frame_style(self):
        self.setStyleSheet(f"""
            SystemMonitorWidget {{
                background-color: {theme.BG_CARD};
                border: 1px solid {theme.BORDER};
                border-radius: 6px;
            }}
        """)

    def _detect_gpu(self):
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except Exception:
            self._gpu_available = False
        if not self._gpu_available:
            self._gpu_bar.hide()
            self._vram_bar.hide()
            self._gpu_na_label.show()

    def start(self):
        if not self._timer.isActive():
            self._poll()
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def _poll(self):
        try:
            import psutil
            if not hasattr(self, '_psutil_primed'):
                psutil.cpu_percent()
                self._psutil_primed = True
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_pct = mem.percent
            ram_used = mem.used / (1024 ** 3)
            ram_total = mem.total / (1024 ** 3)
            self._cpu_bar.set_value(cpu_pct, f"{cpu_pct:.0f}%")
            self._ram_bar.set_value(
                ram_pct,
                get_text("sys_used_of", self.lang).format(
                    used=_fmt_bytes(ram_used), total=_fmt_bytes(ram_total)
                ),
            )
        except ImportError:
            self._cpu_bar.set_value(0, "psutil N/A")
            self._ram_bar.set_value(0, "psutil N/A")

        if self._gpu_available:
            try:
                import torch
                gpu_props = torch.cuda.get_device_properties(0)
                total_vram = gpu_props.total_memory / (1024 ** 3)
                used_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
                vram_pct = (used_vram / total_vram * 100) if total_vram > 0 else 0

                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=3,
                        **_NO_WINDOW_KW,
                    )
                    gpu_util = float(result.stdout.strip().split("\n")[0])
                except Exception:
                    gpu_util = 0

                self._gpu_bar.set_value(gpu_util, f"{gpu_util:.0f}%")
                self._vram_bar.set_value(
                    vram_pct,
                    get_text("sys_used_of", self.lang).format(
                        used=_fmt_bytes(used_vram), total=_fmt_bytes(total_vram)
                    ),
                )
            except Exception:
                self._gpu_bar.set_value(0, "N/A")
                self._vram_bar.set_value(0, "N/A")

    def refresh_styles(self):
        self._apply_frame_style()
        self._title_lbl.setStyleSheet(
            f"color: {theme.ORANGE}; border: none; background: transparent;"
        )
        self._gpu_na_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent;"
        )
        for bar in (self._cpu_bar, self._ram_bar, self._gpu_bar, self._vram_bar):
            bar.refresh_styles()

    def update_language(self, lang: str):
        self.lang = lang
        self._title_lbl.setText(get_text("sys_monitor_title", lang))
        self._cpu_bar.set_label(get_text("sys_cpu", lang))
        self._ram_bar.set_label(get_text("sys_ram", lang))
        self._gpu_bar.set_label(get_text("sys_gpu", lang))
        self._vram_bar.set_label(get_text("sys_vram", lang))
        self._gpu_na_label.setText(get_text("sys_gpu_not_available", lang))


# ═══════════════════════════════════════════════════════════════════════════
#  Compact Horizontal Monitor Bar (for the topbar)
# ═══════════════════════════════════════════════════════════════════════════

class _MonitorPill(QFrame):
    """A single inline pill: ● label value."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self.setStyleSheet("background: transparent; border: none;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self._dot = QLabel("●")
        self._dot.setStyleSheet(
            f"color: {color}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent;"
        )
        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(11)}; "
            f"font-weight: 600; border: none; background: transparent;"
        )
        self._value = QLabel("—")
        self._value.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(11)}; "
            f"font-family: {theme.FONT_MONO}; font-weight: 600; "
            f"border: none; background: transparent;"
        )
        lay.addWidget(self._dot)
        lay.addWidget(self._label)
        lay.addWidget(self._value)

    def set_value(self, text: str):
        self._value.setText(text)

    def set_label(self, text: str):
        self._label.setText(text)

    def refresh_styles(self):
        self._dot.setStyleSheet(
            f"color: {self._color}; font-size: {theme.fs(10)}; "
            f"border: none; background: transparent;"
        )
        self._label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.fs(11)}; "
            f"font-weight: 600; border: none; background: transparent;"
        )
        self._value.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(11)}; "
            f"font-family: {theme.FONT_MONO}; font-weight: 600; "
            f"border: none; background: transparent;"
        )


class SystemMonitorBar(QFrame):
    """Horizontal compact monitor for the topbar.
    Layout: [● CPU 23%] · [● RAM 6.2/16 GB] · [● GPU 78%] · [● VRAM 4.1/24 GB]
    """

    def __init__(self, lang: str = "en", parent=None):
        super().__init__(parent)
        self.lang = lang
        self._gpu_available = False
        self._apply_frame_style()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(10)

        self._cpu_pill = _MonitorPill(get_text("sys_cpu", lang), theme.ORANGE, self)
        self._ram_pill = _MonitorPill(get_text("sys_ram", lang), "#5B9BD5", self)
        self._gpu_pill = _MonitorPill(get_text("sys_gpu", lang), theme.GREEN, self)
        self._vram_pill = _MonitorPill(get_text("sys_vram", lang), theme.YELLOW, self)

        self._sep1 = self._make_sep()
        self._sep2 = self._make_sep()
        self._sep3 = self._make_sep()

        lay.addWidget(self._cpu_pill)
        lay.addWidget(self._sep1)
        lay.addWidget(self._ram_pill)
        lay.addWidget(self._sep2)
        lay.addWidget(self._gpu_pill)
        lay.addWidget(self._sep3)
        lay.addWidget(self._vram_pill)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.setInterval(2000)

        self._detect_gpu()
        self._poll()

    def _make_sep(self) -> QLabel:
        sep = QLabel("·")
        sep.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; "
            f"border: none; background: transparent;"
        )
        return sep

    def _apply_frame_style(self):
        self.setStyleSheet(f"""
            SystemMonitorBar {{
                background-color: transparent;
                border: none;
            }}
        """)

    def _detect_gpu(self):
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except Exception:
            self._gpu_available = False
        if not self._gpu_available:
            self._gpu_pill.hide()
            self._vram_pill.hide()
            self._sep2.hide()
            self._sep3.hide()

    def start(self):
        if not self._timer.isActive():
            self._poll()
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def _poll(self):
        try:
            import psutil
            if not hasattr(self, '_psutil_primed'):
                psutil.cpu_percent()
                self._psutil_primed = True
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_used = mem.used / (1024 ** 3)
            ram_total = mem.total / (1024 ** 3)
            self._cpu_pill.set_value(f"{cpu_pct:.0f}%")
            self._ram_pill.set_value(f"{ram_used:.1f}/{ram_total:.0f} GB")
        except ImportError:
            self._cpu_pill.set_value("N/A")
            self._ram_pill.set_value("N/A")

        if self._gpu_available:
            try:
                import torch
                gpu_props = torch.cuda.get_device_properties(0)
                total_vram = gpu_props.total_memory / (1024 ** 3)
                used_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)

                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=3,
                        **_NO_WINDOW_KW,
                    )
                    gpu_util = float(result.stdout.strip().split("\n")[0])
                except Exception:
                    gpu_util = 0

                self._gpu_pill.set_value(f"{gpu_util:.0f}%")
                self._vram_pill.set_value(f"{used_vram:.1f}/{total_vram:.0f} GB")
            except Exception:
                self._gpu_pill.set_value("N/A")
                self._vram_pill.set_value("N/A")

    def refresh_styles(self):
        self._apply_frame_style()
        for pill in (self._cpu_pill, self._ram_pill, self._gpu_pill, self._vram_pill):
            pill.refresh_styles()
        for sep in (self._sep1, self._sep2, self._sep3):
            sep.setStyleSheet(
                f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; "
                f"border: none; background: transparent;"
            )

    def update_language(self, lang: str):
        self.lang = lang
        self._cpu_pill.set_label(get_text("sys_cpu", lang))
        self._ram_pill.set_label(get_text("sys_ram", lang))
        self._gpu_pill.set_label(get_text("sys_gpu", lang))
        self._vram_pill.set_label(get_text("sys_vram", lang))


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
                font-size: {theme.fs(18)}; border: none; border-radius: 4px;
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

        # ── Language (moved to top) ──────────────────────────────
        self._add_section("res_section_lang", color="#5B9BD5")
        self._lang_placeholder = QHBoxLayout()
        self._lay.addLayout(self._lang_placeholder)

        # ── GPU Section ──────────────────────────────────────────
        self._add_section("res_section_gpu", color=theme.ORANGE)

        self.gpu_cb = self._add_checkbox("res_gpu_enabled", "gpu_enabled")
        self.fp16_cb = self._add_checkbox("res_fp16", "fp16_enabled")

        self.vram_slider, self.vram_val = self._add_slider(
            "res_gpu_mem_limit", 10, 100, 5, "%",
        )

        # ── Batch / Throughput Section ───────────────────────────
        self._add_section("res_section_batch", color="#a78bfa")

        self.batch_slider, self.batch_val = self._add_slider(
            "res_batch_size", 1, 32, 1,
        )
        self.prefetch_slider, self.prefetch_val = self._add_slider(
            "res_prefetch_frames", 1, 120, 1,
        )

        # ── CPU / Threading Section ──────────────────────────────
        self._add_section("res_section_cpu", color="#56c6d0")

        self.threads_slider, self.threads_val = self._add_slider(
            "res_cpu_threads", 1, _CPU_COUNT, 1,
        )
        self.workers_slider, self.workers_val = self._add_slider(
            "res_decode_workers", 1, _CPU_COUNT, 1,
        )

        # ── Memory Section ───────────────────────────────────────
        self._add_section("res_section_memory", color=theme.YELLOW)

        self.ram_slider, self.ram_val = self._add_slider(
            "res_ram_limit", 512, 32768, 256, "MB",
        )

        # ── Misc Performance ─────────────────────────────────────
        self._add_section("res_section_misc", color=theme.GREEN)

        self.async_cb = self._add_checkbox("res_async_save", "async_save")
        self.gc_cb = self._add_checkbox("res_auto_gc", "auto_gc")
        self.jpeg_slider, self.jpeg_val = self._add_slider(
            "res_jpeg_quality", 50, 100, 5,
        )

        # ── Theme / UI ──────────────────────────────────────────
        self._add_section("res_section_theme", color="#ec4899")

        self.theme_cb = self._add_checkbox("res_light_mode", "theme_mode")
        self.font_slider, self.font_val = self._add_slider(
            "res_font_scale", 80, 140, 5, "%",
        )

        # Accent color swatches + custom picker
        self._accent_label = QLabel(get_text("res_accent_color", self.lang))
        self._accent_label.setStyleSheet(theme.label_default())
        self._lay.addWidget(self._accent_label)

        accent_row = QHBoxLayout()
        accent_row.setSpacing(8)
        accent_row.setContentsMargins(0, 2, 0, 6)
        self._accent_btns = []
        self._selected_accent = theme.get_accent()
        for color, name in theme.ACCENT_PRESETS:
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(name)
            btn.clicked.connect(lambda _, c=color: self._pick_accent(c))
            accent_row.addWidget(btn)
            self._accent_btns.append((btn, color))

        self._custom_accent_btn = QPushButton("🎨")
        self._custom_accent_btn.setFixedSize(28, 28)
        self._custom_accent_btn.setCursor(Qt.PointingHandCursor)
        self._custom_accent_btn.setToolTip(get_text("res_accent_custom", self.lang))
        self._custom_accent_btn.clicked.connect(self._pick_custom_accent)
        accent_row.addWidget(self._custom_accent_btn)
        accent_row.addStretch()
        self._lay.addLayout(accent_row)
        self._refresh_accent_swatches()

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

    def _add_section(self, key: str, color: str = None):
        color = color or theme.ORANGE
        lbl = QLabel(get_text(key, self.lang))
        lbl.setStyleSheet(
            f"color: {color}; font-size: {theme.fs(11)}; font-weight: 700; "
            f"letter-spacing: 0.06em; text-transform: uppercase; "
            f"margin-top: 12px; margin-bottom: 2px; padding-bottom: 4px; "
            f"border-bottom: 1px solid {color};"
        )
        self._lay.addWidget(lbl)
        # Store for language update + theme refresh
        if not hasattr(self, "_section_labels"):
            self._section_labels = []
        self._section_labels.append((lbl, key, color))

    def _add_checkbox(self, text_key: str, setting_key: str) -> QCheckBox:
        cb = QCheckBox(get_text(text_key, self.lang))
        cb.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; padding: 3px 0;")
        tooltip_key = text_key + "_tooltip"
        tip = get_text(tooltip_key, self.lang)
        if tip != tooltip_key:
            cb.setToolTip(tip)
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
        tooltip_key = text_key + "_tooltip"
        tip = get_text(tooltip_key, self.lang)
        if tip != tooltip_key:
            lbl.setToolTip(tip)
        row.addWidget(lbl)

        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(mn)
        sl.setMaximum(mx)
        sl.setSingleStep(step)
        sl.setStyleSheet(theme.slider())
        if tip != tooltip_key:
            sl.setToolTip(tip)
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
        self.theme_cb.setChecked(s.get("theme_mode", "dark") == "light")
        self.font_slider.setValue(s.get("font_scale", 100))
        self._selected_accent = s.get("accent", theme.get_accent())
        self._refresh_accent_swatches()

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
            "theme_mode": "light" if self.theme_cb.isChecked() else "dark",
            "font_scale": self.font_slider.value(),
            "accent": self._selected_accent,
        }

    # ─── Accent color helpers ────────────────────────────────────────────

    def _accent_swatch_style(self, color: str, selected: bool) -> str:
        border = (f"2px solid {theme.TEXT_PRIMARY}"
                  if selected else f"1px solid {theme.BORDER}")
        return f"""
            QPushButton {{
                background-color: {color};
                border: {border};
                border-radius: 14px;
            }}
            QPushButton:hover {{
                border: 2px solid {theme.TEXT_PRIMARY};
            }}
        """

    def _refresh_accent_swatches(self):
        current = (self._selected_accent or "").lower()
        for btn, color in self._accent_btns:
            btn.setStyleSheet(
                self._accent_swatch_style(color, selected=(color.lower() == current))
            )
        # Custom button: preview the current accent if it's not in presets
        preset_colors = {c.lower() for c, _ in theme.ACCENT_PRESETS}
        is_custom = current and current not in preset_colors
        custom_bg = self._selected_accent if is_custom else "transparent"
        custom_border = (f"2px solid {theme.TEXT_PRIMARY}"
                         if is_custom else f"1px solid {theme.BORDER}")
        self._custom_accent_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {custom_bg};
                border: {custom_border};
                border-radius: 14px;
                font-size: {theme.fs(12)};
            }}
            QPushButton:hover {{
                border: 2px solid {theme.TEXT_PRIMARY};
            }}
        """)

    def _pick_accent(self, color: str):
        self._selected_accent = color
        self._refresh_accent_swatches()

    def _pick_custom_accent(self):
        from PyQt5.QtWidgets import QColorDialog
        initial = QColor(self._selected_accent or theme.get_accent())
        col = QColorDialog.getColor(
            initial, self, get_text("res_accent_custom", self.lang)
        )
        if col.isValid():
            self._pick_accent(col.name())

    def _apply(self):
        data = self._collect_values()
        self._settings = data
        save_settings(data)
        self.settings_changed.emit(data)
        self.close_drawer()

    def _reset_defaults(self):
        self._settings = _defaults()
        self._load_values()

    def embed_lang_combo(self, combo):
        """Host an external language combo inside the drawer."""
        self._lang_placeholder.addWidget(combo)
        self._lang_placeholder.addStretch()

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
        # Use a one-shot connection to avoid signal accumulation on rapid toggles
        try:
            self._anim.finished.disconnect(self._on_close_done)
        except (TypeError, RuntimeError):
            pass
        self._anim.finished.connect(self._on_close_done)
        self._anim.start()

    def _on_close_done(self):
        try:
            self._anim.finished.disconnect(self._on_close_done)
        except (TypeError, RuntimeError):
            pass
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

    # ─── Theme refresh ──────────────────────────────────────────────────

    def refresh_styles(self):
        """Re-apply all stylesheets after a theme change."""
        self.setStyleSheet(f"""
            ResourceSettingsDrawer {{
                background-color: {theme.BG_PANEL};
                border-left: 2px solid {theme.BORDER_ACCENT};
            }}
        """)
        self._title.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self._subtitle.setStyleSheet(theme.label_muted())
        self._close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {theme.TEXT_SECONDARY};
                font-size: {theme.fs(18)}; border: none; border-radius: 4px;
            }}
            QPushButton:hover {{ background: {theme.BG_HOVER}; color: {theme.RED}; }}
        """)
        self._reset_btn.setStyleSheet(theme.btn_secondary())
        self._apply_btn.setStyleSheet(theme.btn_primary())
        for lbl, _key, color in self._section_labels:
            lbl.setStyleSheet(
                f"color: {color}; font-size: {theme.fs(11)}; font-weight: 700; "
                f"letter-spacing: 0.06em; text-transform: uppercase; "
                f"margin-top: 12px; margin-bottom: 2px; padding-bottom: 4px; "
                f"border-bottom: 1px solid {color};"
            )
        for cb, _key in self._cb_keys:
            cb.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; padding: 3px 0;")
        for lbl, _key in self._slider_keys:
            lbl.setStyleSheet(theme.label_default())
        if hasattr(self, "_accent_label"):
            self._accent_label.setStyleSheet(theme.label_default())
            self._refresh_accent_swatches()

    # ─── Language update ─────────────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self._title.setText(get_text("res_title", lang))
        self._subtitle.setText(get_text("res_subtitle", lang))
        self._reset_btn.setText(get_text("res_reset", lang))
        self._apply_btn.setText(get_text("res_apply", lang))
        if hasattr(self, "_accent_label"):
            self._accent_label.setText(get_text("res_accent_color", lang))
            self._custom_accent_btn.setToolTip(get_text("res_accent_custom", lang))
        for lbl, key, _color in self._section_labels:
            lbl.setText(get_text(key, lang))
        for cb, key in self._cb_keys:
            cb.setText(get_text(key, lang))
            tip_key = key + "_tooltip"
            tip = get_text(tip_key, lang)
            cb.setToolTip(tip if tip != tip_key else "")
        for lbl, key in self._slider_keys:
            lbl.setText(get_text(key, lang))
            tip_key = key + "_tooltip"
            tip = get_text(tip_key, lang)
            lbl.setToolTip(tip if tip != tip_key else "")
