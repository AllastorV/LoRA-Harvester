"""
Unified Theme for LoRA-Harvester v2.0
Color Palette: Black, Gray, Orange (Dark) / White, Gray, Orange (Light)
All UI components reference this module for consistent styling.
Supports runtime theme switching via set_theme().
"""

import json
from pathlib import Path as _Path

# ═══════════════════════════════════════════════════════════
#  THEME STATE — persisted in theme_prefs.json
# ═══════════════════════════════════════════════════════════

_PREFS_PATH = _Path(__file__).resolve().parents[2] / "theme_prefs.json"

_DARK_PALETTE = {
    "BG_WINDOW":    "#1e1e1e",
    "BG_DEEPEST":   "#121212",
    "BG_DEEP":      "#161616",
    "BG_DARK":      "#1a1a1a",
    "BG_PANEL":     "#1e1e1e",
    "BG_CARD":      "#232323",
    "BG_SURFACE":   "#252525",
    "BG_HOVER":     "#2d2d2d",
    "BG_ELEVATED":  "#353535",
    "ORANGE":       "#e8832a",
    "ORANGE_LIGHT": "#f5a623",
    "ORANGE_DARK":  "#c96f1e",
    "ORANGE_GLOW":  "#ff9f43",
    "ORANGE_DIM":   "#8b5e2f",
    "ORANGE_SUBTLE":"#3d2a14",
    "TEXT_PRIMARY": "#e0e0e0",
    "TEXT_SECONDARY":"#a0a0a0",
    "TEXT_MUTED":   "#666666",
    "TEXT_ACCENT":  "#f5a623",
    "BORDER":       "#333333",
    "BORDER_LIGHT": "#444444",
    "BORDER_ACCENT":"#e8832a",
    "RED":          "#c0392b",
    "RED_HOVER":    "#a93226",
    "DISABLED_BG":  "#2a2a2a",
    "DISABLED_TEXT":"#555555",
}

_LIGHT_PALETTE = {
    "BG_WINDOW":    "#f5f5f5",
    "BG_DEEPEST":   "#ffffff",
    "BG_DEEP":      "#fafafa",
    "BG_DARK":      "#eeeeee",
    "BG_PANEL":     "#f5f5f5",
    "BG_CARD":      "#ffffff",
    "BG_SURFACE":   "#e8e8e8",
    "BG_HOVER":     "#e0e0e0",
    "BG_ELEVATED":  "#d5d5d5",
    "ORANGE":       "#e8832a",
    "ORANGE_LIGHT": "#f5a623",
    "ORANGE_DARK":  "#c96f1e",
    "ORANGE_GLOW":  "#ff9f43",
    "ORANGE_DIM":   "#d4a574",
    "ORANGE_SUBTLE":"#fff3e6",
    "TEXT_PRIMARY": "#1a1a1a",
    "TEXT_SECONDARY":"#555555",
    "TEXT_MUTED":   "#888888",
    "TEXT_ACCENT":  "#c96f1e",
    "BORDER":       "#cccccc",
    "BORDER_LIGHT": "#bbbbbb",
    "BORDER_ACCENT":"#e8832a",
    "RED":          "#c0392b",
    "RED_HOVER":    "#a93226",
    "DISABLED_BG":  "#e0e0e0",
    "DISABLED_TEXT":"#aaaaaa",
}

# Current mode: "dark" or "light"
_current_mode = "dark"
_font_scale = 1.0  # 0.8 .. 1.4

def _load_prefs():
    global _current_mode, _font_scale
    if _PREFS_PATH.exists():
        try:
            d = json.loads(_PREFS_PATH.read_text("utf-8"))
            _current_mode = d.get("mode", "dark")
            _font_scale = max(0.8, min(1.4, d.get("font_scale", 1.0)))
        except Exception:
            pass

def save_prefs():
    try:
        _PREFS_PATH.write_text(json.dumps({
            "mode": _current_mode,
            "font_scale": _font_scale,
        }, indent=2), "utf-8")
    except Exception:
        pass

_load_prefs()  # run once at import

def set_theme(mode: str = "dark", font_scale: float = 1.0):
    """Switch palette and update module-level constants."""
    global _current_mode, _font_scale
    _current_mode = mode if mode in ("dark", "light") else "dark"
    _font_scale = max(0.8, min(1.4, font_scale))
    pal = _LIGHT_PALETTE if _current_mode == "light" else _DARK_PALETTE
    g = globals()
    for k, v in pal.items():
        g[k] = v
    save_prefs()

def get_mode() -> str:
    return _current_mode

def get_font_scale() -> float:
    return _font_scale

def fs(base: int) -> str:
    """Scale a font-size value and return CSS string."""
    return f"{max(8, int(base * _font_scale))}px"

# Apply the loaded palette on import
set_theme(_current_mode, _font_scale)

# ═══════════════════════════════════════════════════════════
#  BACKWARDS-COMPAT ALIASES — these are updated by set_theme()
# ═══════════════════════════════════════════════════════════

def global_stylesheet() -> str:
    """Main window-level stylesheet (applied once at root)"""
    return f"""
        QMainWindow {{
            background-color: {BG_WINDOW};
        }}
        QWidget {{
            background-color: {BG_WINDOW};
            color: {TEXT_PRIMARY};
        }}
        QScrollArea {{
            background-color: {BG_WINDOW};
            border: none;
        }}
        QLabel {{
            color: {TEXT_PRIMARY};
            background-color: transparent;
        }}
        QToolTip {{
            background-color: {BG_DARK};
            color: {TEXT_PRIMARY};
            border: 1px solid {ORANGE};
            padding: 8px;
            border-radius: 4px;
            font-size: 18px;
        }}
        QCheckBox {{
            color: {TEXT_PRIMARY};
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {BORDER_LIGHT};
            border-radius: 4px;
            background-color: {BG_PANEL};
        }}
        QCheckBox::indicator:checked {{
            background-color: {ORANGE};
            border-color: {ORANGE};
            image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBvbHlsaW5lIHBvaW50cz0iMjAgNiA5IDE3IDQgMTIiPjwvcG9seWxpbmU+PC9zdmc+);
        }}
        QCheckBox::indicator:hover {{
            border-color: {ORANGE};
        }}
        QScrollBar:vertical {{
            background: {BG_PANEL};
            width: 8px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER_LIGHT};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {ORANGE_DIM};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
    """


# ──────────── Navigation ────────────

def page_btn_active() -> str:
    return f"""
        QPushButton {{
            background-color: {ORANGE};
            color: white;
            border: none;
            padding: 12px 25px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 6px;
            margin-right: 5px;
        }}
    """

def page_btn_inactive() -> str:
    return f"""
        QPushButton {{
            background-color: {BG_PANEL};
            color: {TEXT_MUTED};
            border: 2px solid {BORDER};
            padding: 12px 25px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 6px;
            margin-right: 5px;
        }}
        QPushButton:hover {{
            background-color: {BG_HOVER};
            color: {TEXT_PRIMARY};
            border-color: {ORANGE};
        }}
    """


# ──────────── Collapsible Panels ────────────

def collapsible_btn() -> str:
    return f"""
        QPushButton {{
            background-color: {BG_DARK};
            color: {TEXT_PRIMARY};
            border: 2px solid {BORDER};
            padding: 10px 15px;
            font-size: 13px;
            font-weight: bold;
            border-radius: 6px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {BG_HOVER};
            border-color: {ORANGE};
        }}
        QPushButton:checked {{
            background-color: {ORANGE_DARK};
            border-color: {ORANGE};
            color: white;
        }}
    """


# ──────────── Group Boxes ────────────

def group_box() -> str:
    return f"""
        QGroupBox {{
            font-size: 14px;
            font-weight: bold;
            border: 2px solid {BORDER};
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 15px;
            background-color: {BG_PANEL};
            color: {ORANGE_LIGHT};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {ORANGE_LIGHT};
        }}
    """

def panel_group() -> str:
    """Sub-panel group boxes (quality, caption, tag panels)"""
    return f"""
        QGroupBox {{
            font-size: 12px;
            font-weight: bold;
            border: 2px solid {BORDER};
            border-radius: 6px;
            margin-top: 5px;
            padding-top: 10px;
            background-color: {BG_PANEL};
            color: {ORANGE_LIGHT};
        }}
        QGroupBox::title {{
            color: {ORANGE_LIGHT};
        }}
    """


# ──────────── Input Controls ────────────

def combo() -> str:
    return f"""
        QComboBox {{
            padding: 8px;
            border: 2px solid {BORDER};
            border-radius: 5px;
            background-color: {BG_DEEP};
            color: {TEXT_PRIMARY};
            font-weight: bold;
            min-width: 100px;
        }}
        QComboBox:hover {{
            border-color: {ORANGE};
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_HOVER};
            color: {TEXT_PRIMARY};
            selection-background-color: {ORANGE};
            selection-color: white;
        }}
    """

def combo_compact() -> str:
    """Smaller combo for dense panels"""
    return f"""
        QComboBox {{
            padding: 5px;
            border: 1px solid {BORDER};
            border-radius: 3px;
            background-color: {BG_DEEP};
            color: {TEXT_PRIMARY};
            min-width: 120px;
        }}
        QComboBox:hover {{
            border-color: {ORANGE};
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_HOVER};
            color: {TEXT_PRIMARY};
            selection-background-color: {ORANGE};
            selection-color: white;
        }}
    """

def spinbox() -> str:
    return f"""
        QSpinBox, QDoubleSpinBox {{
            padding: 5px;
            border: 2px solid {BORDER};
            border-radius: 5px;
            background-color: {BG_DEEP};
            color: {TEXT_PRIMARY};
            font-weight: bold;
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {ORANGE};
        }}
    """

def spinbox_compact() -> str:
    """Smaller spinbox for dense panels"""
    return f"""
        QSpinBox, QDoubleSpinBox {{
            padding: 3px;
            border: 1px solid {BORDER};
            border-radius: 3px;
            background-color: {BG_DEEP};
            color: {TEXT_PRIMARY};
            min-width: 60px;
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {ORANGE};
        }}
    """

def slider() -> str:
    return f"""
        QSlider::groove:horizontal {{
            background: {BG_DARK};
            height: 8px;
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background: {ORANGE};
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {ORANGE_LIGHT};
        }}
        QSlider::sub-page:horizontal {{
            background: {ORANGE_DIM};
            border-radius: 4px;
        }}
    """

def line_edit() -> str:
    return f"""
        QLineEdit {{
            padding: 8px 12px;
            border: 2px solid {BORDER};
            border-radius: 5px;
            background-color: {BG_DEEP};
            color: {TEXT_PRIMARY};
        }}
        QLineEdit:hover, QLineEdit:focus {{
            border-color: {ORANGE};
        }}
    """

def line_edit_compact() -> str:
    return f"""
        QLineEdit {{
            padding: 5px;
            border: 1px solid {BORDER};
            border-radius: 3px;
            background-color: {BG_DEEP};
            color: {TEXT_PRIMARY};
        }}
        QLineEdit:hover, QLineEdit:focus {{
            border-color: {ORANGE};
        }}
    """

def text_edit_input() -> str:
    """Multi-line text area for user input (negative tags, etc.)"""
    return f"""
        QTextEdit {{
            border: 2px solid {BORDER};
            border-radius: 5px;
            background-color: {BG_DEEP};
            color: {TEXT_PRIMARY};
            padding: 5px;
        }}
        QTextEdit:hover, QTextEdit:focus {{
            border-color: {ORANGE};
        }}
    """


# ──────────── Buttons ────────────

def btn_primary() -> str:
    """Main action button (orange)"""
    return f"""
        QPushButton {{
            background-color: {ORANGE};
            color: white;
            border: none;
            padding: 15px;
            font-size: 16px;
            border-radius: 6px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {ORANGE_GLOW};
        }}
        QPushButton:pressed {{
            background-color: {ORANGE_DARK};
        }}
        QPushButton:disabled {{
            background-color: {DISABLED_BG};
            color: {DISABLED_TEXT};
        }}
    """

def btn_danger() -> str:
    """Stop / danger button"""
    return f"""
        QPushButton {{
            background-color: {RED};
            color: white;
            border: none;
            padding: 15px;
            font-size: 16px;
            border-radius: 6px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {RED_HOVER};
        }}
        QPushButton:disabled {{
            background-color: {DISABLED_BG};
            color: {DISABLED_TEXT};
        }}
    """

def btn_secondary() -> str:
    """Secondary / outline button (orange outline)"""
    return f"""
        QPushButton {{
            background-color: {BG_SURFACE};
            color: {ORANGE_LIGHT};
            border: 2px solid {ORANGE};
            padding: 15px;
            font-size: 16px;
            border-radius: 6px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {BG_HOVER};
            color: white;
        }}
    """

def btn_browse() -> str:
    """Browse / folder selection button"""
    return f"""
        QPushButton {{
            background-color: {ORANGE};
            color: white;
            border: none;
            padding: 12px;
            font-size: 14px;
            border-radius: 5px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {ORANGE_GLOW};
        }}
        QPushButton:pressed {{
            background-color: {ORANGE_DARK};
        }}
    """


# ──────────── Progress & Log ────────────

def progress_bar() -> str:
    return f"""
        QProgressBar {{
            border: 2px solid {BORDER};
            border-radius: 6px;
            text-align: center;
            height: 25px;
            background-color: {BG_DARK};
            color: {TEXT_PRIMARY};
            font-weight: bold;
        }}
        QProgressBar::chunk {{
            background-color: {ORANGE};
            border-radius: 4px;
        }}
    """

def log_area() -> str:
    return f"""
        QTextEdit {{
            background-color: {BG_DEEPEST};
            color: {ORANGE_LIGHT};
            border: 2px solid {BORDER};
            border-radius: 6px;
            padding: 10px;
            font-family: 'Consolas', 'Cascadia Code', monospace;
            font-size: 17px;
        }}
    """


# ──────────── Drop Zones ────────────

def drop_zone_default() -> str:
    return f"""
        QLabel {{
            border: 3px dashed {ORANGE_DIM};
            border-radius: 10px;
            background-color: {BG_PANEL};
            color: {TEXT_SECONDARY};
            font-size: 14px;
            padding: 20px;
        }}
        QLabel:hover {{
            background-color: {BG_HOVER};
            border-color: {ORANGE};
            color: {ORANGE_LIGHT};
        }}
    """

def drop_zone_active() -> str:
    """When a file is being dragged over"""
    return f"""
        QLabel {{
            border: 3px dashed {ORANGE};
            border-radius: 10px;
            background-color: {ORANGE_SUBTLE};
            color: {ORANGE_LIGHT};
            font-size: 14px;
            padding: 20px;
        }}
    """

def drop_zone_frame_default() -> str:
    """Drop zone frame (captioning page)"""
    return f"""
        QFrame {{
            background: {BG_DARK};
            border: 2px dashed {BORDER_LIGHT};
            border-radius: 8px;
        }}
        QFrame:hover {{
            border-color: {ORANGE};
            background: {BG_SURFACE};
        }}
    """

def drop_zone_frame_active() -> str:
    return f"""
        QFrame {{
            background: {ORANGE_SUBTLE};
            border: 2px dashed {ORANGE};
            border-radius: 8px;
        }}
    """

def drop_zone_frame_success() -> str:
    return f"""
        QFrame {{
            background: {ORANGE_SUBTLE};
            border: 2px solid {ORANGE};
            border-radius: 8px;
        }}
    """


# ──────────── Card Frames ────────────

def card_frame() -> str:
    return f"""
        QFrame {{
            background-color: {BG_PANEL};
            border: 2px solid {BORDER};
            border-radius: 10px;
            padding: 15px;
        }}
    """

def card_frame_compact() -> str:
    return f"""
        QFrame {{
            background-color: {BG_PANEL};
            border: 2px solid {BORDER};
            border-radius: 10px;
            padding: 10px;
        }}
    """


# ──────────── Tab Widget ────────────

def tab_widget() -> str:
    return f"""
        QTabWidget::pane {{
            border: 2px solid {BORDER};
            border-radius: 5px;
            background-color: {BG_PANEL};
        }}
        QTabBar::tab {{
            background-color: {BG_DARK};
            color: {TEXT_SECONDARY};
            padding: 8px 15px;
            margin-right: 2px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }}
        QTabBar::tab:selected {{
            background-color: {ORANGE};
            color: white;
        }}
        QTabBar::tab:hover {{
            background-color: {BG_HOVER};
            color: {TEXT_PRIMARY};
        }}
    """


# ──────────── Inline Helpers ────────────

def info_icon() -> str:
    """Style for ℹ️ info icons"""
    return f"color: {ORANGE}; font-size: 14px;"

def label_default() -> str:
    return f"color: {TEXT_PRIMARY};"

def label_accent() -> str:
    return f"color: {ORANGE_LIGHT}; font-weight: bold;"

def label_muted() -> str:
    return f"color: {TEXT_SECONDARY}; font-size: 18px; margin-bottom: 20px;"

def label_title() -> str:
    return f"color: {ORANGE_LIGHT}; margin: 20px;"

def label_section() -> str:
    """Section title inside cards"""
    return f"color: {ORANGE_LIGHT}; border: none; font-weight: bold;"

def label_value() -> str:
    return f"font-weight: bold; color: {TEXT_PRIMARY}; min-width: 30px;"

def label_transparent() -> str:
    """For labels inside frames (no border inheritance)"""
    return f"color: {TEXT_SECONDARY}; font-size: 12px; border: none; background: transparent;"

def label_success() -> str:
    return f"color: {ORANGE_LIGHT}; font-weight: bold; font-size: 13px; border: none;"

def icon_transparent() -> str:
    return f"font-size: 20px; border: none; background: transparent;"

def label_frame() -> str:
    """Label inside a styled frame (needs border: none to avoid frame cascade)"""
    return f"color: {TEXT_PRIMARY}; border: none;"

def info_icon_frame() -> str:
    """Info icon inside a styled frame"""
    return f"color: {ORANGE}; font-size: 14px; border: none; margin-right: 5px;"

def info_icon_frame_compact() -> str:
    """Info icon inside a frame without margin"""
    return f"color: {ORANGE}; font-size: 14px; border: none;"

def checkbox_frame() -> str:
    """Checkbox inside a styled frame"""
    return f"color: {TEXT_PRIMARY}; border: none;"
