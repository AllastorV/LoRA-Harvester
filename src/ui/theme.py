"""
Unified Theme for LoRA-Harvester v4.0
Design Language: Dark-mode SaaS dashboard (Linear / Vercel / Raycast)
Single accent: #e8832a amber-orange.  Minimal, info-dense, professional tool.
Supports runtime theme switching via set_theme().
"""

import json
from pathlib import Path as _Path
from src.ui.theme_bindings import bind_style, refresh_styles, set_style_if_changed, render_style

# ═══════════════════════════════════════════════════════════
#  THEME STATE — persisted in theme_prefs.json
# ═══════════════════════════════════════════════════════════

_PREFS_PATH = _Path(__file__).resolve().parents[2] / "data" / "theme_prefs.json"

_DARK_PALETTE = {
    "BG_WINDOW":     "#0f0f10",
    "BG_DEEPEST":    "#080809",
    "BG_DEEP":       "#0d0d0f",
    "BG_DARK":       "#111113",
    "BG_PANEL":      "#1a1714",
    "BG_CARD":       "#1c1410",
    "BG_SURFACE":    "#2c2018",
    "BG_HOVER":      "#38291e",
    "BG_ELEVATED":   "#443a32",
    "ORANGE":        "#e8832a",
    "ORANGE_LIGHT":  "#ffb782",
    "ORANGE_DARK":   "#c96f1e",
    "ORANGE_GLOW":   "#f09030",
    "ORANGE_DIM":    "#703800",
    "ORANGE_SUBTLE": "rgba(232,131,42,0.12)",
    "TEXT_PRIMARY":  "#f5e8e0",
    "TEXT_SECONDARY":"#e2cfbf",
    "TEXT_MUTED":    "#d0b9a8",
    "TEXT_ACCENT":   "#ffb782",
    "BORDER":        "#383838",
    "BORDER_LIGHT":  "#4a403a",
    "BORDER_ACCENT": "#e8832a",
    "RED":           "#ef4444",
    "RED_HOVER":     "#dc2626",
    "GREEN":         "#6fb35a",
    "YELLOW":        "#e5c07b",
    "BLUE":          "#7cd0ff",
    "DISABLED_BG":   "#1e1e22",
    "DISABLED_TEXT": "#6e6e78",
    "SIDEBAR_BG":    "#18181b",
    "SIDEBAR_HOVER": "rgba(255,255,255,0.05)",
    "SIDEBAR_ACTIVE":"rgba(232,131,42,0.10)",
    "NAV_LABEL":     "#b89d8e",
}

_LIGHT_PALETTE = {
    "BG_WINDOW":     "#fafaf9",
    "BG_DEEPEST":    "#ffffff",
    "BG_DEEP":       "#f5f5f3",
    "BG_DARK":       "#f0f0ee",
    "BG_PANEL":      "#ebebea",
    "BG_CARD":       "#ffffff",
    "BG_SURFACE":    "#e8e8e5",
    "BG_HOVER":      "#e0e0dd",
    "BG_ELEVATED":   "#d5d5d2",
    "ORANGE":        "#e8832a",
    "ORANGE_LIGHT":  "#d47a25",
    "ORANGE_DARK":   "#b86a1c",
    "ORANGE_GLOW":   "#f09030",
    "ORANGE_DIM":    "#d4a574",
    "ORANGE_SUBTLE": "rgba(232,131,42,0.08)",
    "TEXT_PRIMARY":  "#1a1918",
    "TEXT_SECONDARY":"#3d3a36",
    "TEXT_MUTED":    "#8a8884",
    "TEXT_ACCENT":   "#c96f1e",
    "BORDER":        "#e0dfdb",
    "BORDER_LIGHT":  "#cccbc7",
    "BORDER_ACCENT": "#e8832a",
    "RED":           "#dc2626",
    "RED_HOVER":     "#b91c1c",
    "GREEN":         "#5c9a48",
    "YELLOW":        "#c9a84e",
    "BLUE":          "#00a8e2",
    "DISABLED_BG":   "#e8e8e5",
    "DISABLED_TEXT": "#b0afa9",
    "SIDEBAR_BG":    "#f0f0ee",
    "SIDEBAR_HOVER": "rgba(0,0,0,0.03)",
    "SIDEBAR_ACTIVE":"rgba(232,131,42,0.08)",
    "NAV_LABEL":     "#8a8884",
}

_DEFAULT_ACCENT = "#e8832a"
ACCENT_PRESETS = [
    ("#e8832a", "Orange"),
    ("#3b82f6", "Blue"),
    ("#10b981", "Green"),
    ("#a855f7", "Purple"),
    ("#ec4899", "Pink"),
]

_current_mode = "dark"
_font_scale = 1.0
_current_accent = _DEFAULT_ACCENT
_current_lang = "en"


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r, g, b):
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _lighten(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(r + (255 - r) * amount,
                       g + (255 - g) * amount,
                       b + (255 - b) * amount)


def _darken(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(r * (1 - amount), g * (1 - amount), b * (1 - amount))


def _accent_tint(accent: str, strength: float) -> str:
    """Create a very dark tinted background derived from the accent hue."""
    r, g, b = _hex_to_rgb(accent)
    return _rgb_to_hex(r * strength, g * strength, b * strength)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_accent(accent: str):
    """Derive the full accent family from a single base hex color."""
    g = globals()
    g["ORANGE"]        = accent
    g["ORANGE_LIGHT"]  = _lighten(accent, 0.15)
    g["ORANGE_DARK"]   = _darken(accent, 0.20)
    g["ORANGE_GLOW"]   = _lighten(accent, 0.10)
    g["ORANGE_DIM"]    = _darken(accent, 0.30)
    g["ORANGE_SUBTLE"] = _rgba(accent, 0.12)
    g["BORDER_ACCENT"] = accent
    g["TEXT_ACCENT"]   = _lighten(accent, 0.10)
    g["SIDEBAR_ACTIVE"]= _rgba(accent, 0.12)
    # Tint dark background surfaces with accent hue (dark mode only)
    import sys as _sys
    _mod = _sys.modules[__name__]
    if globals().get('_studio_design', False):
        pal = _LIGHT_PALETTE if _current_mode == 'light' else _DARK_PALETTE
        for name in ('BG_CARD', 'BG_SURFACE', 'BG_PANEL'):
            g[name] = pal[name]
        if _current_mode == 'light':
            g['ORANGE_LIGHT'] = _darken(accent, .12)
            g['TEXT_ACCENT'] = _darken(accent, .15)
    elif getattr(_mod, '_current_mode', 'dark') == "dark":
        g["BG_CARD"]    = _accent_tint(accent, 0.110)
        g["BG_SURFACE"] = _accent_tint(accent, 0.190)
        g["BG_PANEL"]   = _accent_tint(accent, 0.085)
    else:
        # Restore light palette values (don't tint in light mode)
        g["BG_CARD"]    = _LIGHT_PALETTE["BG_CARD"]
        g["BG_SURFACE"] = _LIGHT_PALETTE["BG_SURFACE"]
        g["BG_PANEL"]   = _LIGHT_PALETTE["BG_PANEL"]


def _load_prefs():
    global _current_mode, _font_scale, _current_accent, _current_lang
    if _PREFS_PATH.exists():
        try:
            d = json.loads(_PREFS_PATH.read_text("utf-8"))
            _current_mode = d.get("mode", "dark")
            _font_scale = max(0.8, min(1.4, d.get("font_scale", 1.0)))
            acc = d.get("accent", _DEFAULT_ACCENT)
            if isinstance(acc, str) and acc.startswith("#") and len(acc) in (4, 7):
                _current_accent = acc
            lang = d.get("lang", "en")
            if lang in ("en", "tr"):
                _current_lang = lang
        except Exception:
            pass

def save_prefs():
    try:
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PREFS_PATH.write_text(json.dumps({
            "mode": _current_mode,
            "font_scale": _font_scale,
            "accent": _current_accent,
            "lang": _current_lang,
        }, indent=2), "utf-8")
    except Exception:
        pass

_load_prefs()

def normalize_state(mode="dark", font_scale=1.0, accent=None):
    """Validate appearance settings and canonicalize #RGB/#RRGGBB colors."""
    import math
    import re
    mode = mode if mode in ("dark", "light") else "dark"
    try:
        scale = float(font_scale)
        if not math.isfinite(scale):
            scale = 1.0
    except (TypeError, ValueError):
        scale = 1.0
    scale = round(max(0.8, min(1.4, scale)), 4)
    color = accent if isinstance(accent, str) else _current_accent
    if not re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", color):
        color = _DEFAULT_ACCENT
    color = color.lower()
    if len(color) == 4:
        color = "#" + "".join(c * 2 for c in color[1:])
    return mode, scale, color


def state_key():
    return _current_mode, _font_scale, _current_accent


def set_theme(mode: str = "dark", font_scale: float = 1.0, accent: str = None,
              *, persist: bool = True):
    global _current_mode, _font_scale, _current_accent
    new_state = normalize_state(mode, font_scale, accent)
    changed = new_state != state_key() or "BG_WINDOW" not in globals()
    if not changed:
        return False
    _current_mode, _font_scale, _current_accent = new_state
    pal = _LIGHT_PALETTE if _current_mode == "light" else _DARK_PALETTE
    globals().update(pal)
    _apply_accent(_current_accent)
    if persist:
        save_prefs()
    return True


def apply_global_style(app):
    """Apply the global QSS once, at QApplication level (never on a window)."""
    if app is not None:
        return set_style_if_changed(app, global_stylesheet())
    return False


def get_mode() -> str:
    return _current_mode

def get_font_scale() -> float:
    return _font_scale

def get_accent() -> str:
    return _current_accent

def get_lang() -> str:
    return _current_lang

def set_lang(lang: str):
    """Persist the UI language choice ('en' or 'tr')."""
    global _current_lang
    if lang in ("en", "tr"):
        _current_lang = lang
        save_prefs()

_FONT_BASELINE = 1.30

def fs(base: int) -> str:
    return f"{max(12, int(base * _FONT_BASELINE * _font_scale))}px"

set_theme(_current_mode, _font_scale, _current_accent, persist=False)

# ═══════════════════════════════════════════════════════════
#  FONT STACKS
# ═══════════════════════════════════════════════════════════

FONT_BODY = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_MONO = "'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace"

# ═══════════════════════════════════════════════════════════
#  RADIUS TOKENS
# ═══════════════════════════════════════════════════════════

R = "10px"
R_SM = "6px"
R_LG = "14px"

# ═══════════════════════════════════════════════════════════
#  GLOBAL STYLESHEET
# ═══════════════════════════════════════════════════════════

def global_stylesheet() -> str:
    return f"""
        * {{
            font-family: {FONT_BODY};
            letter-spacing: -0.003em;
        }}
        QMainWindow {{
            background-color: {BG_WINDOW};
        }}
        QWidget {{
            background-color: {BG_WINDOW};
            color: {TEXT_PRIMARY};
            font-size: {fs(13)};
        }}
        QScrollArea {{
            background-color: transparent;
            border: none;
        }}
        QLabel {{
            color: {TEXT_PRIMARY};
            background-color: transparent;
        }}
        QTableView, QTableWidget, QListWidget {{
            background-color: {BG_CARD};
            alternate-background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            gridline-color: {BORDER};
            selection-background-color: {ORANGE_SUBTLE};
            selection-color: {TEXT_PRIMARY};
        }}
        QTableView::item:selected, QTableWidget::item:selected, QListWidget::item:selected {{
            background-color: {ORANGE_SUBTLE};
            color: {TEXT_PRIMARY};
        }}
        QHeaderView::section {{
            background-color: {BG_SURFACE};
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER};
        }}
        QToolTip {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_LIGHT};
            padding: 8px 10px;
            border-radius: {R_SM};
            font-size: {fs(12)};
        }}
        QCheckBox {{
            color: {TEXT_PRIMARY};
            spacing: 8px;
            font-size: {fs(12)};
            background: transparent;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {BORDER_LIGHT};
            border-radius: 4px;
            background-color: {BG_SURFACE};
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
            background: transparent;
            width: 6px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER_LIGHT};
            border-radius: 3px;
            min-height: 24px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {TEXT_MUTED};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 6px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal {{
            background: {BORDER_LIGHT};
            border-radius: 3px;
            min-width: 24px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {TEXT_MUTED};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        QComboBox {{
            combobox-popup: 0;
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER};
            selection-background-color: {ORANGE};
            selection-color: white;
            border-radius: 0px;
            padding: 2px;
            outline: 0px;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 6px 12px;
            min-height: 28px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {BG_HOVER};
        }}
    """


# ═══════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════

def sidebar_frame() -> str:
    return f"""
        QFrame {{
            background-color: {SIDEBAR_BG};
            border: none;
            border-right: 1px solid {BORDER};
        }}
    """

def sidebar_brand() -> str:
    return f"""
        color: {TEXT_PRIMARY};
        font-size: {fs(15)};
        font-weight: 700;
        letter-spacing: -0.02em;
        padding: 0;
        border: none;
        background: transparent;
    """

def sidebar_section_label() -> str:
    return f"""
        color: {NAV_LABEL};
        font-size: {fs(10)};
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0;
        border: none;
        background: transparent;
    """

def nav_btn_active() -> str:
    return f"""
        QPushButton {{
            background-color: {SIDEBAR_ACTIVE};
            color: {ORANGE};
            border-top: none;
            border-right: none;
            border-bottom: none;
            border-left: 2px solid {ORANGE};
            padding: 7px 14px;
            font-size: {fs(13)};
            font-weight: 600;
            border-radius: 0px;
            text-align: left;
        }}
    """

def nav_btn_inactive() -> str:
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {TEXT_SECONDARY};
            border-top: none;
            border-right: none;
            border-bottom: none;
            border-left: 2px solid transparent;
            padding: 7px 14px;
            font-size: {fs(13)};
            font-weight: 500;
            border-radius: 0px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {SIDEBAR_HOVER};
            color: {TEXT_PRIMARY};
        }}
    """

def page_btn_active() -> str:
    return nav_btn_active()

def page_btn_inactive() -> str:
    return nav_btn_inactive()


# ═══════════════════════════════════════════════════════════
#  TOPBAR
# ═══════════════════════════════════════════════════════════

def topbar_frame() -> str:
    return f"""
        QFrame {{
            background-color: {BG_WINDOW};
            border: none;
            border-bottom: 1px solid {BORDER};
        }}
    """


# ═══════════════════════════════════════════════════════════
#  COLLAPSIBLE PANELS
# ═══════════════════════════════════════════════════════════

def collapsible_btn() -> str:
    return f"""
        QPushButton {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER};
            padding: 9px 14px;
            font-size: {fs(12)};
            font-weight: 600;
            border-radius: {R_SM};
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {BG_HOVER};
            border-color: {BORDER_LIGHT};
        }}
        QPushButton:checked {{
            background-color: {SIDEBAR_ACTIVE};
            border-color: {ORANGE};
            color: {ORANGE};
        }}
    """


# ═══════════════════════════════════════════════════════════
#  GROUP BOXES
# ═══════════════════════════════════════════════════════════

def group_box() -> str:
    return f"""
        QGroupBox {{
            font-size: {fs(13)};
            font-weight: 600;
            border: 1px solid {BORDER};
            border-radius: {R};
            margin-top: 8px;
            padding-top: 14px;
            background-color: {BG_CARD};
            color: {ORANGE};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: {ORANGE};
        }}
    """

def panel_group() -> str:
    return f"""
        QGroupBox {{
            font-size: {fs(12)};
            font-weight: 600;
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            margin-top: 4px;
            padding-top: 10px;
            background-color: {BG_PANEL};
            color: {ORANGE};
        }}
        QGroupBox::title {{
            color: {ORANGE};
        }}
    """


# ═══════════════════════════════════════════════════════════
#  INPUT CONTROLS
# ═══════════════════════════════════════════════════════════

def combo() -> str:
    return f"""
        QComboBox {{
            padding: 7px 10px;
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            font-weight: 500;
            min-width: 100px;
            font-size: {fs(12)};
        }}
        QComboBox:hover {{
            border-color: {BORDER_LIGHT};
        }}
        QComboBox:focus {{
            border-color: {ORANGE};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_LIGHT};
            selection-background-color: {ORANGE};
            selection-color: white;
            border-radius: {R_SM};
            padding: 4px;
            outline: 0px;
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 28px;
            padding: 4px 10px;
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: {ORANGE};
            color: white;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {BG_HOVER};
        }}
    """

def combo_compact() -> str:
    return f"""
        QComboBox {{
            padding: 4px 8px;
            border: 1px solid {BORDER};
            border-radius: 4px;
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            min-width: 110px;
            font-size: {fs(11)};
        }}
        QComboBox:hover {{ border-color: {BORDER_LIGHT}; }}
        QComboBox:focus {{ border-color: {ORANGE}; }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
            selection-background-color: {ORANGE};
            selection-color: white;
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 26px;
            padding: 3px 8px;
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: {ORANGE};
            color: white;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {BG_HOVER};
        }}
    """

def spinbox() -> str:
    return f"""
        QSpinBox, QDoubleSpinBox {{
            padding: 5px 24px 5px 8px;
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            font-weight: 500;
            font-family: {FONT_MONO};
            font-size: {fs(12)};
            selection-background-color: {ORANGE};
            selection-color: white;
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {BORDER_LIGHT};
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {ORANGE};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 18px;
            background-color: {BG_HOVER};
            border-left: 1px solid {BORDER};
            border-top-right-radius: {R_SM};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 18px;
            background-color: {BG_HOVER};
            border-left: 1px solid {BORDER};
            border-top: 1px solid {BORDER};
            border-bottom-right-radius: {R_SM};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {BG_ELEVATED};
        }}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background-color: {ORANGE};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            width: 7px; height: 7px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            width: 7px; height: 7px;
        }}
    """

def spinbox_compact() -> str:
    return f"""
        QSpinBox, QDoubleSpinBox {{
            padding: 3px 6px;
            border: 1px solid {BORDER};
            border-radius: 4px;
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            min-width: 55px;
            font-family: {FONT_MONO};
            font-size: {fs(11)};
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {BORDER_LIGHT}; }}
        QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ORANGE}; }}
    """

def slider() -> str:
    return f"""
        QSlider {{
            background: transparent;
        }}
        QSlider::groove:horizontal {{
            background: {BG_SURFACE};
            height: 4px;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {ORANGE};
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {ORANGE_GLOW};
        }}
        QSlider::sub-page:horizontal {{
            background: {ORANGE_DIM};
            border-radius: 2px;
        }}
    """

def line_edit() -> str:
    return f"""
        QLineEdit {{
            padding: 7px 10px;
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            font-size: {fs(12)};
        }}
        QLineEdit:hover {{ border-color: {BORDER_LIGHT}; }}
        QLineEdit:focus {{ border-color: {ORANGE}; }}
    """

def line_edit_compact() -> str:
    return f"""
        QLineEdit {{
            padding: 4px 8px;
            border: 1px solid {BORDER};
            border-radius: 4px;
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            font-size: {fs(11)};
        }}
        QLineEdit:hover {{ border-color: {BORDER_LIGHT}; }}
        QLineEdit:focus {{ border-color: {ORANGE}; }}
    """

def text_edit_input() -> str:
    return f"""
        QTextEdit {{
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            padding: 6px;
            font-size: {fs(12)};
        }}
        QTextEdit:hover {{ border-color: {BORDER_LIGHT}; }}
        QTextEdit:focus {{ border-color: {ORANGE}; }}
    """


# ═══════════════════════════════════════════════════════════
#  BUTTONS
# ═══════════════════════════════════════════════════════════

def btn_primary() -> str:
    return f"""
        QPushButton {{
            background-color: {ORANGE};
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: {fs(13)};
            border-radius: {R_SM};
            font-weight: 600;
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
    return f"""
        QPushButton {{
            background-color: {RED};
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: {fs(13)};
            border-radius: {R_SM};
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {RED_HOVER}; }}
        QPushButton:disabled {{
            background-color: {DISABLED_BG};
            color: {DISABLED_TEXT};
        }}
    """

def btn_secondary() -> str:
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER};
            padding: 10px 20px;
            font-size: {fs(13)};
            border-radius: {R_SM};
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {BG_HOVER};
            color: {TEXT_PRIMARY};
            border-color: {BORDER_LIGHT};
        }}
    """

def btn_browse() -> str:
    return f"""
        QPushButton {{
            background-color: {ORANGE};
            color: white;
            border: none;
            padding: 9px 16px;
            font-size: {fs(12)};
            border-radius: {R_SM};
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {ORANGE_GLOW}; }}
        QPushButton:pressed {{ background-color: {ORANGE_DARK}; }}
    """

def btn_icon_square() -> str:
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER};
            padding: 0px;
            min-width: 32px;
            max-width: 32px;
            min-height: 32px;
            max-height: 32px;
            font-size: {fs(14)};
            border-radius: {R_SM};
        }}
        QPushButton:hover {{
            background-color: {BG_HOVER};
            color: {TEXT_PRIMARY};
            border-color: {BORDER_LIGHT};
        }}
    """

def btn_action_start() -> str:
    return f"""
        QPushButton {{
            background-color: {ORANGE};
            color: white;
            border: none;
            padding: 9px 24px;
            font-size: {fs(12)};
            border-radius: {R_SM};
            font-weight: 600;
            min-width: 130px;
        }}
        QPushButton:hover {{ background-color: {ORANGE_GLOW}; }}
        QPushButton:pressed {{ background-color: {ORANGE_DARK}; }}
        QPushButton:disabled {{
            background-color: {DISABLED_BG};
            color: {DISABLED_TEXT};
        }}
    """

def btn_action_pause() -> str:
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER};
            padding: 9px 16px;
            font-size: {fs(12)};
            border-radius: {R_SM};
            font-weight: 600;
            min-width: 95px;
        }}
        QPushButton:hover {{
            background-color: {BG_HOVER};
            border-color: {BORDER_LIGHT};
        }}
        QPushButton:disabled {{
            background-color: {DISABLED_BG};
            color: {DISABLED_TEXT};
            border-color: {DISABLED_BG};
        }}
    """

def btn_action_skip() -> str:
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {ORANGE};
            border: 1px solid {BORDER};
            padding: 9px 16px;
            font-size: {fs(12)};
            border-radius: {R_SM};
            font-weight: 600;
            min-width: 95px;
        }}
        QPushButton:hover {{
            background-color: {SIDEBAR_ACTIVE};
            border-color: {ORANGE_DIM};
        }}
        QPushButton:disabled {{
            color: {DISABLED_TEXT};
            border-color: {DISABLED_BG};
        }}
    """

def btn_action_stop() -> str:
    return f"""
        QPushButton {{
            background-color: {RED};
            color: white;
            border: none;
            padding: 9px 16px;
            font-size: {fs(12)};
            border-radius: {R_SM};
            font-weight: 600;
            min-width: 95px;
        }}
        QPushButton:hover {{ background-color: {RED_HOVER}; }}
        QPushButton:disabled {{
            background-color: {DISABLED_BG};
            color: {DISABLED_TEXT};
        }}
    """


# ═══════════════════════════════════════════════════════════
#  PROGRESS & LOG
# ═══════════════════════════════════════════════════════════

def progress_bar() -> str:
    return f"""
        QProgressBar {{
            border: 1px solid {BORDER};
            border-radius: 4px;
            text-align: center;
            height: 20px;
            background-color: {BG_SURFACE};
            color: {TEXT_PRIMARY};
            font-weight: 600;
            font-family: {FONT_MONO};
            font-size: {fs(11)};
        }}
        QProgressBar::chunk {{
            background-color: {ORANGE};
            border-radius: 3px;
        }}
    """

def log_area() -> str:
    return f"""
        QTextEdit {{
            background-color: {BG_DEEPEST};
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            padding: 10px;
            font-family: {FONT_MONO};
            font-size: {fs(11)};
            line-height: 1.5;
        }}
    """


# ═══════════════════════════════════════════════════════════
#  DROP ZONES
# ═══════════════════════════════════════════════════════════

def drop_zone_default() -> str:
    return f"""
        QLabel {{
            border: 2px dashed {BORDER_LIGHT};
            border-radius: {R};
            background-color: {BG_PANEL};
            color: {TEXT_MUTED};
            font-size: {fs(13)};
            padding: 24px;
        }}
        QLabel:hover {{
            background-color: {BG_HOVER};
            border-color: {ORANGE_DIM};
            color: {TEXT_SECONDARY};
        }}
    """

def drop_zone_active() -> str:
    return f"""
        QLabel {{
            border: 2px dashed {ORANGE};
            border-radius: {R};
            background-color: {ORANGE_SUBTLE};
            color: {ORANGE};
            font-size: {fs(13)};
            padding: 24px;
        }}
    """

def drop_zone_frame_default() -> str:
    return f"""
        QFrame {{
            background: {BG_PANEL};
            border: 2px dashed {BORDER_ACCENT};
            border-radius: {R};
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
            border-radius: {R};
        }}
    """

def drop_zone_frame_success() -> str:
    return f"""
        QFrame {{
            background: {ORANGE_SUBTLE};
            border: 2px solid {ORANGE};
            border-radius: {R};
        }}
    """


# ═══════════════════════════════════════════════════════════
#  CARD FRAMES
# ═══════════════════════════════════════════════════════════

def card_frame() -> str:
    return f"""
        QFrame {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER_LIGHT};
            border-radius: {R};
            padding: 14px;
        }}
    """

def card_frame_compact() -> str:
    return f"""
        QFrame {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: {R};
            padding: 10px;
        }}
    """


# ═══════════════════════════════════════════════════════════
#  TAB WIDGET
# ═══════════════════════════════════════════════════════════

def tab_widget() -> str:
    return f"""
        QTabWidget::pane {{
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            background-color: {BG_PANEL};
            margin-top: -1px;
        }}
        QTabBar::tab {{
            background-color: transparent;
            color: {TEXT_MUTED};
            padding: 8px 16px;
            margin-right: 2px;
            border-bottom: 2px solid transparent;
            font-weight: 500;
            font-size: {fs(12)};
        }}
        QTabBar::tab:selected {{
            color: {ORANGE};
            border-bottom: 2px solid {ORANGE};
        }}
        QTabBar::tab:hover {{
            color: {TEXT_PRIMARY};
        }}
    """


# ═══════════════════════════════════════════════════════════
#  SYSTEM MONITOR (sidebar)
# ═══════════════════════════════════════════════════════════

def monitor_frame() -> str:
    return f"""
        QFrame {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: {R_SM};
            padding: 0;
        }}
    """

def monitor_bar(color: str) -> str:
    return f"""
        QProgressBar {{
            background-color: {BG_SURFACE};
            border: none;
            border-radius: 2px;
            height: 4px;
        }}
        QProgressBar::chunk {{
            background-color: {color};
            border-radius: 2px;
        }}
    """


# ═══════════════════════════════════════════════════════════
#  INLINE HELPERS
# ═══════════════════════════════════════════════════════════

def info_icon() -> str:
    return f"color: {TEXT_MUTED}; font-size: {fs(13)};"

def label_default() -> str:
    return f"color: {TEXT_PRIMARY}; font-size: {fs(12)};"

def label_accent() -> str:
    return f"color: {ORANGE}; font-weight: 600;"

def label_muted() -> str:
    return f"color: {TEXT_MUTED}; font-size: {fs(11)}; margin-bottom: 12px;"

def label_title() -> str:
    return f"color: {TEXT_PRIMARY}; font-size: {fs(18)}; font-weight: 700; letter-spacing: -0.02em;"

def label_section() -> str:
    return f"color: {ORANGE}; border: none; font-weight: 600; font-size: {fs(12)};"

def label_value() -> str:
    return f"font-weight: 600; color: {TEXT_PRIMARY}; min-width: 30px; font-family: {FONT_MONO}; font-size: {fs(12)};"

def label_transparent() -> str:
    return f"color: {TEXT_MUTED}; font-size: {fs(11)}; border: none; background: transparent;"

def label_success() -> str:
    return f"color: {GREEN}; font-weight: 600; font-size: {fs(12)}; border: none;"

def icon_transparent() -> str:
    return f"font-size: {fs(16)}; border: none; background: transparent;"

def label_frame() -> str:
    return f"color: {TEXT_PRIMARY}; border: none; font-size: {fs(12)};"

def info_icon_frame() -> str:
    return f"color: {TEXT_MUTED}; font-size: {fs(13)}; border: none; margin-right: 4px;"

def info_icon_frame_compact() -> str:
    return f"color: {TEXT_MUTED}; font-size: {fs(13)}; border: none;"

def checkbox_frame() -> str:
    return f"color: {TEXT_PRIMARY}; border: none; font-size: {fs(12)};"


def resolve_color(value):
    """Resolve a color supplier; explicit semantic/fixed colors stay explicit."""
    return value() if callable(value) else value


def enable_studio_design():
    """Use neutral professional surfaces. Preserve existing appearance preferences.

    Only a new installation defaults to dark/violet and English. This does not
    write user settings; the normal Settings page still owns persistence.
    """
    global _studio_design, _current_mode, _current_accent, _current_lang, _FONT_BASELINE
    _studio_design = True
    _FONT_BASELINE = 1.12
    _LIGHT_PALETTE.update({
        'BG_WINDOW': '#f3f4f8', 'BG_DEEPEST': '#e9ebf2', 'BG_DEEP': '#eff0f6',
        'BG_DARK': '#eeeff5', 'BG_PANEL': '#f8f9fc', 'BG_CARD': '#ffffff',
        'BG_SURFACE': '#f2f3f8', 'BG_HOVER': '#e9eaf3', 'BG_ELEVATED': '#e3e5ee',
        'TEXT_PRIMARY': '#23263c', 'TEXT_SECONDARY': '#4d5369', 'TEXT_MUTED': '#646b82',
        'BORDER': '#dfe3ee', 'BORDER_LIGHT': '#d4d9e7', 'SIDEBAR_BG': '#fafbfe',
        'NAV_LABEL': '#6a7087', 'GREEN': '#12845e', 'BLUE': '#2563a0',
    })
    _DARK_PALETTE.update({
        'BG_WINDOW': '#171920', 'BG_DEEPEST': '#101219', 'BG_DEEP': '#181b24',
        'BG_DARK': '#1c1f29', 'BG_PANEL': '#20232e', 'BG_CARD': '#232733',
        'BG_SURFACE': '#2a2e3b', 'BG_HOVER': '#323747', 'BG_ELEVATED': '#3a4052',
        'TEXT_PRIMARY': '#ebedf7', 'TEXT_SECONDARY': '#c0c6da', 'TEXT_MUTED': '#9ea8c2',
        'BORDER': '#343b4e', 'BORDER_LIGHT': '#444d64', 'SIDEBAR_BG': '#1d202a',
        'NAV_LABEL': '#a4aec7',
    })
    if not _PREFS_PATH.exists():
        _current_mode, _current_accent, _current_lang = 'dark', '#7253d8', 'en'
    globals().update(_LIGHT_PALETTE if _current_mode == 'light' else _DARK_PALETTE)
    _apply_accent(_current_accent)
