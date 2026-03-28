"""
Unified Dark Theme for LoRA-Harvester v2.0
Color Palette: Black, Gray, Orange
All UI components reference this module for consistent styling.
"""

# ═══════════════════════════════════════════════════════════
#  COLOR PALETTE
# ═══════════════════════════════════════════════════════════

# Backgrounds (darkest → lightest)
BG_WINDOW    = "#1e1e1e"   # Main window
BG_DEEPEST   = "#121212"   # Deepest inputs, log
BG_DEEP      = "#161616"   # Deep panels
BG_DARK      = "#1a1a1a"   # Dark panels
BG_PANEL     = "#1e1e1e"   # Card / panel backgrounds
BG_SURFACE   = "#252525"   # Elevated surfaces
BG_HOVER     = "#2d2d2d"   # Hover states
BG_ELEVATED  = "#353535"   # Higher elevation

# Orange accent family
ORANGE         = "#e8832a"  # Primary accent
ORANGE_LIGHT   = "#f5a623"  # Highlights, active text
ORANGE_DARK    = "#c96f1e"  # Pressed states
ORANGE_GLOW    = "#ff9f43"  # Glow / bright hover
ORANGE_DIM     = "#8b5e2f"  # Dimmed / inactive accent
ORANGE_SUBTLE  = "#3d2a14"  # Very subtle orange tint bg

# Text
TEXT_PRIMARY   = "#e0e0e0"  # Main text
TEXT_SECONDARY = "#a0a0a0"  # Secondary text
TEXT_MUTED     = "#666666"  # Muted / hint text
TEXT_ACCENT    = "#f5a623"  # Orange accent text

# Borders
BORDER         = "#333333"  # Default border
BORDER_LIGHT   = "#444444"  # Lighter border
BORDER_ACCENT  = "#e8832a"  # Orange accent border

# Semantic
RED            = "#c0392b"  # Stop / error only
RED_HOVER      = "#a93226"  # Red hover
DISABLED_BG    = "#2a2a2a"  # Disabled background
DISABLED_TEXT  = "#555555"  # Disabled text


# ═══════════════════════════════════════════════════════════
#  STYLE GENERATORS
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
