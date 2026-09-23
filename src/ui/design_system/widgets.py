"""Small native Qt components; all colors are live theme bindings."""
from __future__ import annotations
from PyQt5.QtCore import Qt, QByteArray, QSize, QRectF
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (QFrame, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QToolButton, QSizePolicy, QTabWidget)
from src.ui import theme

PATHS = {
    'overview': '<path d="M3 11 12 3l9 8v10h-6v-7H9v7H3z"/>',
    'video': '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M8 5v14M16 5v14M3 9h5M3 15h5M16 9h5M16 15h5"/>',
    'library': '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    'edit': '<path d="m4 16 12-12 4 4L8 20l-5 1zM14 6l4 4"/>',
    'caption': '<path d="M6 3h10l4 4v14H4V3zM8 11h8M8 15h8M8 18h5"/>',
    'clothing': '<path d="m8 3-6 4 3 5 3-2v11h8V10l3 2 3-5-6-4c-1 4-7 4-8 0z"/>',
    'balance': '<path d="M5 21V11h4v10M11 21V3h4v18M17 21v-6h4v6"/>',
    'compare': '<rect x="2" y="5" width="8" height="14" rx="1"/><rect x="14" y="5" width="8" height="14" rx="1"/>',
    'export': '<path d="M12 16V2m-5 5 5-5 5 5M4 13v8h16v-8"/>',
    'settings': '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="16" cy="17" r="3"/>',
    'tools': '<path d="m3 21 9-9a6 6 0 0 1 7-8l-4 4 2 2 4-4a6 6 0 0 1-8 7l-9 9z"/>',
    'folder': '<path d="M3 5h7l2 3h9v12H3z"/>',
    'spark': '<path d="m12 2 2 7 7 3-7 2-2 8-2-8-8-2 8-3z"/>',
}


def line_icon(name, color=None, size=20):
    color = color or theme.TEXT_SECONDARY
    body = PATHS.get(name, PATHS['folder'])
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    pix = QPixmap(size * 2, size * 2)
    pix.fill(Qt.transparent)
    pix.setDevicePixelRatio(2)
    painter = QPainter(pix)
    QSvgRenderer(QByteArray(svg.encode())).render(painter, QRectF(0, 0, size, size))
    painter.end()
    return QIcon(pix)


def label(text='', *, role='body', parent=None):
    widget = QLabel(text, parent)
    widget.setTextFormat(Qt.PlainText)
    widget.setWordWrap(True)
    def style():
        color = theme.TEXT_MUTED if role == 'muted' else theme.TEXT_PRIMARY
        size = 20 if role == 'title' else 12
        weight = '600' if role in ('title', 'heading') else '400'
        return f'background:transparent;border:none;color:{color};font-size:{theme.fs(size)};font-weight:{weight};'
    theme.bind_style(widget, style)
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.setCursor(Qt.PointingHandCursor)
    theme.bind_style(widget, theme.btn_primary if primary else theme.btn_secondary)
    widget.clicked.connect(lambda checked=False: callback())
    return widget


class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('studioCard')
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 16)
        self.body.setSpacing(10)
        theme.bind_style(self, lambda: f'QFrame#studioCard {{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:10px;}}')


class NavButton(QPushButton):
    def __init__(self, key, text, parent=None):
        super().__init__(text, parent)
        self.key = key
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setIconSize(QSize(18, 18))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggled.connect(lambda _: self.refresh_icon())
        self.refresh_icon()
        theme.bind_style(self, lambda: f'''
            QPushButton {{ background:transparent; border:none; border-radius:7px;
                padding:9px 12px; text-align:left; color:{theme.TEXT_SECONDARY}; font-size:{theme.fs(12)}; }}
            QPushButton:hover {{ background:{theme.BG_HOVER}; }}
            QPushButton:checked {{ background:{theme.ORANGE_SUBTLE}; color:{theme.get_accent()}; font-weight:600; }}
            QPushButton:disabled {{ color:{theme.DISABLED_TEXT}; }}''')

    def refresh_icon(self):
        self.setIcon(line_icon(self.key, theme.get_accent() if self.isChecked() else theme.TEXT_SECONDARY))


class Disclosure(QFrame):
    """Progressive disclosure without disabling or changing underlying values."""
    def __init__(self, title, content, expanded=False, parent=None):
        super().__init__(parent)
        self.setObjectName('studioDisclosure')
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(12, 8, 12, 10)
        self.body.setSpacing(8)
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content = content
        self.body.addWidget(self.header)
        self.body.addWidget(content)
        content.setVisible(expanded)
        self.header.toggled.connect(self._toggle)
        theme.bind_style(self, lambda: f'QFrame#studioDisclosure{{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:8px;}}')
        theme.bind_style(self.header, lambda: f'QToolButton{{border:none;background:transparent;text-align:left;color:{theme.TEXT_PRIMARY};font-weight:600;padding:8px 0;}}')

    def _toggle(self, checked):
        self.header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)


class ActiveTabWidget(QTabWidget):
    """Do not size the current editor from the largest hidden tool form."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _: self.updateGeometry())

    def _content_size(self, minimum=False):
        current = self.currentWidget()
        if current is None:
            return QSize(320, 240)
        size = current.minimumSizeHint() if minimum else current.sizeHint()
        bar = self.tabBar().sizeHint() if not self.tabBar().isHidden() else QSize(0, 0)
        return QSize(max(0, size.width(), bar.width()) + 4, max(0, size.height()) + bar.height() + 4)

    def sizeHint(self):
        return self._content_size()

    def minimumSizeHint(self):
        return self._content_size(True)
