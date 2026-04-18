"""
Animation helpers for LoRA-Harvester UI.

Lightweight, GPU-friendly helpers built on QPropertyAnimation + QGraphicsOpacityEffect.
All durations are kept under ~300ms to preserve workflow speed.
Each helper returns the animation instance so callers can chain or cancel it.
"""

from __future__ import annotations

from PyQt5.QtCore import (
    QEasingCurve, QPropertyAnimation, Qt, QPoint, QRect, QObject,
    QParallelAnimationGroup, QSequentialAnimationGroup, QAbstractAnimation,
    pyqtSignal, QEvent, QVariantAnimation,
)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QGraphicsOpacityEffect, QWidget, QPushButton, QFrame, QVBoxLayout,
)


# ══════════════════════════════════════════════════════════════
#  Fade
# ══════════════════════════════════════════════════════════════

def fade_in(widget: QWidget, duration: int = 220,
            easing: QEasingCurve.Type = QEasingCurve.OutCubic) -> QPropertyAnimation:
    """Fade widget from transparent to fully opaque."""
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
    effect.setOpacity(0.0)
    widget.show()
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(easing)
    anim.start(QAbstractAnimation.DeleteWhenStopped)
    return anim


def fade_out(widget: QWidget, duration: int = 180,
             easing: QEasingCurve.Type = QEasingCurve.InCubic,
             hide_on_finish: bool = True) -> QPropertyAnimation:
    """Fade widget to transparent, optionally hiding when complete."""
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
    start = effect.opacity()
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(0.0)
    anim.setEasingCurve(easing)
    if hide_on_finish:
        anim.finished.connect(widget.hide)
    anim.start(QAbstractAnimation.DeleteWhenStopped)
    return anim


def crossfade(old_widget: QWidget, new_widget: QWidget,
              duration: int = 220) -> QParallelAnimationGroup:
    """Fade old out and new in simultaneously."""
    group = QParallelAnimationGroup(new_widget)

    old_eff = QGraphicsOpacityEffect(old_widget)
    old_widget.setGraphicsEffect(old_eff)
    old_eff.setOpacity(1.0)
    fade_o = QPropertyAnimation(old_eff, b"opacity")
    fade_o.setDuration(duration)
    fade_o.setStartValue(1.0)
    fade_o.setEndValue(0.0)
    fade_o.setEasingCurve(QEasingCurve.InCubic)

    new_eff = QGraphicsOpacityEffect(new_widget)
    new_widget.setGraphicsEffect(new_eff)
    new_eff.setOpacity(0.0)
    new_widget.show()
    fade_n = QPropertyAnimation(new_eff, b"opacity")
    fade_n.setDuration(duration)
    fade_n.setStartValue(0.0)
    fade_n.setEndValue(1.0)
    fade_n.setEasingCurve(QEasingCurve.OutCubic)

    group.addAnimation(fade_o)
    group.addAnimation(fade_n)
    group.finished.connect(lambda: old_widget.setGraphicsEffect(None))
    group.finished.connect(lambda: new_widget.setGraphicsEffect(None))
    group.start(QAbstractAnimation.DeleteWhenStopped)
    return group


# ══════════════════════════════════════════════════════════════
#  Slide
# ══════════════════════════════════════════════════════════════

def slide_in(widget: QWidget, direction: str = "right", duration: int = 280,
             distance: int = 40) -> QPropertyAnimation:
    """Slide widget in from a direction while fading in. Directions: left/right/up/down."""
    target = widget.pos()
    if direction == "right":
        start = target + QPoint(-distance, 0)
    elif direction == "left":
        start = target + QPoint(distance, 0)
    elif direction == "up":
        start = target + QPoint(0, distance)
    else:  # down
        start = target + QPoint(0, -distance)

    widget.move(start)
    widget.show()

    group = QParallelAnimationGroup(widget)

    pos_anim = QPropertyAnimation(widget, b"pos")
    pos_anim.setDuration(duration)
    pos_anim.setStartValue(start)
    pos_anim.setEndValue(target)
    pos_anim.setEasingCurve(QEasingCurve.OutCubic)
    group.addAnimation(pos_anim)

    # pair with fade for a modern feel
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    effect.setOpacity(0.0)
    op_anim = QPropertyAnimation(effect, b"opacity")
    op_anim.setDuration(duration)
    op_anim.setStartValue(0.0)
    op_anim.setEndValue(1.0)
    op_anim.setEasingCurve(QEasingCurve.OutCubic)
    group.addAnimation(op_anim)

    group.finished.connect(lambda: widget.setGraphicsEffect(None))
    group.start(QAbstractAnimation.DeleteWhenStopped)
    return group


# ══════════════════════════════════════════════════════════════
#  Smooth collapsible expansion (maxHeight)
# ══════════════════════════════════════════════════════════════

def smooth_expand(widget: QWidget, expand: bool, duration: int = 240) -> QPropertyAnimation:
    """Animate maximumHeight between 0 and content height."""
    widget.setMaximumHeight(widget.maximumHeight() if widget.maximumHeight() < 16777215 else widget.sizeHint().height())
    content_h = widget.sizeHint().height()
    start_h = widget.maximumHeight() if widget.isVisible() else 0
    end_h = content_h if expand else 0

    if expand:
        widget.setMaximumHeight(0)
        widget.show()

    anim = QPropertyAnimation(widget, b"maximumHeight", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_h)
    anim.setEndValue(end_h)
    anim.setEasingCurve(QEasingCurve.InOutCubic)

    def _done():
        if expand:
            widget.setMaximumHeight(16777215)  # release the cap
        else:
            widget.hide()

    anim.finished.connect(_done)
    anim.start(QAbstractAnimation.DeleteWhenStopped)
    return anim


# ══════════════════════════════════════════════════════════════
#  Progress bar smoothing
# ══════════════════════════════════════════════════════════════

def progress_smooth(bar, target: int, duration: int = 260) -> QPropertyAnimation:
    """Interpolate a QProgressBar to `target` instead of jumping."""
    start = bar.value()
    if start == target:
        return None
    anim = QPropertyAnimation(bar, b"value", bar)
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(target)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    anim.start(QAbstractAnimation.DeleteWhenStopped)
    return anim


# ══════════════════════════════════════════════════════════════
#  Pulsing frame (drop zones)
# ══════════════════════════════════════════════════════════════

class PulseEffect(QObject):
    """Attach to a widget's graphicsEffect; pulses opacity 0.75 ↔ 1.0."""

    def __init__(self, widget: QWidget, min_opacity: float = 0.72,
                 duration: int = 900, parent=None):
        super().__init__(parent or widget)
        self._widget = widget
        self._effect = QGraphicsOpacityEffect(widget)
        self._effect.setOpacity(1.0)
        widget.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(duration)
        self._anim.setStartValue(1.0)
        self._anim.setKeyValueAt(0.5, min_opacity)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.InOutSine)

    def start(self):
        if self._anim.state() != QAbstractAnimation.Running:
            self._anim.start()

    def stop(self):
        self._anim.stop()
        self._effect.setOpacity(1.0)
        self._widget.setGraphicsEffect(None)


# ══════════════════════════════════════════════════════════════
#  Hover lift for buttons
# ══════════════════════════════════════════════════════════════

class HoverLift(QObject):
    """Event filter that nudges a button's Y by a few pixels on hover."""

    def __init__(self, button: QWidget, lift_px: int = 2, parent=None):
        super().__init__(parent or button)
        self._btn = button
        self._lift = lift_px
        self._base_margin = button.contentsMargins()
        button.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._btn:
            if event.type() == QEvent.Enter:
                m = self._base_margin
                self._btn.setContentsMargins(m.left(), max(0, m.top() - self._lift),
                                             m.right(), m.bottom() + self._lift)
            elif event.type() == QEvent.Leave:
                m = self._base_margin
                self._btn.setContentsMargins(m.left(), m.top(), m.right(), m.bottom())
        return False


# ══════════════════════════════════════════════════════════════
#  Animated stacked widget (page switch crossfade + slide)
# ══════════════════════════════════════════════════════════════

def animate_page_switch(stack, old_index: int, new_index: int,
                        duration: int = 260) -> None:
    """Crossfade + 20px slide when QStackedWidget changes page."""
    if old_index == new_index or old_index < 0:
        stack.setCurrentIndex(new_index)
        return

    new_w = stack.widget(new_index)
    if new_w is None:
        stack.setCurrentIndex(new_index)
        return

    # Switch index immediately; animate only the new page
    stack.setCurrentIndex(new_index)

    # Slide from +18px right, fade 0→1
    target = new_w.pos()
    new_w.move(target + QPoint(18, 0))

    effect = QGraphicsOpacityEffect(new_w)
    new_w.setGraphicsEffect(effect)
    effect.setOpacity(0.0)

    group = QParallelAnimationGroup(new_w)

    pos_anim = QPropertyAnimation(new_w, b"pos")
    pos_anim.setDuration(duration)
    pos_anim.setStartValue(target + QPoint(18, 0))
    pos_anim.setEndValue(target)
    pos_anim.setEasingCurve(QEasingCurve.OutCubic)
    group.addAnimation(pos_anim)

    op_anim = QPropertyAnimation(effect, b"opacity")
    op_anim.setDuration(duration)
    op_anim.setStartValue(0.0)
    op_anim.setEndValue(1.0)
    op_anim.setEasingCurve(QEasingCurve.OutCubic)
    group.addAnimation(op_anim)

    group.finished.connect(lambda: new_w.setGraphicsEffect(None))
    group.start(QAbstractAnimation.DeleteWhenStopped)


# ══════════════════════════════════════════════════════════════
#  Underline indicator for navigation (animated active tab marker)
# ══════════════════════════════════════════════════════════════

class NavIndicator(QWidget):
    """A slim horizontal bar that slides under the active nav button."""

    def __init__(self, parent: QWidget, color: str = "#D97757",
                 height: int = 3):
        super().__init__(parent)
        self._color = color
        self._height = height
        self.setFixedHeight(height)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet(f"background-color: {color}; border-radius: {height // 2}px;")
        self.hide()
        self._anim = QPropertyAnimation(self, b"geometry", self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_color(self, color: str):
        self._color = color
        self.setStyleSheet(f"background-color: {color}; border-radius: {self._height // 2}px;")

    def move_under(self, button: QWidget):
        """Animate the bar to sit under `button`."""
        if button is None:
            return
        # Translate button geometry to our parent's coordinates
        btn_geo = button.geometry()
        parent = self.parentWidget()
        if parent is None:
            return
        # Position just below the button
        top_left = button.mapTo(parent, QPoint(0, button.height()))
        target = QRect(top_left.x(), top_left.y() + 2,
                       btn_geo.width(), self._height)
        if not self.isVisible():
            self.setGeometry(target)
            self.show()
            return
        self._anim.stop()
        self._anim.setStartValue(self.geometry())
        self._anim.setEndValue(target)
        self._anim.start()
