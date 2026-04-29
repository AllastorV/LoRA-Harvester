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
    pyqtSignal, QEvent, QVariantAnimation, QTimer, QSize,
)
from PyQt5.QtGui import (
    QColor, QPainter, QBrush, QPen, QLinearGradient, QFontMetrics,
)
from PyQt5.QtWidgets import (
    QGraphicsOpacityEffect, QGraphicsDropShadowEffect,
    QWidget, QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
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


# ══════════════════════════════════════════════════════════════
#  Press flash (accent swatch / button tap feedback)
# ══════════════════════════════════════════════════════════════

def press_flash(widget: QWidget, duration: int = 180) -> QSequentialAnimationGroup:
    """Opacity dip 1.0 → 0.6 → 1.0 for instant press feedback."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    effect.setOpacity(1.0)
    group = QSequentialAnimationGroup(widget)
    dip = QPropertyAnimation(effect, b"opacity")
    dip.setDuration(duration // 2)
    dip.setStartValue(1.0)
    dip.setEndValue(0.6)
    dip.setEasingCurve(QEasingCurve.OutCubic)
    recover = QPropertyAnimation(effect, b"opacity")
    recover.setDuration(duration // 2)
    recover.setStartValue(0.6)
    recover.setEndValue(1.0)
    recover.setEasingCurve(QEasingCurve.InCubic)
    group.addAnimation(dip)
    group.addAnimation(recover)
    group.finished.connect(lambda: widget.setGraphicsEffect(None))
    group.start(QAbstractAnimation.DeleteWhenStopped)
    return group


# ══════════════════════════════════════════════════════════════
#  Status dot — heartbeat colour-morphing indicator
# ══════════════════════════════════════════════════════════════

class StatusDot(QWidget):
    """10px dot with colour states + heartbeat pulse for 'processing'."""

    COLORS = {
        'idle':       '#555555',
        'processing': '#D97757',
        'paused':     '#F0C040',
        'done':       '#4CAF50',
        'error':      '#F44336',
    }

    def __init__(self, parent=None, size: int = 10):
        super().__init__(parent)
        self._sz = size
        self.setFixedSize(size, size)
        self._state = 'idle'
        self._color = QColor(self.COLORS['idle'])
        self._pulse_anim = None
        self._color_anim = None
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(0, 0, self._sz, self._sz)

    def set_state(self, state: str):
        self._state = state if state in self.COLORS else 'idle'
        target = QColor(self.COLORS.get(state, self.COLORS['idle']))

        # Smooth colour morph
        if self._color_anim is not None:
            try:
                self._color_anim.stop()
            except RuntimeError:
                pass
        self._color_anim = QVariantAnimation(self)
        self._color_anim.setDuration(300)
        self._color_anim.setStartValue(self._color)
        self._color_anim.setEndValue(target)
        self._color_anim.valueChanged.connect(self._set_color)
        self._color_anim.start()  # no DeleteWhenStopped — parent=self keeps C++ object alive

        # Stop old pulse
        if self._pulse_anim:
            self._pulse_anim.stop()
            self._effect.setOpacity(1.0)
            self._pulse_anim = None

        # Heartbeat double-pulse for processing
        if state == 'processing':
            seq = QSequentialAnimationGroup(self)
            for end_op in (0.3, 1.0, 0.4, 1.0):
                a = QPropertyAnimation(self._effect, b"opacity")
                a.setDuration(100)
                a.setEndValue(end_op)
                seq.addAnimation(a)
            # Long rest
            rest = QPropertyAnimation(self._effect, b"opacity")
            rest.setDuration(700)
            rest.setStartValue(1.0)
            rest.setEndValue(1.0)
            seq.addAnimation(rest)
            seq.setLoopCount(-1)
            seq.start()
            self._pulse_anim = seq

    def _set_color(self, c):
        self._color = c
        self.update()


# ══════════════════════════════════════════════════════════════
#  Count-up label animation
# ══════════════════════════════════════════════════════════════

def count_up(label: QWidget, from_val: int, to_val: int,
             duration: int = 400) -> QVariantAnimation:
    """Animate a QLabel's text from *from_val* to *to_val*."""
    if from_val == to_val:
        label.setText(str(to_val))
        return None
    anim = QVariantAnimation(label)
    anim.setDuration(duration)
    anim.setStartValue(float(from_val))
    anim.setEndValue(float(to_val))
    anim.setEasingCurve(QEasingCurve.OutCubic)
    anim.valueChanged.connect(lambda v: label.setText(str(int(v))))
    anim.start(QAbstractAnimation.DeleteWhenStopped)
    return anim


# ══════════════════════════════════════════════════════════════
#  Shake widget (validation error)
# ══════════════════════════════════════════════════════════════

def shake_widget(widget: QWidget, distance: int = 8,
                 duration: int = 350) -> QSequentialAnimationGroup:
    """Shake a widget horizontally to signal an error."""
    origin = widget.pos()
    group = QSequentialAnimationGroup(widget)
    offsets = [distance, -distance, distance // 2, -distance // 2, 0]
    step_dur = duration // len(offsets)
    for offset in offsets:
        a = QPropertyAnimation(widget, b"pos")
        a.setDuration(step_dur)
        a.setEndValue(origin + QPoint(offset, 0))
        a.setEasingCurve(QEasingCurve.InOutQuad)
        group.addAnimation(a)
    group.start(QAbstractAnimation.DeleteWhenStopped)
    return group


# ══════════════════════════════════════════════════════════════
#  Ripple effect (Material Design click ripple)
# ══════════════════════════════════════════════════════════════

class _RippleOverlay(QWidget):
    """Transparent child that paints expanding ripple circles."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self._ripples: list = []

    def trigger(self, pos: QPoint):
        max_r = max(
            (pos.x() ** 2 + pos.y() ** 2) ** 0.5,
            ((self.width() - pos.x()) ** 2 + pos.y() ** 2) ** 0.5,
            (pos.x() ** 2 + (self.height() - pos.y()) ** 2) ** 0.5,
            ((self.width() - pos.x()) ** 2 + (self.height() - pos.y()) ** 2) ** 0.5,
        )
        ripple = {'cx': pos.x(), 'cy': pos.y(), 'r': 0.0, 'op': 0.22}
        self._ripples.append(ripple)

        r_anim = QVariantAnimation(self)
        r_anim.setDuration(420)
        r_anim.setStartValue(0.0)
        r_anim.setEndValue(float(max_r))
        r_anim.setEasingCurve(QEasingCurve.OutCubic)
        r_anim.valueChanged.connect(lambda v, rip=ripple: self._set(rip, 'r', v))

        o_anim = QVariantAnimation(self)
        o_anim.setDuration(380)
        o_anim.setStartValue(0.22)
        o_anim.setEndValue(0.0)
        o_anim.setEasingCurve(QEasingCurve.InCubic)
        o_anim.valueChanged.connect(lambda v, rip=ripple: self._set(rip, 'op', v))

        grp = QParallelAnimationGroup(self)
        grp.addAnimation(r_anim)
        grp.addAnimation(o_anim)
        grp.finished.connect(lambda rip=ripple: self._remove(rip))
        grp.start(QAbstractAnimation.DeleteWhenStopped)

    def _set(self, rip, key, val):
        rip[key] = val
        self.update()

    def _remove(self, rip):
        if rip in self._ripples:
            self._ripples.remove(rip)
        self.update()

    def paintEvent(self, event):
        if not self._ripples:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        for rip in self._ripples:
            c = QColor(255, 255, 255, int(rip['op'] * 255))
            p.setBrush(QBrush(c))
            r = int(rip['r'])
            p.drawEllipse(rip['cx'] - r, rip['cy'] - r, r * 2, r * 2)


class RippleButton(QObject):
    """Event filter — adds a Material-style click ripple to any button."""

    def __init__(self, button: QWidget, parent=None):
        super().__init__(parent or button)
        self._btn = button
        self._overlay = _RippleOverlay(button)
        self._overlay.setGeometry(button.rect())
        self._overlay.show()
        self._overlay.raise_()
        button.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._btn:
            if event.type() == QEvent.Resize:
                self._overlay.setGeometry(self._btn.rect())
            elif event.type() == QEvent.MouseButtonPress:
                self._overlay.trigger(event.pos())
        return False


# ══════════════════════════════════════════════════════════════
#  Scale pop (elastic bounce on action)
# ══════════════════════════════════════════════════════════════

def scale_pop(widget: QWidget, factor: float = 1.06,
              duration: int = 280) -> QSequentialAnimationGroup:
    """Briefly inflate a widget then snap back with OutBack easing."""
    geo = widget.geometry()
    cx, cy = geo.center().x(), geo.center().y()
    w, h = geo.width(), geo.height()
    ew, eh = int(w * factor), int(h * factor)
    expanded = QRect(cx - ew // 2, cy - eh // 2, ew, eh)

    grp = QSequentialAnimationGroup(widget)

    grow = QPropertyAnimation(widget, b"geometry")
    grow.setDuration(duration * 2 // 5)
    grow.setStartValue(geo)
    grow.setEndValue(expanded)
    grow.setEasingCurve(QEasingCurve.OutCubic)

    shrink = QPropertyAnimation(widget, b"geometry")
    shrink.setDuration(duration * 3 // 5)
    shrink.setStartValue(expanded)
    shrink.setEndValue(geo)
    shrink.setEasingCurve(QEasingCurve.OutBack)

    grp.addAnimation(grow)
    grp.addAnimation(shrink)
    grp.start(QAbstractAnimation.DeleteWhenStopped)
    return grp


# ══════════════════════════════════════════════════════════════
#  Toast notification (slide-in from bottom-right)
# ══════════════════════════════════════════════════════════════

class ToastNotification(QFrame):
    """Auto-dismiss toast that slides in from the bottom-right corner."""

    _active_toasts: list = []          # class-level stack

    def __init__(self, message: str, icon: str = "ℹ️",
                 duration_ms: int = 3000, accent: str = "#D97757",
                 parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("ToastNotification")
        self.setFixedHeight(42)
        self.setMinimumWidth(220)
        self.setMaximumWidth(420)
        self.setStyleSheet(
            f"#ToastNotification {{"
            f"  background-color: #2a2a2e; border: 1px solid #3a3a3e;"
            f"  border-left: 3px solid {accent}; border-radius: 8px;"
            f"}}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 14, 4)
        lay.setSpacing(8)
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 15px; background: transparent; border: none;")
        msg_lbl = QLabel(message)
        msg_lbl.setStyleSheet(
            "color: #e0e0e0; font-size: 12px; font-weight: 500;"
            " background: transparent; border: none;"
        )
        lay.addWidget(icon_lbl)
        lay.addWidget(msg_lbl, stretch=1)
        self.adjustSize()

        # Track this toast
        ToastNotification._active_toasts.append(self)
        self._dismissed = False

        # Position & animate
        QTimer.singleShot(0, self._enter)
        QTimer.singleShot(duration_ms, self._exit)

    def _slot_y(self, index: int) -> int:
        p = self.parentWidget()
        if not p:
            return 0
        return p.height() - (index + 1) * (self.height() + 8) - 12

    def _enter(self):
        p = self.parentWidget()
        if not p:
            return
        idx = ToastNotification._active_toasts.index(self)
        x = p.width() - self.width() - 16
        self.move(x, p.height() + 10)
        self.show()
        self.raise_()
        target_y = self._slot_y(idx)
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(300)
        anim.setEndValue(QPoint(x, target_y))
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start(QAbstractAnimation.DeleteWhenStopped)

    def _exit(self):
        if self._dismissed:
            return
        self._dismissed = True
        p = self.parentWidget()
        target_y = (p.height() + 10) if p else self.y() + 60
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(240)
        anim.setEndValue(QPoint(self.x(), target_y))
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(self._cleanup)
        anim.start(QAbstractAnimation.DeleteWhenStopped)

    def _cleanup(self):
        if self in ToastNotification._active_toasts:
            ToastNotification._active_toasts.remove(self)
        self.deleteLater()


# ══════════════════════════════════════════════════════════════
#  Loading spinner (rotating arc)
# ══════════════════════════════════════════════════════════════

class LoadingSpinner(QWidget):
    """Circular indeterminate spinner drawn with QPainter arcs."""

    def __init__(self, parent=None, size: int = 28,
                 color: str = "#D97757", line_width: int = 3):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor(color)
        self._lw = line_width
        self._angle = 0
        self._span = 80

        self._rot = QVariantAnimation(self)
        self._rot.setDuration(1100)
        self._rot.setStartValue(0)
        self._rot.setEndValue(360)
        self._rot.setLoopCount(-1)
        self._rot.setEasingCurve(QEasingCurve.Linear)
        self._rot.valueChanged.connect(self._on_angle)

        # Arc span oscillation 40 deg <-> 270 deg
        self._sw = QVariantAnimation(self)
        self._sw.setDuration(1400)
        self._sw.setStartValue(40)
        self._sw.setEndValue(270)
        self._sw.setLoopCount(-1)
        self._sw.setEasingCurve(QEasingCurve.InOutQuad)
        self._sw.valueChanged.connect(self._on_span)

        self.hide()

    def _on_angle(self, v):
        self._angle = v
        self.update()

    def _on_span(self, v):
        self._span = v
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self._color, self._lw, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        m = self._lw + 1
        r = self.rect().adjusted(m, m, -m, -m)
        p.drawArc(r, int(self._angle * 16), int(self._span * 16))

    def start(self):
        self.show()
        self._rot.start()
        self._sw.start()

    def stop(self):
        self._rot.stop()
        self._sw.stop()
        self.hide()

    def set_color(self, color: str):
        self._color = QColor(color)


# ══════════════════════════════════════════════════════════════
#  Skeleton shimmer (loading placeholder)
# ══════════════════════════════════════════════════════════════

class SkeletonShimmer(QWidget):
    """Overlay that paints a moving gradient shimmer over its parent."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._offset = -1.0

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1500)
        self._anim.setStartValue(-1.0)
        self._anim.setEndValue(2.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.valueChanged.connect(self._on_offset)
        self.hide()

    def _on_offset(self, v):
        self._offset = v
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(35, 35, 39))
        sx = int(w * self._offset)
        sw = int(w * 0.35)
        grad = QLinearGradient(sx, 0, sx + sw, 0)
        grad.setColorAt(0.0, QColor(35, 35, 39, 0))
        grad.setColorAt(0.5, QColor(58, 58, 64, 200))
        grad.setColorAt(1.0, QColor(35, 35, 39, 0))
        p.fillRect(0, 0, w, h, grad)

    def start(self):
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self._anim.start()

    def stop(self):
        self._anim.stop()
        self.hide()


# ══════════════════════════════════════════════════════════════
#  Stagger fade-in (sequential reveal of child widgets)
# ══════════════════════════════════════════════════════════════

def stagger_fade_in(widgets: list, delay_per: int = 50,
                    duration: int = 220) -> None:
    """Fade-in a list of widgets with staggered delays."""
    for i, widget in enumerate(widgets):
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
        eff.setOpacity(0.0)
        widget.show()
        QTimer.singleShot(
            i * delay_per,
            lambda w=widget, e=eff, d=duration: _stagger_do(w, e, d),
        )


def _stagger_do(widget, effect, duration):
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start(QAbstractAnimation.DeleteWhenStopped)


# ══════════════════════════════════════════════════════════════
#  Glitch effect (error visual)
# ══════════════════════════════════════════════════════════════

def glitch_effect(widget: QWidget, duration: int = 320) -> QSequentialAnimationGroup:
    """Rapid position + opacity jitter for an error state."""
    origin = widget.pos()
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    eff.setOpacity(1.0)
    group = QSequentialAnimationGroup(widget)

    # Fixed jitter pattern (deterministic)
    jitter = [(4, -1, 0.5), (-3, 2, 0.8), (5, 0, 0.4),
              (-4, -1, 0.9), (2, 1, 0.6), (0, 0, 1.0)]
    step = duration // len(jitter)
    for dx, dy, op in jitter:
        para = QParallelAnimationGroup()
        pa = QPropertyAnimation(widget, b"pos")
        pa.setDuration(step)
        pa.setEndValue(origin + QPoint(dx, dy))
        oa = QPropertyAnimation(eff, b"opacity")
        oa.setDuration(step)
        oa.setEndValue(op)
        para.addAnimation(pa)
        para.addAnimation(oa)
        group.addAnimation(para)

    group.finished.connect(lambda: widget.setGraphicsEffect(None))
    group.start(QAbstractAnimation.DeleteWhenStopped)
    return group


# ══════════════════════════════════════════════════════════════
#  Badge bounce (elastic pop for nav badges)
# ══════════════════════════════════════════════════════════════

def badge_bounce(widget: QWidget, duration: int = 500) -> QPropertyAnimation:
    """Elastic bounce a widget from slightly above its position."""
    origin = widget.pos()
    anim = QPropertyAnimation(widget, b"pos", widget)
    anim.setDuration(duration)
    anim.setStartValue(origin + QPoint(0, -8))
    anim.setEndValue(origin)
    anim.setEasingCurve(QEasingCurve.OutElastic)
    anim.start(QAbstractAnimation.DeleteWhenStopped)
    return anim


# ══════════════════════════════════════════════════════════════
#  Progress glow (pulsing drop shadow on progress bar)
# ══════════════════════════════════════════════════════════════

class ProgressGlow(QObject):
    """Pulsing glow effect around a widget (typically a progress bar)."""

    def __init__(self, widget: QWidget, color: str = "#D97757", parent=None):
        super().__init__(parent or widget)
        self._widget = widget
        self._effect = QGraphicsDropShadowEffect(widget)
        self._effect.setColor(QColor(color))
        self._effect.setOffset(0, 0)
        self._effect.setBlurRadius(0)
        self._anim = QPropertyAnimation(self._effect, b"blurRadius", self)
        self._anim.setDuration(1200)
        self._anim.setStartValue(0.0)
        self._anim.setKeyValueAt(0.5, 18.0)
        self._anim.setEndValue(0.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.InOutSine)

    def start(self):
        self._widget.setGraphicsEffect(self._effect)
        self._anim.start()

    def stop(self):
        self._anim.stop()
        self._effect.setBlurRadius(0)
        self._widget.setGraphicsEffect(None)

    def set_color(self, color: str):
        self._effect.setColor(QColor(color))


# ══════════════════════════════════════════════════════════════
#  Shimmer label (brand text with travelling highlight)
# ══════════════════════════════════════════════════════════════

class ShimmerLabel(QLabel):
    """QLabel whose text colour has a travelling gradient highlight."""

    def __init__(self, text: str = "", parent=None,
                 base_color: str = "#e0e0e0",
                 highlight_color: str = "#ffffff"):
        super().__init__(text, parent)
        self._base = QColor(base_color)
        self._hi = QColor(highlight_color)
        self._pos = -0.5
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(2800)
        self._anim.setStartValue(-0.5)
        self._anim.setEndValue(1.5)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_pos)

    def _on_pos(self, v):
        self._pos = v
        self.update()

    def set_colors(self, base: str, highlight: str):
        self._base = QColor(base)
        self._hi = QColor(highlight)
        self.update()

    def start_shimmer(self):
        self._anim.start()

    def stop_shimmer(self):
        self._anim.stop()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setFont(self.font())
        w = self.width()
        sx = w * self._pos
        sw = int(w * 0.35)
        grad = QLinearGradient(sx, 0, sx + sw, 0)
        grad.setColorAt(0.0, self._base)
        grad.setColorAt(0.5, self._hi)
        grad.setColorAt(1.0, self._base)
        pen = QPen()
        pen.setBrush(QBrush(grad))
        p.setPen(pen)
        p.drawText(self.rect(), self.alignment(), self.text())


# ══════════════════════════════════════════════════════════════
#  Typewriter log (character-by-character text append)
# ══════════════════════════════════════════════════════════════

class TypewriterLog(QObject):
    """Appends messages to a QTextEdit character-by-character."""

    finished = pyqtSignal()

    def __init__(self, text_edit, chars_per_tick: int = 3,
                 interval_ms: int = 10, parent=None):
        super().__init__(parent or text_edit)
        self._edit = text_edit
        self._cpt = chars_per_tick
        self._queue: list = []
        self._current = ""
        self._idx = 0
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def append(self, text: str):
        """Queue a message for typewriter display."""
        self._queue.append(text)
        if not self._timer.isActive():
            self._next()

    def _next(self):
        if not self._queue:
            self.finished.emit()
            return
        self._current = self._queue.pop(0)
        self._idx = 0
        self._edit.append("")
        self._timer.start()

    def _tick(self):
        end = min(self._idx + self._cpt, len(self._current))
        chunk = self._current[self._idx:end]
        cursor = self._edit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(chunk)
        self._edit.setTextCursor(cursor)
        self._idx = end
        if self._idx >= len(self._current):
            self._timer.stop()
            self._next()


# ══════════════════════════════════════════════════════════════
#  Sidebar active pulse (left border throb on active nav item)
# ══════════════════════════════════════════════════════════════

class SidebarPulse(QObject):
    """Pulses a nav button's left-border opacity while processing."""

    def __init__(self, button: QWidget, color: str = "#D97757", parent=None):
        super().__init__(parent or button)
        self._btn = button
        self._color = color
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1400)
        self._anim.setStartValue(0.3)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.InOutSine)
        self._anim.valueChanged.connect(self._on_val)
        self._base_ss = ""

    def _on_val(self, v):
        c = QColor(self._color)
        border = f"rgba({c.red()},{c.green()},{c.blue()},{v:.2f})"
        self._btn.setStyleSheet(
            self._base_ss + f" border-left: 3px solid {border};"
        )

    def start(self):
        self._base_ss = self._btn.styleSheet()
        self._anim.start()

    def stop(self):
        self._anim.stop()
        self._btn.setStyleSheet(self._base_ss)


# ══════════════════════════════════════════════════════════════
#  ProgressGlowBar  — orange progress bar with sweep animation
# ══════════════════════════════════════════════════════════════

from PyQt5.QtWidgets import QProgressBar, QSizePolicy


class ProgressGlowBar(QProgressBar):
    """QProgressBar with animated white-sweep glow (ProgressGlow pattern)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sweep_pos = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setFixedHeight(8)
        self.setTextVisible(False)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_base_style()

    def _apply_base_style(self):
        from src.ui import theme
        self.setStyleSheet(f"""
            QProgressBar {{
                background-color: {theme.BORDER};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {theme.ORANGE};
                border-radius: 4px;
            }}
        """)

    def setValue(self, v: int):
        super().setValue(v)
        if v > 0 and not self._timer.isActive():
            self._timer.start()
        elif v == 0:
            self._timer.stop()

    def _tick(self):
        self._sweep_pos = (self._sweep_pos + 0.012) % 1.0
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.value() <= 0 or self.maximum() <= 0:
            return
        filled_w = int(self.width() * self.value() / self.maximum())
        if filled_w < 4:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        sweep_x = int((self._sweep_pos * 2 - 0.5) * filled_w)
        grad = QLinearGradient(sweep_x - filled_w, 0, sweep_x + filled_w, 0)
        grad.setColorAt(0.0, QColor(255, 255, 255, 0))
        grad.setColorAt(0.5, QColor(255, 255, 255, 55))
        grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setClipRect(0, 0, filled_w, self.height())
        p.fillRect(0, 0, filled_w, self.height(), grad)
        p.end()


# ══════════════════════════════════════════════════════════════
#  DrawerSlide  — slide a QWidget open/close on maximumWidth
# ══════════════════════════════════════════════════════════════

class DrawerSlide(QObject):
    """Animates a widget's maximumWidth to create a slide-in drawer effect."""

    def __init__(self, widget: QWidget, target_width: int = 380, duration: int = 250):
        super().__init__(widget)
        self._widget = widget
        self._target = target_width
        self._duration = duration
        self._anim = QPropertyAnimation(widget, b"maximumWidth", widget)
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def expand(self):
        self._widget.setMaximumWidth(0)
        self._widget.show()
        self._anim.stop()
        self._anim.setStartValue(self._widget.maximumWidth())
        self._anim.setEndValue(self._target)
        self._anim.start()

    def collapse(self, hide_on_finish: bool = True):
        self._anim.stop()
        self._anim.setStartValue(self._widget.maximumWidth())
        self._anim.setEndValue(0)
        if hide_on_finish:
            self._anim.finished.connect(self._widget.hide)
        self._anim.start()


# ══════════════════════════════════════════════════════════════
#  install_hover_lift  — 2px Y on hover via event filter
# ══════════════════════════════════════════════════════════════

class _HoverLiftFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            anim = QPropertyAnimation(obj, b"pos", obj)
            anim.setDuration(150)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            p = obj.pos()
            anim.setStartValue(p)
            anim.setEndValue(QPoint(p.x(), p.y() - 2))
            anim.start(QAbstractAnimation.DeleteWhenStopped)
        elif event.type() == QEvent.Leave:
            anim = QPropertyAnimation(obj, b"pos", obj)
            anim.setDuration(150)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            p = obj.pos()
            anim.setStartValue(p)
            anim.setEndValue(QPoint(p.x(), p.y() + 2))
            anim.start(QAbstractAnimation.DeleteWhenStopped)
        return False


def install_hover_lift(widget: QWidget):
    """Install 2px upward hover animation on any widget."""
    f = _HoverLiftFilter(widget)
    widget.installEventFilter(f)
    widget.setProperty("_hover_lift", True)
