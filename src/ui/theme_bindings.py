"""Live stylesheet bindings for Qt widgets, with no widget-tree rescans.

A rendered QSS string freezes the palette at creation time. Register a *factory*
instead: ``theme.bind_style(card, lambda: f'background: {theme.BG_CARD};')``.
Factories must be side-effect-free and run on the Qt GUI thread. They must not
load data/models, rebuild widgets or change application state.

Callbacks live on the widget; the registry contains weak references only. This
also covers cards added after startup, without retaining discarded thumbnails.
This module has no Qt import, so its lifecycle/deduplication logic is unit-testable
without a display server or the ML dependencies.
"""
from __future__ import annotations

import logging
import weakref
from typing import Callable

_LOG = logging.getLogger(__name__)
_widgets: dict[int, weakref.ReferenceType] = {}


def _remove(key, reference):
    if _widgets.get(key) is reference:
        _widgets.pop(key, None)


def set_style_if_changed(widget, stylesheet: str) -> bool:
    """Avoid expensive Qt repolishing when the rendered stylesheet is identical."""
    if not isinstance(stylesheet, str):
        raise TypeError('A theme stylesheet factory must return str')
    if widget.styleSheet() == stylesheet:
        return False
    widget.setStyleSheet(stylesheet)
    return True


def bind_style(widget, factory: Callable[[], str]):
    """Apply and remember a live style; subsequent state changes replace it.

    Capture loop indices and fixed per-widget parameters in lambda defaults.
    Refer to palette tokens *inside* the factory, not to pre-rendered strings.
    """
    if not callable(factory):
        raise TypeError('bind_style requires a callable, not a rendered QSS string')
    stylesheet = factory()
    set_style_if_changed(widget, stylesheet)
    widget._lh_theme_style_factory = factory
    key = id(widget)
    old_reference = _widgets.get(key)
    if old_reference is None or old_reference() is not widget:
        reference = weakref.ref(widget, lambda ref, k=key: _remove(k, ref))
        _widgets[key] = reference
        # C++ deletion can happen before the Python wrapper is garbage-collected.
        signal = getattr(widget, 'destroyed', None)
        if signal is not None:
            signal.connect(lambda *args, k=key, ref=reference: _remove(k, ref))
    return widget


def render_style(widget):
    """Return the current base style, even while an animation overlays its QSS."""
    factory = getattr(widget, '_lh_theme_style_factory', None)
    return factory() if factory is not None else widget.styleSheet()


def refresh_styles(root=None):
    """Refresh registered widgets once, optionally within a particular page.

    Does not rebuild content or overwrite styles by guessing border radii/types.
    Preserve a snapshot because QObject destruction may modify the registry.
    Unexpected factory errors are logged and counted, not silently swallowed.
    """
    stats = {'visited': 0, 'updated': 0, 'failed': 0}
    for key, reference in tuple(_widgets.items()):
        widget = reference()
        if widget is None:
            _remove(key, reference)
            continue
        try:
            if root is not None and widget is not root and not root.isAncestorOf(widget):
                continue
            stats['visited'] += 1
            factory = widget._lh_theme_style_factory
            stats['updated'] += int(set_style_if_changed(widget, factory()))
        except RuntimeError as exc:
            if 'has been deleted' in str(exc):
                _remove(key, reference)
            else:
                stats['failed'] += 1
                _LOG.exception('Theme refresh failed for %s', type(widget).__name__)
        except Exception:
            stats['failed'] += 1
            _LOG.exception('Theme refresh failed for %s', type(widget).__name__)
    return stats
