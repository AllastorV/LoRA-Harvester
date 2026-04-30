import pytest
from src.ui import theme


def test_dark_palette_bg():
    theme.set_theme("dark")
    assert theme.BG_WINDOW == "#0f0f10"
    assert theme.BG_DEEPEST == "#080809"
    assert theme.SIDEBAR_BG == "#18181b"


def test_dark_palette_accent():
    theme.set_theme("dark")
    assert theme.ORANGE == "#e8832a"
    # ORANGE_LIGHT and TEXT_ACCENT are derived by _apply_accent() — just verify they differ from base
    assert theme.ORANGE_LIGHT != theme.ORANGE
    assert theme.TEXT_ACCENT != theme.ORANGE
    # Static palette tokens
    assert theme.TEXT_PRIMARY == "#f1dfd4"
    assert theme.TEXT_MUTED == "#a38c7d"


def test_dark_palette_surfaces():
    theme.set_theme("dark", accent="#e8832a")
    # BG_CARD/BG_SURFACE are now accent-tinted — verify they are dark and orange-tinted
    # (r > b for orange accent)
    def _rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, g, b = _rgb(theme.BG_CARD)
    assert r > b, f"BG_CARD should have orange tint: {theme.BG_CARD}"
    assert r < 40 and g < 30 and b < 20, f"BG_CARD should be very dark: {theme.BG_CARD}"
    assert theme.BORDER == "#27272a"


def test_accent_presets_count():
    assert len(theme.ACCENT_PRESETS) == 5
    colors = [c for c, _ in theme.ACCENT_PRESETS]
    assert "#e8832a" in colors
    assert "#3b82f6" in colors
    assert "#10b981" in colors
    assert "#a855f7" in colors
    assert "#ec4899" in colors


def test_set_theme_persists_accent():
    theme.set_theme("dark", 1.0, "#3b82f6")
    assert theme.ORANGE == "#3b82f6"
    theme.set_theme("dark", 1.0, "#e8832a")  # reset
