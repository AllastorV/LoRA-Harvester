"""
UI module initialization v3.0

Lazy exports so importing a stdlib-only submodule (e.g. translations)
does not pull in PyQt5 via this package __init__.
"""

__version__ = "3.0.0"

__all__ = ["QualitySettingsPanel", "CaptioningSettingsPanel", "TagSettingsPanel"]


def __getattr__(name):
    """PEP 562 lazy attribute access — loads PyQt5-dependent panels on demand."""
    if name in __all__:
        import importlib
        mod = importlib.import_module("src.ui.advanced_settings")
        return getattr(mod, name)
    raise AttributeError(f"module 'src.ui' has no attribute {name!r}")
