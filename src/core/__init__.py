"""
Core module initialization
LoRA-Harvester - AI Powered Dataset Collection v3.0

Heavy imports are lazy so lightweight CLI tools (kohya_export.py,
character_sort.py, dataset_scanner) can import from src.core without
pulling in cv2 / torch / PyQt5.
"""

__version__ = "3.0.0"

# ── Lazy import helpers ──────────────────────────────────────────────────────
# Use TYPE_CHECKING guard or plain function wrappers so nothing is loaded
# at package-import time.  IDE autocomplete still works via __all__.

def _get(module: str, *names):
    """Import *names* from *module* and return them (single or tuple)."""
    import importlib
    mod = importlib.import_module(module)
    objs = tuple(getattr(mod, n) for n in names)
    return objs[0] if len(objs) == 1 else objs


def __getattr__(name: str):
    """PEP 562 module-level __getattr__ — loads symbols on first access."""
    _MAP = {
        # Core
        'ObjectDetector':           ('src.core.detector',             'ObjectDetector'),
        'EnsembleDetector':         ('src.core.ensemble_detector',     'EnsembleDetector'),
        'SmartCropper':             ('src.core.cropper',               'SmartCropper'),
        'SubtitleDetector':         ('src.core.text_detector',         'SubtitleDetector'),
        'UnifiedVideoProcessor':    ('src.core.unified_processor',     'UnifiedVideoProcessor'),
        # Enhanced (v2.0)
        'QualityAnalyzer':          ('src.core.quality_analyzer',      'QualityAnalyzer'),
        'SceneChangeDetector':      ('src.core.quality_analyzer',      'SceneChangeDetector'),
        'EnhancedVideoProcessor':   ('src.core.enhanced_processor',    'EnhancedVideoProcessor'),
        'AsyncFrameSaver':          ('src.core.enhanced_processor',    'AsyncFrameSaver'),
        'AdvancedCaptioner':        ('src.core.advanced_captioner',    'AdvancedCaptioner'),
        'TagSettings':              ('src.core.advanced_captioner',    'TagSettings'),
        'WD14Tagger':               ('src.core.advanced_captioner',    'WD14Tagger'),
        # New (v2.2)
        'Florence2Captioner':       ('src.core.florence2_captioner',   'Florence2Captioner'),
        # v3.x
        'FrameUpscaler':            ('src.core.upscaler',              'FrameUpscaler'),
        'KohyaExporter':            ('src.core.kohya_exporter',        'KohyaExporter'),
        'scan_dataset':             ('src.core.dataset_scanner',       'scan_dataset'),
        'detect_concepts':          ('src.core.dataset_scanner',       'detect_concepts'),
    }
    if name in _MAP:
        module, attr = _MAP[name]
        return _get(module, attr)
    raise AttributeError(f"module 'src.core' has no attribute {name!r}")


__all__ = [
    # Core
    'ObjectDetector', 'EnsembleDetector', 'SmartCropper', 'SubtitleDetector',
    'UnifiedVideoProcessor',
    # Enhanced (v2.0)
    'QualityAnalyzer', 'SceneChangeDetector', 'EnhancedVideoProcessor',
    'AsyncFrameSaver', 'AdvancedCaptioner', 'TagSettings', 'WD14Tagger',
    # New (v2.2)
    'Florence2Captioner',
    # v3.x
    'FrameUpscaler', 'KohyaExporter', 'scan_dataset', 'detect_concepts',
]
