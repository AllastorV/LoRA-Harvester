"""
Core module initialization
LoRA-Harvester - AI Powered Dataset Collection v2.0
"""

__version__ = "2.0.0"

# Core modules
from src.core.detector import ObjectDetector
from src.core.ensemble_detector import EnsembleDetector
from src.core.cropper import SmartCropper
from src.core.text_detector import SubtitleDetector
from src.core.video_processor import VideoProcessor
from src.core.unified_processor import UnifiedVideoProcessor

# Enhanced modules (v2.0)
from src.core.quality_analyzer import QualityAnalyzer, SceneChangeDetector
from src.core.enhanced_processor import EnhancedVideoProcessor, AsyncFrameSaver
from src.core.auto_captioner import AutoCaptioner, TagGenerator
from src.core.advanced_captioner import AdvancedCaptioner, TagSettings, BLIPCaptioner, WD14Tagger

__all__ = [
    # Core
    'ObjectDetector',
    'EnsembleDetector',
    'SmartCropper',
    'SubtitleDetector',
    'VideoProcessor',
    'UnifiedVideoProcessor',
    # Enhanced (v2.0)
    'QualityAnalyzer',
    'SceneChangeDetector',
    'EnhancedVideoProcessor',
    'AsyncFrameSaver',
    'AutoCaptioner',
    'TagGenerator',
    # Advanced Captioning (v2.0)
    'AdvancedCaptioner',
    'TagSettings',
    'BLIPCaptioner',
    'WD14Tagger',
]
