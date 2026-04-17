"""
Core module initialization
LoRA-Harvester - AI Powered Dataset Collection v3.0
"""

__version__ = "2.2.0"

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
from src.core.auto_captioner import TagGenerator
from src.core.advanced_captioner import AdvancedCaptioner, TagSettings, WD14Tagger

# New modules (v2.2)
from src.core.florence2_captioner import Florence2Captioner
from src.core.scene_detector import detect_scene_keyframes, keyframes_as_set
from src.core.sam_segmenter import SAMSegmenter
from src.core.memory_optimizations import get_available_backends, apply_torch_optimizations

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
    'TagGenerator',
    # Advanced Captioning (v2.0)
    'AdvancedCaptioner',
    'TagSettings',
    'WD14Tagger',
    # New (v2.2)
    'Florence2Captioner',
    'detect_scene_keyframes',
    'keyframes_as_set',
    'SAMSegmenter',
    'get_available_backends',
    'apply_torch_optimizations',
]
