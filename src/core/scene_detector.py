"""
Scene-based keyframe extraction using PySceneDetect.

Replaces fixed-interval frame extraction with intelligent scene-boundary
detection. Only extracts unique keyframes at shot transitions, eliminating
duplicate/redundant frames from static shots.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


def detect_scene_keyframes(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> List[int]:
    """
    Detect scene boundaries and return one keyframe number per scene.

    Uses the ContentDetector from PySceneDetect which compares
    adjacent frames using HSV-based content scoring.

    Args:
        video_path: Path to video file.
        threshold: Content-change sensitivity (lower = more scenes).
                   Default 27.0 works well for anime/live action.
        min_scene_len: Minimum scene length in frames.
        start_frame: Skip frames before this number.
        end_frame: Stop after this frame.

    Returns:
        Sorted list of frame numbers (one per detected scene).
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        logger.warning(
            "scenedetect not installed — falling back to fixed interval. "
            "Install with: pip install scenedetect[opencv]"
        )
        return []

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )

    # Seek to start if trimming.
    if start_frame > 0:
        video.seek(start_frame)

    total = end_frame if end_frame else video.duration.get_frames()
    scene_manager.detect_scenes(video, end_time=total)
    scene_list = scene_manager.get_scene_list()

    # Pick the midpoint of each scene as the representative keyframe.
    keyframes: List[int] = []
    for start, end in scene_list:
        mid = (start.get_frames() + end.get_frames()) // 2
        if start_frame <= mid and (end_frame is None or mid <= end_frame):
            keyframes.append(mid)

    # If no scenes detected (single-shot video), pick the first frame.
    if not keyframes and total > 0:
        keyframes.append(start_frame if start_frame > 0 else 0)

    logger.info(
        "PySceneDetect: %d scenes → %d keyframes in %s",
        len(scene_list), len(keyframes), Path(video_path).name,
    )
    return sorted(keyframes)


def keyframes_as_set(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> set:
    """
    Same as detect_scene_keyframes but returns a set for O(1) lookup.
    Convenient for the frame-by-frame loop in unified_processor.py.
    """
    return set(
        detect_scene_keyframes(
            video_path, threshold, min_scene_len, start_frame, end_frame
        )
    )
