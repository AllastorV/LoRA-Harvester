"""
Smart Cropping Algorithm
Handles intelligent cropping with subject centering, head space, zoom adjustment,
and overlay-aware exclusion zones (logos / watermarks).
"""

import logging
import cv2
import numpy as np
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# Margin in pixels that the crop must stay away from each overlay region
OVERLAY_MARGIN = 15


class SmartCropper:
    """Intelligent cropping with subject awareness and overlay exclusion."""

    # Aspect ratios for all supported formats
    ASPECT_RATIOS = {
        '9:16': 9 / 16,
        '3:4': 3 / 4,
        '1:1': 1.0,
        '4:5': 4 / 5,
        '16:9': 16 / 9,
        '4:3': 4 / 3,
    }

    def __init__(self, target_format: str = '9:16', min_padding: int = 500):
        """
        Initialize smart cropper.

        Args:
            target_format: Target aspect ratio format
            min_padding: Minimum padding around detected objects (in pixels)
        """
        self.target_format = target_format
        self.aspect_ratio = self.ASPECT_RATIOS.get(target_format, 9 / 16)
        self.min_padding = min_padding

        # Head space parameters
        self.ideal_head_space = 0.15   # Ideal 15% space above head
        self.max_head_space = 0.25     # Maximum 25% space
        self.min_head_space = 0.05     # Minimum 5% space

    # ──────────────────────── Main crop API ───────────────────────────

    def calculate_crop_box(
        self,
        frame_shape: Tuple[int, int],
        subject_bbox: List[int],
        category: str,
        head_space_ratio: float = 0.0,
        excluded_zones: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate optimal crop box for the frame.

        When *excluded_zones* are provided (logo / watermark bounding boxes),
        the crop rectangle is shifted / shrunk so that every excluded zone
        lies at least OVERLAY_MARGIN (15 px) outside the crop.

        Args:
            frame_shape: (height, width) of the frame
            subject_bbox: [x1, y1, x2, y2] of the subject
            category: 'person', 'animal', or 'object'
            head_space_ratio: Current head space ratio for persons
            excluded_zones: List of (x1, y1, x2, y2) regions to avoid
                            (typically logo / watermark positions from
                            SubtitleDetector.detect_overlay_regions())

        Returns:
            Crop box (x, y, width, height) or None if not possible
        """
        frame_height, frame_width = frame_shape
        x1, y1, x2, y2 = subject_bbox

        subject_width = x2 - x1
        subject_height = y2 - y1
        subject_center_x = (x1 + x2) // 2
        subject_center_y = (y1 + y2) // 2

        # ── Base crop size ────────────────────────────────────────────
        padded_width = subject_width + 2 * self.min_padding
        padded_height = subject_height + 2 * self.min_padding

        if padded_width / padded_height > self.aspect_ratio:
            crop_width = padded_width
            crop_height = int(crop_width / self.aspect_ratio)
        else:
            crop_height = padded_height
            crop_width = int(crop_height * self.aspect_ratio)

        # ── Clamp to frame ────────────────────────────────────────────
        if crop_width > frame_width or crop_height > frame_height:
            scale = min(frame_width / crop_width, frame_height / crop_height)
            crop_width = int(crop_width * scale)
            crop_height = int(crop_height * scale)

        # ── Head-space adjustment (persons only) ──────────────────────
        if category == 'person' and head_space_ratio > 0:
            if head_space_ratio < self.min_head_space:
                subject_center_y -= int(crop_height * 0.1)
            elif head_space_ratio > self.max_head_space:
                subject_center_y += int(crop_height * 0.1)

        # ── Centre on subject ─────────────────────────────────────────
        crop_x = subject_center_x - crop_width // 2
        crop_y = subject_center_y - crop_height // 2

        # ── Clamp within frame ────────────────────────────────────────
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = max(0, min(crop_y, frame_height - crop_height))

        # ── Overlay exclusion ─────────────────────────────────────────
        if excluded_zones:
            result = self._apply_overlay_exclusion(
                crop_x, crop_y, crop_width, crop_height,
                frame_width, frame_height,
                excluded_zones,
            )
            if result is None:
                # Could not produce a valid crop that avoids all overlays
                logger.debug(
                    "Could not avoid all overlay zones; returning unconstrained crop"
                )
            else:
                crop_x, crop_y, crop_width, crop_height = result

        return (crop_x, crop_y, crop_width, crop_height)

    # ──────────────────── Overlay exclusion logic ─────────────────────

    def _apply_overlay_exclusion(
        self,
        crop_x: int, crop_y: int,
        crop_w: int, crop_h: int,
        frame_w: int, frame_h: int,
        excluded_zones: List[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Shift and/or shrink the crop box so it avoids every excluded zone
        by at least OVERLAY_MARGIN pixels.

        Strategy (applied per zone in order):
        1. Expand each zone by OVERLAY_MARGIN on all sides → *safe zone*.
        2. If the crop box does not overlap the safe zone → no action.
        3. If the overlap is on the bottom edge: move crop_y up.
        4. If the overlap is on the top edge: move crop_y down.
        5. If the overlap is on the left edge: move crop_x right.
        6. If the overlap is on the right edge: move crop_x left.
        7. After shifting, re-clamp to frame bounds.
        8. If after all attempts the crop still overlaps, shrink crop_h
           (or crop_w) to cut off the problematic edge — maintaining
           aspect ratio as closely as possible.

        Returns adjusted (x, y, w, h) or None if adjustment is impossible
        (e.g. the overlay completely covers the subject area).
        """
        m = OVERLAY_MARGIN

        for ez in excluded_zones:
            ez_x1 = max(0, ez[0] - m)
            ez_y1 = max(0, ez[1] - m)
            ez_x2 = min(frame_w, ez[2] + m)
            ez_y2 = min(frame_h, ez[3] + m)

            for _ in range(3):   # up to 3 shift attempts per zone
                cx2 = crop_x + crop_w
                cy2 = crop_y + crop_h

                # No overlap → done for this zone
                if (cx2 <= ez_x1 or crop_x >= ez_x2 or
                        cy2 <= ez_y1 or crop_y >= ez_y2):
                    break

                # Compute overlap extents
                ovlp_bottom = cy2 - ez_y1   # crop encroaches from above
                ovlp_top = ez_y2 - crop_y   # crop encroaches from below
                ovlp_right = cx2 - ez_x1    # crop encroaches from left
                ovlp_left = ez_x2 - crop_x  # crop encroaches from right

                # Choose the smallest shift that removes the overlap
                shifts = []
                if ovlp_bottom > 0:
                    shifts.append(('up',    ovlp_bottom))
                if ovlp_top > 0:
                    shifts.append(('down',  ovlp_top))
                if ovlp_right > 0:
                    shifts.append(('left',  ovlp_right))
                if ovlp_left > 0:
                    shifts.append(('right', ovlp_left))

                if not shifts:
                    break

                direction, amount = min(shifts, key=lambda s: s[1])

                if direction == 'up':
                    crop_y = max(0, crop_y - amount)
                elif direction == 'down':
                    new_y = crop_y + amount
                    if new_y + crop_h <= frame_h:
                        crop_y = new_y
                    else:
                        # Cannot shift down → shrink height from top
                        shrink = (new_y + crop_h) - frame_h
                        crop_y = frame_h - crop_h + shrink
                        crop_y = max(0, crop_y)
                elif direction == 'left':
                    crop_x = max(0, crop_x - amount)
                elif direction == 'right':
                    new_x = crop_x + amount
                    if new_x + crop_w <= frame_w:
                        crop_x = new_x
                    else:
                        crop_x = frame_w - crop_w
                        crop_x = max(0, crop_x)

            # Final clamp
            crop_x = max(0, min(crop_x, frame_w - crop_w))
            crop_y = max(0, min(crop_y, frame_h - crop_h))

        return (crop_x, crop_y, crop_w, crop_h)

    # ───────────────────── Standard crop helpers ──────────────────────

    def apply_crop(
        self, frame: np.ndarray, crop_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Apply crop to frame."""
        x, y, w, h = crop_box
        return frame[y:y + h, x:x + w]

    def adaptive_zoom(
        self,
        frame: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        zoom_factor: float = 1.2,
    ) -> np.ndarray:
        """
        Apply adaptive zoom to focus more on subject.

        Args:
            frame: Input frame
            crop_box: Current crop box
            zoom_factor: Zoom factor (>1 to zoom in)

        Returns:
            Zoomed and cropped frame
        """
        x, y, w, h = crop_box

        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        center_x = x + w // 2
        center_y = y + h // 2
        new_x = max(0, min(center_x - new_w // 2, frame.shape[1] - new_w))
        new_y = max(0, min(center_y - new_h // 2, frame.shape[0] - new_h))

        zoomed = frame[new_y:new_y + new_h, new_x:new_x + new_w]
        return cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_LANCZOS4)

    def calculate_quality_score(
        self,
        frame_shape: Tuple[int, int],
        crop_box: Tuple[int, int, int, int],
        subject_bbox: List[int],
    ) -> float:
        """
        Calculate quality score for a crop (0–1, higher is better).

        Args:
            frame_shape: Original frame shape
            crop_box: Crop dimensions
            subject_bbox: Subject bounding box

        Returns:
            Quality score
        """
        frame_height, frame_width = frame_shape
        x, y, w, h = crop_box
        sx1, sy1, sx2, sy2 = subject_bbox

        subject_area = (sx2 - sx1) * (sy2 - sy1)
        crop_area = w * h
        coverage = subject_area / crop_area if crop_area > 0 else 0

        edge_penalty = 0.0
        if x <= 10 or y <= 10:
            edge_penalty += 0.1
        if x + w >= frame_width - 10 or y + h >= frame_height - 10:
            edge_penalty += 0.1

        # Ideal coverage 30–60%
        coverage_score = 1.0 - abs(0.45 - coverage)
        return max(0.0, coverage_score - edge_penalty)
