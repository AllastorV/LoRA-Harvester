"""
Text and Subtitle Detection Module
Detects text in frames to skip subtitle/text-heavy scenes

Improved v2.2:
- Multi-zone detection (bottom subtitle + top title + center watermark + corner logo)
- Morphological text-line grouping for quick_text_check
- High-contrast band detection for hard-coded subtitle backgrounds
- EasyOCR confidence filtering
- Configurable sensitivity
- Detection statistics tracking
"""

import cv2
import logging
import numpy as np
from typing import Tuple, Optional, List, Dict, NamedTuple

# Lazy import – easyocr is heavy and may not be installed
try:
    import easyocr as _easyocr
except ImportError:
    _easyocr = None

logger = logging.getLogger(__name__)


class SubtitleDetector:
    """Detects subtitles, watermarks and text overlays in video frames."""

    def __init__(self, languages: list = ['en', 'tr'],
                 sensitivity: str = 'normal'):
        """
        Initialize subtitle detector.

        Args:
            languages: List of languages to detect
            sensitivity: 'low', 'normal', or 'high'
                - low  : only skip obvious hard-sub with dark band
                - normal: balanced (default)
                - high : aggressive, catches faint soft-sub / watermarks too
        """
        self.languages = languages
        self.reader = None

        # Sensitivity presets
        presets = {
            'low':    {'min_text_area_ratio': 0.03,  'min_confidence': 0.5,
                       'quick_min_regions': 3, 'contrast_threshold': 60,
                       'check_center_zone': False, 'check_corner_logos': False},
            'normal': {'min_text_area_ratio': 0.015, 'min_confidence': 0.3,
                       'quick_min_regions': 2, 'contrast_threshold': 40,
                       'check_center_zone': True,  'check_corner_logos': True},
            'high':   {'min_text_area_ratio': 0.008, 'min_confidence': 0.15,
                       'quick_min_regions': 1, 'contrast_threshold': 25,
                       'check_center_zone': True,  'check_corner_logos': True},
        }
        cfg = presets.get(sensitivity, presets['normal'])

        self.min_text_area_ratio = cfg['min_text_area_ratio']
        self.min_confidence = cfg['min_confidence']
        self.quick_min_regions = cfg['quick_min_regions']
        self.contrast_threshold = cfg['contrast_threshold']
        self.check_center_zone = cfg['check_center_zone']
        self.check_corner_logos = cfg['check_corner_logos']

        # Detection zones (fraction of frame height / width)
        self.subtitle_bottom = 0.28   # bottom 28% – subtitle zone
        self.title_top = 0.12         # top 12% – title / episode-number zone
        self.center_zone_frac = 0.30  # center 30% (height) × 50% (width) – watermarks
        self.corner_size = 0.12       # corner squares (12% × 12%) – channel logos

        # Statistics
        self._stats = {
            'total_checked': 0,
            'detected_subtitle': 0,
            'detected_band': 0,
            'detected_watermark': 0,
            'detected_logo': 0,
            'ocr_used': 0,
        }

        logger.info("Initializing text detector (EasyOCR)...")
        try:
            if _easyocr is None:
                raise ImportError("easyocr not installed")
            import torch
            use_gpu = torch.cuda.is_available()
            self.reader = _easyocr.Reader(languages, gpu=use_gpu, verbose=False)
            logger.info("Text detector ready (GPU=%s)", use_gpu)
        except Exception as e:
            logger.warning("EasyOCR load failed, using quick mode only: %s", e)
            self.reader = None

    # ─────────────────── Full OCR detection ───────────────────────────

    def has_text(self, frame: np.ndarray,
                 check_subtitle_region: bool = True) -> Tuple[bool, float]:
        """
        Check if frame contains significant text using EasyOCR.

        Falls back to quick_text_check when EasyOCR is unavailable.

        Args:
            frame: Input frame (BGR)
            check_subtitle_region: If True, focus on subtitle zones

        Returns:
            (has_text, text_coverage_ratio)
        """
        self._stats['total_checked'] += 1

        if self.reader is None:
            result = self.quick_text_check(frame)
            return result, 0.0

        self._stats['ocr_used'] += 1
        height, width = frame.shape[:2]

        zones: List[np.ndarray] = []
        if check_subtitle_region:
            # Bottom subtitle zone
            crop_y = int(height * (1 - self.subtitle_bottom))
            zones.append(frame[crop_y:, :])
            # Top title zone
            top_y = int(height * self.title_top)
            zones.append(frame[:top_y, :])
            # Center watermark zone
            if self.check_center_zone:
                cy1 = int(height * (0.5 - self.center_zone_frac / 2))
                cy2 = int(height * (0.5 + self.center_zone_frac / 2))
                cx1 = int(width * 0.25)
                cx2 = int(width * 0.75)
                zones.append(frame[cy1:cy2, cx1:cx2])
        else:
            zones.append(frame)

        total_text_area = 0
        total_zone_area = 0

        for zone in zones:
            zone_area = zone.shape[0] * zone.shape[1]
            total_zone_area += zone_area

            rgb = cv2.cvtColor(zone, cv2.COLOR_BGR2RGB)
            try:
                results = self.reader.readtext(rgb)
            except Exception as e:
                logger.debug("OCR read error in zone: %s", e)
                continue

            for bbox, text, conf in results:
                if conf < self.min_confidence:
                    continue
                if len(text.strip()) < 2:
                    continue

                xs = [float(p[0]) for p in bbox]
                ys = [float(p[1]) for p in bbox]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                total_text_area += w * h

        text_ratio = total_text_area / total_zone_area if total_zone_area > 0 else 0.0
        return text_ratio > self.min_text_area_ratio, text_ratio

    # ─────────────────── Fast heuristic detection ─────────────────────

    def quick_text_check(self, frame: np.ndarray) -> bool:
        """
        Fast subtitle / watermark / logo detection using morphological
        text-line grouping and high-contrast band analysis.  No OCR needed.

        Returns True if the frame likely contains text overlays.
        """
        self._stats['total_checked'] += 1
        height, width = frame.shape[:2]

        # ── Bottom subtitle zone ──
        crop_y = int(height * (1 - self.subtitle_bottom))
        if self._zone_has_text_features(frame[crop_y:, :]):
            self._stats['detected_subtitle'] += 1
            return True

        # ── Hard-sub dark/bright band at bottom ──
        if self._has_subtitle_band(frame):
            self._stats['detected_band'] += 1
            return True

        # ── Top title zone ──
        top_y = int(height * self.title_top)
        if self._zone_has_text_features(frame[:top_y, :]):
            self._stats['detected_subtitle'] += 1
            return True

        # ── Center watermark zone (semi-transparent logos / copyright text) ──
        if self.check_center_zone:
            cy1 = int(height * (0.5 - self.center_zone_frac / 2))
            cy2 = int(height * (0.5 + self.center_zone_frac / 2))
            cx1 = int(width * 0.25)
            cx2 = int(width * 0.75)
            if self._zone_has_text_features(frame[cy1:cy2, cx1:cx2]):
                self._stats['detected_watermark'] += 1
                return True

        # ── Corner logo zones (top-left, top-right) ──
        if self.check_corner_logos:
            ch = int(height * self.corner_size)
            cw = int(width * self.corner_size)
            corners = [
                frame[:ch, :cw],           # top-left
                frame[:ch, width - cw:],   # top-right
            ]
            for corner in corners:
                if self._corner_has_logo(corner):
                    self._stats['detected_logo'] += 1
                    return True

        return False

    # ─────────────────── Zone analysis helpers ────────────────────────

    def _zone_has_text_features(self, zone: np.ndarray) -> bool:
        """
        Morphological text-line detection in a cropped zone.

        Strategy:
        1. Binarise (adaptive threshold)
        2. Dilate horizontally to merge characters into text lines
        3. Count connected components that look like text lines
        """
        zh, zw = zone.shape[:2]
        if zh < 10 or zw < 10:
            return False

        # Downsample for speed (target ~480px width)
        if zw > 480:
            scale = 480 / zw
            zone = cv2.resize(zone, (480, max(10, int(zh * scale))),
                              interpolation=cv2.INTER_AREA)
            zh, zw = zone.shape[:2]

        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold highlights text on any background
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )

        # Horizontal dilation merges individual chars → text lines
        kw = max(7, int(zw * 0.05))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_line_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0

            is_wide_enough = w > zw * 0.15
            is_reasonable_height = (zh * 0.05) < h < (zh * 0.50)
            is_text_aspect = aspect > 3.0

            if is_wide_enough and is_reasonable_height and is_text_aspect:
                text_line_count += 1

        return text_line_count >= self.quick_min_regions

    def _corner_has_logo(self, corner: np.ndarray) -> bool:
        """
        Detect channel / network logos in corner patches.

        A logo typically contains:
        - High-contrast blob(s) against the background
        - Compact shape (not spanning the full width like a subtitle)

        Uses a simpler threshold: look for compact high-contrast blobs
        that are roughly square (aspect ratio < 3).
        """
        ch, cw = corner.shape[:2]
        if ch < 8 or cw < 8:
            return False

        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)

        # Threshold to find bright/dark regions
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (ch * cw) * 0.05:   # ignore tiny noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            # Compact blob (not a long text line, not full-width)
            if aspect < 3.0 and w < cw * 0.90:
                return True

        return False

    def _has_subtitle_band(self, frame: np.ndarray) -> bool:
        """
        Detect hard-coded subtitle bands: a dark (or bright) horizontal
        band at the very bottom of the frame that spans most of the width.
        Common in anime / TV rips with burned-in subs on a solid strip.
        """
        height, width = frame.shape[:2]

        band_h = max(10, int(height * 0.08))
        band = frame[height - band_h:, :]

        gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)

        row_stds = np.std(gray_band, axis=1)
        mean_std = float(np.mean(row_stds))

        if mean_std > self.contrast_threshold:
            return False   # Too much variation → not a solid band

        above_band = frame[height - 2 * band_h:height - band_h, :]
        gray_above = cv2.cvtColor(above_band, cv2.COLOR_BGR2GRAY)

        band_mean = float(np.mean(gray_band))
        above_mean = float(np.mean(gray_above))

        return abs(band_mean - above_mean) > 30

    # ────────────────────────── Statistics ────────────────────────────

    def get_stats(self) -> Dict:
        """Return detection statistics."""
        return self._stats.copy()

    def print_stats(self):
        """Log detection statistics."""
        s = self._stats
        logger.info(
            "Text detector stats — checked=%d subtitle=%d band=%d "
            "watermark=%d logo=%d ocr_used=%d",
            s['total_checked'], s['detected_subtitle'], s['detected_band'],
            s['detected_watermark'], s['detected_logo'], s['ocr_used'],
        )

    # ──────────────────── Overlay Region Detection ────────────────────

    def detect_overlay_regions(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect the pixel coordinates of logo / watermark overlay regions.

        Instead of just returning True/False, this returns a list of
        bounding boxes [x1, y1, x2, y2] (in frame coordinates) for each
        detected overlay.  The cropper can then exclude these areas with
        a configurable margin.

        Detection order (fastest first):
        1. Corner logo patches (top-left, top-right)
        2. Center watermark zone (morphological)
        3. Bottom subtitle band
        4. Top title zone

        Returns:
            List of (x1, y1, x2, y2) tuples — empty if nothing detected.
        """
        height, width = frame.shape[:2]
        regions: List[Tuple[int, int, int, int]] = []

        # 1. Corner logos
        if self.check_corner_logos:
            ch = int(height * self.corner_size)
            cw = int(width * self.corner_size)
            corners = [
                (0,          0,           cw,          ch,          frame[:ch, :cw]),
                (width - cw, 0,           width,        ch,          frame[:ch, width - cw:]),
            ]
            for x1, y1, x2, y2, patch in corners:
                if self._corner_has_logo(patch):
                    regions.append((x1, y1, x2, y2))

        # 2. Center watermark zone
        if self.check_center_zone:
            cy1 = int(height * (0.5 - self.center_zone_frac / 2))
            cy2 = int(height * (0.5 + self.center_zone_frac / 2))
            cx1 = int(width * 0.25)
            cx2 = int(width * 0.75)
            center_patch = frame[cy1:cy2, cx1:cx2]
            if self._zone_has_text_features(center_patch):
                # Narrow the region to the actual text contours for a tighter fit
                tight = self._get_text_bbox(center_patch)
                if tight:
                    tx1, ty1, tx2, ty2 = tight
                    regions.append((cx1 + tx1, cy1 + ty1, cx1 + tx2, cy1 + ty2))
                else:
                    regions.append((cx1, cy1, cx2, cy2))

        # 3. Bottom subtitle band
        if self._has_subtitle_band(frame):
            band_h = max(10, int(height * 0.08))
            regions.append((0, height - band_h, width, height))

        # 4. Bottom subtitle zone (morphological)
        crop_y = int(height * (1 - self.subtitle_bottom))
        sub_patch = frame[crop_y:, :]
        if self._zone_has_text_features(sub_patch):
            tight = self._get_text_bbox(sub_patch)
            if tight:
                tx1, ty1, tx2, ty2 = tight
                regions.append((tx1, crop_y + ty1, tx2, crop_y + ty2))
            else:
                regions.append((0, crop_y, width, height))

        # 5. Top title zone
        top_y = int(height * self.title_top)
        top_patch = frame[:top_y, :]
        if self._zone_has_text_features(top_patch):
            tight = self._get_text_bbox(top_patch)
            if tight:
                tx1, ty1, tx2, ty2 = tight
                regions.append((tx1, ty1, tx2, ty2))
            else:
                regions.append((0, 0, width, top_y))

        return regions

    def _get_text_bbox(
        self, zone: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return tight bounding box (x1, y1, x2, y2) around the largest
        text region found in *zone*, or None if nothing found.

        Uses the same morphological pipeline as _zone_has_text_features
        but returns coordinates instead of a boolean.
        """
        zh, zw = zone.shape[:2]
        if zh < 10 or zw < 10:
            return None

        scale = 1.0
        if zw > 480:
            scale = 480 / zw
            zone = cv2.resize(zone, (480, max(10, int(zh * scale))),
                              interpolation=cv2.INTER_AREA)
            zh, zw = zone.shape[:2]

        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4,
        )
        kw = max(7, int(zw * 0.05))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        text_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if w > zw * 0.15 and (zh * 0.05) < h < (zh * 0.50) and aspect > 3.0:
                text_contours.append((x, y, w, h))

        if not text_contours:
            return None

        # Union bounding box of all text contours, then un-scale
        x1 = min(c[0] for c in text_contours)
        y1 = min(c[1] for c in text_contours)
        x2 = max(c[0] + c[2] for c in text_contours)
        y2 = max(c[1] + c[3] for c in text_contours)

        if scale != 1.0:
            inv = 1.0 / scale
            x1, y1, x2, y2 = (
                int(x1 * inv), int(y1 * inv),
                int(x2 * inv), int(y2 * inv),
            )

        return (x1, y1, x2, y2)
