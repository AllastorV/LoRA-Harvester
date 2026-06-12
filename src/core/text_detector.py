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
from typing import Tuple, Optional, List, Dict

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
        # NOTE: detect_colored / detect_stroke / the edge-density band check are
        # AGGRESSIVE and cause heavy false-positives on anime/illustration art
        # (thick outlines + saturated colors look like text). They are therefore
        # OFF for 'low' and 'normal' (default) and only enabled on 'high'.
        presets = {
            'low':    {'min_text_area_ratio': 0.03,  'min_confidence': 0.5,
                       'quick_min_regions': 3, 'contrast_threshold': 60,
                       'check_center_zone': False, 'check_corner_logos': False,
                       'detect_colored': False, 'detect_stroke': False,
                       'edge_density_threshold': 9999.0},   # disabled
            'normal': {'min_text_area_ratio': 0.015, 'min_confidence': 0.3,
                       # 1 line is enough — most hard-subs are single-line;
                       # requiring 2 lines made detection miss them entirely.
                       'quick_min_regions': 1, 'contrast_threshold': 40,
                       # center watermark + corner logo detection fire on anime
                       # character art → disabled by default; only bottom subtitle
                       # + solid-band detection run (real hardcoded subs).
                       'check_center_zone': False, 'check_corner_logos': False,
                       'detect_colored': False, 'detect_stroke': False,
                       'edge_density_threshold': 9999.0},   # disabled (anime-safe)
            'high':   {'min_text_area_ratio': 0.008, 'min_confidence': 0.15,
                       'quick_min_regions': 1, 'contrast_threshold': 25,
                       'check_center_zone': True,  'check_corner_logos': True,
                       'detect_colored': True,  'detect_stroke': True,
                       'edge_density_threshold': 8.0},
        }
        cfg = presets.get(sensitivity, presets['normal'])

        self.min_text_area_ratio = cfg['min_text_area_ratio']
        self.min_confidence = cfg['min_confidence']
        self.quick_min_regions = cfg['quick_min_regions']
        self.contrast_threshold = cfg['contrast_threshold']
        self.check_center_zone = cfg['check_center_zone']
        self.check_corner_logos = cfg['check_corner_logos']
        self.detect_colored = cfg['detect_colored']
        self.detect_stroke = cfg['detect_stroke']
        self.edge_density_threshold = cfg['edge_density_threshold']

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
            'detected_colored': 0,
            'detected_stroke': 0,
            'detected_freescan': 0,
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

        # ── Free-position scan (vertical video / letterbox / custom subs) ──
        # Subtitles outside the fixed zones (e.g. centered subs in 9:16
        # shorts, subs inside letterbox bars) are caught here.
        if len(self.scan_text_lines(frame)) >= self.quick_min_regions:
            self._stats['detected_freescan'] += 1
            return True

        return False

    # ─────────────────── Free-position text scan ──────────────────────

    def scan_text_lines(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Scan the WHOLE frame for subtitle-like text lines, independent of
        the fixed zone fractions.  Handles any frame size / aspect ratio:
        vertical (9:16) videos with centered subs, letterboxed content
        with subs in the black bars, top-positioned subs, etc.

        Filters are frame-relative: a text line must span >12% of the
        frame width, be 1–8% of the frame height tall, and have aspect
        ratio > 3.5.  Adjacent lines (multi-line subtitles) are merged
        into one block.

        Returns:
            List of merged (x1, y1, x2, y2) boxes in frame coordinates.
        """
        fh, fw = frame.shape[:2]
        if fh < 20 or fw < 20:
            return []

        scale = 1.0
        img = frame
        if fw > 640:
            scale = 640 / fw
            img = cv2.resize(frame, (640, max(20, int(fh * scale))),
                             interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Dual direction with negative C for light text (see
        # _threshold_text_check for why positive C is wrong here)
        binary_dark = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10)
        binary_light = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -10)
        binary = cv2.bitwise_or(binary_dark, binary_light)

        kw = max(7, int(iw * 0.04))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if (w > iw * 0.12
                    and (ih * 0.01) < h < (ih * 0.08)
                    and aspect > 3.5):
                lines.append((x, y, x + w, y + h))

        if not lines:
            return []

        merged = self._merge_line_boxes(lines)

        if scale != 1.0:
            inv = 1.0 / scale
            merged = [
                (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
                for x1, y1, x2, y2 in merged
            ]
        return merged

    @staticmethod
    def _merge_line_boxes(
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Merge text-line boxes that belong to the same multi-line subtitle:
        vertical gap < 1.5 × line height and overlapping x-ranges.
        """
        boxes = sorted(boxes, key=lambda b: b[1])
        merged: List[List[int]] = []
        for x1, y1, x2, y2 in boxes:
            line_h = y2 - y1
            for m in merged:
                gap = y1 - m[3]
                x_overlap = min(x2, m[2]) - max(x1, m[0])
                if gap < 1.5 * max(line_h, m[3] - m[1]) and x_overlap > 0:
                    m[0] = min(m[0], x1)
                    m[1] = min(m[1], y1)
                    m[2] = max(m[2], x2)
                    m[3] = max(m[3], y2)
                    break
            else:
                merged.append([x1, y1, x2, y2])
        return [tuple(m) for m in merged]

    # ─────────────────── Zone analysis helpers ────────────────────────

    def _zone_has_text_features(self, zone: np.ndarray) -> bool:
        """
        Text-line detection in a cropped zone — multi-pass:
        1. Dual-direction adaptive threshold (both dark-on-light and light-on-dark)
        2. Colored text (yellow, cyan, white, red) via HSV masking
        3. Stroke/outline-based detection (anime thick outlines)
        """
        zh, zw = zone.shape[:2]
        if zh < 10 or zw < 10:
            return False

        # Fast path: dual-direction threshold
        if self._threshold_text_check(zone):
            return True

        # Colored subtitle detection (sarı/cyan/beyaz/kırmızı)
        if self.detect_colored and self._detect_colored_text(zone):
            self._stats['detected_colored'] += 1
            return True

        # Stroke/outline detection (anime kalın outline)
        if self.detect_stroke and self._detect_stroke_text(zone):
            self._stats['detected_stroke'] += 1
            return True

        return False

    def _threshold_text_check(self, zone: np.ndarray) -> bool:
        """
        Dual-direction adaptive threshold text detection.
        Detects both light-on-dark AND dark-on-light text.
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

        # Dual direction: dark text (below local mean) + light text (above).
        # NOTE: THRESH_BINARY with a POSITIVE C fires on nearly every pixel
        # (src > mean - C) and saturates the whole zone white, which made
        # detection always fail. The light-text mask needs a NEGATIVE C so
        # only pixels clearly brighter than the local mean trigger.
        binary_dark = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        binary_light = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -10
        )
        binary = cv2.bitwise_or(binary_dark, binary_light)

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
            if (w > zw * 0.15
                    and (zh * 0.05) < h < (zh * 0.50)
                    and aspect > 3.0):
                text_line_count += 1

        return text_line_count >= self.quick_min_regions

    def _detect_colored_text(self, zone: np.ndarray) -> bool:
        """
        Detect colored subtitles (yellow, white, cyan, red) via HSV masking.
        Common in anime, TV broadcasts, game UI.
        Minimum 800px region to avoid false positives from small colored objects.
        """
        zh, zw = zone.shape[:2]
        if zh < 10 or zw < 10:
            return False

        if zw > 480:
            scale = 480 / zw
            zone = cv2.resize(zone, (480, max(10, int(zh * scale))),
                              interpolation=cv2.INTER_AREA)
            zh, zw = zone.shape[:2]

        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Color masks for common subtitle colors
        color_masks = {
            'white':  (v > 200) & (s < 50),
            'yellow': (h >= 20) & (h <= 40)  & (s > 120) & (v > 150),
            'cyan':   (h >= 85) & (h <= 100) & (s > 100) & (v > 120),
            'red_lo': (h <= 10)              & (s > 120) & (v > 100),
            'red_hi': (h >= 170)             & (s > 120) & (v > 100),
        }

        kw = max(7, int(zw * 0.05))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        min_area = 800  # minimum piksel alanı

        for color_name, mask_bool in color_masks.items():
            mask = mask_bool.astype(np.uint8) * 255
            if cv2.countNonZero(mask) < 50:
                continue

            dilated = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            text_line_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, hh = cv2.boundingRect(cnt)
                aspect = w / hh if hh > 0 else 0
                if (w > zw * 0.15
                        and (zh * 0.05) < hh < (zh * 0.55)
                        and aspect > 2.5):
                    text_line_count += 1

            if text_line_count >= self.quick_min_regions:
                logger.debug("Colored text detected: %s (%d lines)", color_name, text_line_count)
                return True

        return False

    def _detect_stroke_text(self, zone: np.ndarray) -> bool:
        """
        Detect text via stroke/outline gradient magnitude.
        Anime subs have thick dark outlines → strong Sobel response
        regardless of text color or background.
        """
        zh, zw = zone.shape[:2]
        if zh < 10 or zw < 10:
            return False

        if zw > 480:
            scale = 480 / zw
            zone = cv2.resize(zone, (480, max(10, int(zh * scale))),
                              interpolation=cv2.INTER_AREA)
            zh, zw = zone.shape[:2]

        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)

        # Strong gradient threshold (outline pixels)
        strong = (mag > 60).astype(np.uint8) * 255

        kw = max(7, int(zw * 0.05))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        dilated = cv2.dilate(strong, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_line_count = 0
        for cnt in contours:
            x, y, w, hh = cv2.boundingRect(cnt)
            aspect = w / hh if hh > 0 else 0
            if (w > zw * 0.15
                    and (zh * 0.05) < hh < (zh * 0.50)
                    and aspect > 3.0):
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
        """True if a hard-coded subtitle band exists at the bottom."""
        return self._find_subtitle_band(frame) > 0

    def _find_subtitle_band(self, frame: np.ndarray) -> int:
        """
        Detect hard-coded subtitle bands at the bottom of the frame and
        return the band height in pixels (0 = no band).

        Band heights are probed from large to small (20% → 6% of the frame
        height) so videos with different band/letterbox sizes all match,
        and the LARGEST uniform band wins (a too-large probe that includes
        scene content fails the uniformity check and falls through).

        Two detection methods (either triggers a match):
        1. Solid color band: low row variance + brightness contrast vs above
           (classic dark letterbox + white text)
        2. Edge density: high horizontal Sobel in the band even without solid bg
           (semi-transparent or gradient subtitle backgrounds)
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for frac in (0.20, 0.14, 0.10, 0.06):
            band_h = max(10, int(height * frac))
            if band_h * 2 > height:
                continue
            gray_band = gray[height - band_h:, :]

            # Method 1: solid color band
            row_stds = np.std(gray_band, axis=1)
            if float(np.mean(row_stds)) <= self.contrast_threshold:
                gray_above = gray[height - 2 * band_h:height - band_h, :]
                band_mean = float(np.mean(gray_band))
                above_mean = float(np.mean(gray_above))
                if abs(band_mean - above_mean) > 30:
                    return band_h

            # Method 2: horizontal edge density
            sx = cv2.Sobel(gray_band, cv2.CV_32F, 1, 0, ksize=3)
            if float(np.mean(np.abs(sx))) > self.edge_density_threshold:
                return band_h

        return 0

    # ──────────────────── Subtitle Removal ───────────────────────────

    def remove_subtitle_regions(
        self,
        frame: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        crop_margin: int = 5,
    ) -> np.ndarray:
        """
        Remove subtitle regions by **cropping the frame** just above the
        detected subtitle area.

        Finds the topmost Y coordinate across all detected regions, then
        returns `frame[:top_y - crop_margin]`.  The image becomes slightly
        shorter but is completely subtitle-free with no reconstruction artefacts.

        Args:
            frame:        Input BGR frame.
            regions:      List of (x1, y1, x2, y2) bounding boxes to remove.
            crop_margin:  Pixels above the topmost subtitle row to keep.
                          Default 5 — ensures the cut is clean.

        Returns:
            Cropped frame (shorter height, no subtitles).
            Returns original frame unchanged if regions is empty.
        """
        if not regions:
            return frame

        # Cropping can only remove regions near the frame edges:
        #   - bottom-half regions  → cut the frame from below
        #   - top-quarter regions  → cut the frame from above
        # Mid-frame regions are left for exclusion-aware cropping.
        frame_h = frame.shape[0]
        bottom_cut = frame_h
        top_cut = 0
        for r in regions:
            if r[1] >= frame_h * 0.5:
                bottom_cut = min(bottom_cut, max(0, r[1] - crop_margin))
            elif r[3] <= frame_h * 0.25:
                top_cut = max(top_cut, min(frame_h, r[3] + crop_margin))

        if top_cut == 0 and bottom_cut == frame_h:
            return frame

        # Keep at least 30% of the frame, otherwise removal destroys it
        if bottom_cut - top_cut < max(10, int(frame_h * 0.30)):
            return frame

        return frame[top_cut:bottom_cut, :].copy()

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

        # 3. Bottom subtitle band (variable height — letterbox sizes differ)
        band_h = self._find_subtitle_band(frame)
        if band_h > 0:
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

        # 6. Free-position text lines anywhere in the frame (vertical video
        #    centered subs, letterboxed subs, custom positions). Skip boxes
        #    already covered by an existing region.
        for box in self.scan_text_lines(frame):
            bx1, by1, bx2, by2 = box
            covered = any(
                bx1 >= rx1 and by1 >= ry1 and bx2 <= rx2 and by2 <= ry2
                for rx1, ry1, rx2, ry2 in regions
            )
            if not covered:
                regions.append(box)

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
        binary_dark = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10,
        )
        binary_light = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -10,
        )
        binary = cv2.bitwise_or(binary_dark, binary_light)
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
