"""
Text and Subtitle Detection Module
Detects text in frames to skip subtitle/text-heavy scenes

Improved v2.1:
- Multi-zone detection (bottom subtitle + top title)
- Morphological text-line grouping for quick_text_check
- High-contrast band detection for hard-coded subtitle backgrounds
- EasyOCR confidence filtering
- Configurable sensitivity
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

# Lazy import – easyocr is heavy and may not be installed
try:
    import easyocr as _easyocr
except ImportError:
    _easyocr = None


class SubtitleDetector:
    """Detects subtitles and text in video frames"""
    
    def __init__(self, languages: list = ['en', 'tr'],
                 sensitivity: str = 'normal'):
        """
        Initialize subtitle detector
        
        Args:
            languages: List of languages to detect
            sensitivity: 'low', 'normal', or 'high'
                - low  : only skip obvious hard-sub with dark band
                - normal: balanced (default)
                - high : aggressive, catches faint soft-sub too
        """
        self.languages = languages
        self.reader = None
        
        # Sensitivity presets
        presets = {
            'low':    {'min_text_area_ratio': 0.03,  'min_confidence': 0.5,
                       'quick_min_regions': 3, 'contrast_threshold': 60},
            'normal': {'min_text_area_ratio': 0.015, 'min_confidence': 0.3,
                       'quick_min_regions': 2, 'contrast_threshold': 40},
            'high':   {'min_text_area_ratio': 0.008, 'min_confidence': 0.15,
                       'quick_min_regions': 1, 'contrast_threshold': 25},
        }
        cfg = presets.get(sensitivity, presets['normal'])
        
        self.min_text_area_ratio = cfg['min_text_area_ratio']
        self.min_confidence = cfg['min_confidence']
        self.quick_min_regions = cfg['quick_min_regions']
        self.contrast_threshold = cfg['contrast_threshold']
        
        # Detection zones (fraction of frame height from bottom / top)
        self.subtitle_bottom = 0.28   # bottom 28% – subtitle zone
        self.title_top = 0.12         # top 12% – title / episode-number zone
        
        print("🔤 Initializing text detector (EasyOCR)...")
        try:
            if _easyocr is None:
                raise ImportError("easyocr not installed")
            import torch
            use_gpu = torch.cuda.is_available()
            self.reader = _easyocr.Reader(languages, gpu=use_gpu, verbose=False)
            print(f"✅ Text detector ready (GPU: {use_gpu})")
        except Exception as e:
            print(f"⚠️  EasyOCR load failed, using quick mode only: {e}")
            self.reader = None
    
    # ─────────────── Full OCR detection ───────────────
    
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
        if self.reader is None:
            return self.quick_text_check(frame), 0.0
        
        height, width = frame.shape[:2]
        
        zones: List[np.ndarray] = []
        if check_subtitle_region:
            # Bottom subtitle zone
            crop_y = int(height * (1 - self.subtitle_bottom))
            zones.append(frame[crop_y:, :])
            # Top title zone (episode numbers, channel logos with text)
            top_y = int(height * self.title_top)
            zones.append(frame[:top_y, :])
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
            except Exception:
                continue
            
            for bbox, text, conf in results:
                # Filter by confidence
                if conf < self.min_confidence:
                    continue
                # Filter very short strings (noise)
                if len(text.strip()) < 2:
                    continue
                
                xs = [float(p[0]) for p in bbox]
                ys = [float(p[1]) for p in bbox]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                total_text_area += w * h
        
        text_ratio = total_text_area / total_zone_area if total_zone_area > 0 else 0
        return text_ratio > self.min_text_area_ratio, text_ratio
    
    # ─────────────── Fast heuristic detection ───────────────
    
    def quick_text_check(self, frame: np.ndarray) -> bool:
        """
        Fast subtitle detection using morphological text-line grouping
        and high-contrast band analysis.  No OCR needed.
        
        Returns True if the frame likely contains subtitles / burned-in text.
        """
        height, width = frame.shape[:2]
        
        # ── Check bottom subtitle zone ──
        crop_y = int(height * (1 - self.subtitle_bottom))
        bottom_zone = frame[crop_y:, :]
        
        if self._zone_has_text_features(bottom_zone):
            return True
        
        # ── Check for hard-sub dark band at bottom ──
        if self._has_subtitle_band(frame):
            return True
        
        # ── Optionally check top zone for title text ──
        top_y = int(height * self.title_top)
        top_zone = frame[:top_y, :]
        if self._zone_has_text_features(top_zone):
            return True
        
        return False
    
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
        scale = 1.0
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
        
        # Horizontal dilation to merge individual chars → text lines
        # Kernel width ~5 % of zone width (min 7 px)
        kw = max(7, int(zw * 0.05))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find connected components
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_line_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            
            # A text line is wide, not too tall, and reasonably sized
            # Width must be > 15% of zone width (a subtitle spans a good portion)
            # Height must be between 5% – 50% of zone height
            # Aspect ratio > 3 (wider than tall)
            is_wide_enough = w > zw * 0.15
            is_reasonable_height = (zh * 0.05) < h < (zh * 0.50)
            is_text_aspect = aspect > 3.0
            
            if is_wide_enough and is_reasonable_height and is_text_aspect:
                text_line_count += 1
        
        return text_line_count >= self.quick_min_regions
    
    def _has_subtitle_band(self, frame: np.ndarray) -> bool:
        """
        Detect hard-coded subtitle bands: a dark (or bright) horizontal
        band at the very bottom of the frame that spans most of the width.
        Common in anime / TV rips with burned-in subs on a solid strip.
        """
        height, width = frame.shape[:2]
        
        # Examine the bottom 8% of the frame
        band_h = max(10, int(height * 0.08))
        band = frame[height - band_h:, :]
        
        gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        
        # A subtitle band has low variance (it's a solid-ish bar)
        row_stds = np.std(gray_band, axis=1)
        mean_std = np.mean(row_stds)
        
        if mean_std > self.contrast_threshold:
            # Too much variation → not a solid band
            return False
        
        # Compare brightness of the band vs the region just above it
        above_band = frame[height - 2 * band_h:height - band_h, :]
        gray_above = cv2.cvtColor(above_band, cv2.COLOR_BGR2GRAY)
        
        band_mean = np.mean(gray_band)
        above_mean = np.mean(gray_above)
        
        brightness_diff = abs(float(band_mean) - float(above_mean))
        
        # A distinct brightness shift signals a hard-sub bar
        return brightness_diff > 30
