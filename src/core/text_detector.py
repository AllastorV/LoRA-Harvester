"""
Text and Subtitle Detection Module
Detects text in frames to skip subtitle/text-heavy scenes
"""

import cv2
import numpy as np
import easyocr
from typing import Tuple, Optional


class SubtitleDetector:
    """Detects subtitles and text in video frames"""
    
    def __init__(self, languages: list = ['en', 'tr']):
        """
        Initialize subtitle detector
        
        Args:
            languages: List of languages to detect
        """
        print("🔤 Initializing text detector (EasyOCR)...")
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(languages, gpu=use_gpu, verbose=False)
            print(f"✅ Text detector ready (GPU: {use_gpu})")
        except Exception as e:
            print(f"⚠️  EasyOCR load failed, using quick mode only: {e}")
            self.reader = None
        
        # Text detection parameters
        self.min_text_area_ratio = 0.015  # Minimum text area relative to frame (lowered for better detection)
        self.subtitle_region_height = 0.25  # Bottom 25% of frame for subtitle check
    
    def has_text(self, frame: np.ndarray, check_subtitle_region: bool = True) -> Tuple[bool, float]:
        """
        Check if frame contains significant text
        
        Args:
            frame: Input frame (BGR format)
            check_subtitle_region: If True, focus on bottom region for subtitles
            
        Returns:
            Tuple of (has_text, text_coverage_ratio)
        """
        # Fallback to quick check if reader not available
        if self.reader is None:
            return self.quick_text_check(frame), 0.0
        
        height, width = frame.shape[:2]
        
        # If checking subtitle region, crop to bottom portion
        if check_subtitle_region:
            crop_y = int(height * (1 - self.subtitle_region_height))
            detection_frame = frame[crop_y:, :]
        else:
            detection_frame = frame
        
        # Convert to RGB for EasyOCR
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        
        # Detect text
        try:
            results = self.reader.readtext(rgb_frame)
            
            if not results:
                return False, 0.0
            
            # Calculate total text area
            total_text_area = 0
            detection_area = detection_frame.shape[0] * detection_frame.shape[1]
            
            for detection in results:
                bbox = detection[0]
                # Calculate bounding box area
                x_coords = [float(point[0]) for point in bbox]
                y_coords = [float(point[1]) for point in bbox]
                width_box = max(x_coords) - min(x_coords)
                height_box = max(y_coords) - min(y_coords)
                total_text_area += width_box * height_box
            
            text_ratio = total_text_area / detection_area if detection_area > 0 else 0
            
            # Return True if text coverage exceeds threshold
            has_significant_text = text_ratio > self.min_text_area_ratio
            
            return has_significant_text, text_ratio
            
        except Exception as e:
            # Fallback to quick check on error
            return self.quick_text_check(frame), 0.0
    
    def quick_text_check(self, frame: np.ndarray) -> bool:
        """
        Fast text detection using edge detection (for performance)
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            True if likely contains text
        """
        height, width = frame.shape[:2]
        
        # Check bottom region for subtitles
        crop_y = int(height * (1 - self.subtitle_region_height))
        subtitle_region = frame[crop_y:, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine binary and edges
        combined = cv2.bitwise_or(binary, edges)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count text-like regions
        text_regions = 0
        subtitle_height = subtitle_region.shape[0]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Text characteristics:
            # - High aspect ratio (wide)
            # - Reasonable size
            # - In subtitle zone (bottom half)
            if (aspect_ratio > 2.5 and 
                w > width * 0.08 and 
                area > 100 and
                y > subtitle_height * 0.3):
                text_regions += 1
        
        # If multiple text-like regions detected, likely subtitle
        return text_regions >= 2
