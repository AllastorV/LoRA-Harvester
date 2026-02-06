"""
Quality Analyzer Module for LoRA-Harvester
Analyzes frame quality: blur, lighting, duplicates, composition
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import deque
import hashlib


class QualityAnalyzer:
    """
    Comprehensive frame quality analysis:
    - Blur detection (Laplacian variance)
    - Lighting analysis (histogram)
    - Duplicate detection (perceptual hash)
    - Composition scoring
    - Motion blur detection
    """
    
    def __init__(self, 
                 blur_threshold: float = 100.0,
                 brightness_range: Tuple[int, int] = (40, 220),
                 duplicate_threshold: float = 0.92,
                 history_size: int = 50):
        """
        Initialize quality analyzer
        
        Args:
            blur_threshold: Minimum Laplacian variance (higher = sharper required)
            brightness_range: Acceptable mean brightness range (0-255)
            duplicate_threshold: Similarity threshold for duplicate detection (0-1)
            history_size: Number of recent frames to check for duplicates
        """
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.duplicate_threshold = duplicate_threshold
        self.history_size = history_size
        
        # Frame history for duplicate detection
        self.frame_hashes: deque = deque(maxlen=history_size)
        self.frame_histograms: deque = deque(maxlen=history_size)
        
        # Statistics
        self.stats = {
            'analyzed': 0,
            'blur_rejected': 0,
            'dark_rejected': 0,
            'bright_rejected': 0,
            'duplicate_rejected': 0,
            'low_contrast_rejected': 0
        }
        
        print("🔍 Quality Analyzer initialized")
        print(f"   Blur threshold: {blur_threshold}")
        print(f"   Brightness range: {brightness_range}")
        print(f"   Duplicate threshold: {duplicate_threshold:.0%}")
    
    def check_frame_quality(self, frame: np.ndarray,
                           check_blur: bool = True,
                           check_lighting: bool = True,
                           check_duplicate: bool = True) -> Tuple[bool, Dict]:
        """
        Wrapper for analyze_frame() - compatibility method
        Returns: (is_quality_ok, quality_info)
        """
        return self.analyze_frame(frame, check_blur, check_lighting, check_duplicate)
    
    def analyze_frame(self, frame: np.ndarray, 
                     check_blur: bool = True,
                     check_lighting: bool = True,
                     check_duplicate: bool = True) -> Tuple[bool, Dict]:
        """
        Analyze frame quality
        
        Args:
            frame: Input frame (BGR)
            check_blur: Enable blur detection
            check_lighting: Enable lighting analysis
            check_duplicate: Enable duplicate detection
            
        Returns:
            Tuple of (is_quality_ok, analysis_details)
        """
        self.stats['analyzed'] += 1
        
        analysis = {
            'blur_score': 0.0,
            'brightness': 0,
            'contrast': 0.0,
            'is_duplicate': False,
            'duplicate_similarity': 0.0,
            'quality_score': 0.0,
            'rejection_reason': None
        }
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downsample for faster analysis (512px width max)
        h, w = gray.shape[:2]
        if w > 512:
            scale = 512 / w
            small_gray = cv2.resize(gray, (512, int(h * scale)), interpolation=cv2.INTER_AREA)
            small_frame = cv2.resize(frame, (512, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small_gray = gray
            small_frame = frame
        
        # 1. Blur Detection (on downsampled)
        if check_blur:
            blur_score = self._calculate_blur_score(small_gray)
            analysis['blur_score'] = blur_score
            
            if blur_score < self.blur_threshold:
                analysis['rejection_reason'] = 'blur'
                self.stats['blur_rejected'] += 1
                return False, analysis
        
        # 2. Lighting Analysis (on downsampled)
        if check_lighting:
            brightness, contrast = self._analyze_lighting(small_gray)
            analysis['brightness'] = brightness
            analysis['contrast'] = contrast
            
            if brightness < self.brightness_range[0]:
                analysis['rejection_reason'] = 'too_dark'
                self.stats['dark_rejected'] += 1
                return False, analysis
            
            if brightness > self.brightness_range[1]:
                analysis['rejection_reason'] = 'too_bright'
                self.stats['bright_rejected'] += 1
                return False, analysis
            
            if contrast < 20:  # Very low contrast
                analysis['rejection_reason'] = 'low_contrast'
                self.stats['low_contrast_rejected'] += 1
                return False, analysis
        
        # 3. Duplicate Detection (on downsampled)
        if check_duplicate:
            is_dup, similarity = self._check_duplicate(small_frame, small_gray)
            analysis['is_duplicate'] = is_dup
            analysis['duplicate_similarity'] = similarity
            
            if is_dup:
                analysis['rejection_reason'] = 'duplicate'
                self.stats['duplicate_rejected'] += 1
                return False, analysis
        
        # Calculate overall quality score (0-100)
        analysis['quality_score'] = self._calculate_quality_score(analysis)
        
        return True, analysis
    
    def _calculate_blur_score(self, gray: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance
        Higher score = sharper image
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    
    def _analyze_lighting(self, gray: np.ndarray) -> Tuple[int, float]:
        """
        Analyze lighting conditions
        Returns: (mean_brightness, contrast)
        """
        # Mean brightness
        brightness = int(np.mean(gray))
        
        # Contrast (standard deviation)
        contrast = float(np.std(gray))
        
        return brightness, contrast
    
    def _check_duplicate(self, frame: np.ndarray, gray: np.ndarray) -> Tuple[bool, float]:
        """
        Check if frame is duplicate of recent frames
        Uses perceptual hashing + histogram comparison
        """
        # Calculate perceptual hash
        frame_hash = self._perceptual_hash(gray)
        
        # Calculate color histogram
        hist = self._calculate_histogram(frame)
        
        max_similarity = 0.0
        
        # Compare with recent frames
        for old_hash, old_hist in zip(self.frame_hashes, self.frame_histograms):
            # Hash similarity
            hash_sim = self._hash_similarity(frame_hash, old_hash)
            
            # Histogram similarity
            hist_sim = cv2.compareHist(hist, old_hist, cv2.HISTCMP_CORREL)
            
            # Combined similarity
            similarity = (hash_sim * 0.6 + hist_sim * 0.4)
            max_similarity = max(max_similarity, similarity)
            
            if similarity > self.duplicate_threshold:
                return True, similarity
        
        # Add to history (only if not duplicate)
        self.frame_hashes.append(frame_hash)
        self.frame_histograms.append(hist)
        
        return False, max_similarity
    
    def _perceptual_hash(self, gray: np.ndarray, hash_size: int = 16) -> str:
        """
        Calculate perceptual hash (pHash) of image
        """
        # Resize to hash_size
        resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        
        # Apply DCT
        dct = cv2.dct(np.float32(resized))
        
        # Use top-left 8x8 (low frequencies)
        dct_low = dct[:8, :8]
        
        # Calculate median
        median = np.median(dct_low)
        
        # Create hash
        hash_bits = (dct_low > median).flatten()
        hash_str = ''.join(['1' if b else '0' for b in hash_bits])
        
        return hash_str
    
    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate Hamming distance based similarity between hashes
        """
        if len(hash1) != len(hash2):
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculate color histogram for comparison
        """
        # Convert to HSV for better color comparison
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        return hist
    
    def _calculate_quality_score(self, analysis: Dict) -> float:
        """
        Calculate overall quality score (0-100)
        """
        score = 50.0  # Base score
        
        # Blur contribution (max +30)
        blur_norm = min(analysis['blur_score'] / 500.0, 1.0)
        score += blur_norm * 30
        
        # Brightness contribution (max +10)
        brightness = analysis['brightness']
        if 80 <= brightness <= 180:
            score += 10
        elif 60 <= brightness <= 200:
            score += 5
        
        # Contrast contribution (max +10)
        contrast = analysis['contrast']
        if contrast > 50:
            score += 10
        elif contrast > 30:
            score += 5
        
        return min(100.0, max(0.0, score))
    
    def clear_history(self):
        """Clear frame history (call between videos)"""
        self.frame_hashes.clear()
        self.frame_histograms.clear()
    
    def get_stats(self) -> Dict:
        """Get analysis statistics"""
        total_rejected = (
            self.stats['blur_rejected'] + 
            self.stats['dark_rejected'] + 
            self.stats['bright_rejected'] +
            self.stats['duplicate_rejected'] +
            self.stats['low_contrast_rejected']
        )
        
        return {
            **self.stats,
            'total_rejected': total_rejected,
            'acceptance_rate': (
                (self.stats['analyzed'] - total_rejected) / self.stats['analyzed'] * 100
                if self.stats['analyzed'] > 0 else 0
            )
        }
    
    def print_stats(self):
        """Print analysis statistics"""
        stats = self.get_stats()
        print("\n📊 Quality Analysis Statistics:")
        print(f"   Total analyzed: {stats['analyzed']}")
        print(f"   Blur rejected: {stats['blur_rejected']}")
        print(f"   Dark rejected: {stats['dark_rejected']}")
        print(f"   Bright rejected: {stats['bright_rejected']}")
        print(f"   Low contrast rejected: {stats['low_contrast_rejected']}")
        print(f"   Duplicates rejected: {stats['duplicate_rejected']}")
        print(f"   Acceptance rate: {stats['acceptance_rate']:.1f}%")


class SceneChangeDetector:
    """
    Detects scene changes for smart frame interval
    Instead of fixed intervals, extract frames at scene changes
    """
    
    def __init__(self, threshold: float = 30.0, min_scene_frames: int = 15):
        """
        Args:
            threshold: Histogram difference threshold for scene change
            min_scene_frames: Minimum frames between scene changes
        """
        self.threshold = threshold
        self.min_scene_frames = min_scene_frames
        
        self.prev_hist = None
        self.frames_since_change = 0
        self.scene_count = 0
    
    def is_scene_change(self, frame: np.ndarray) -> bool:
        """
        Check if current frame is a scene change
        """
        # Calculate histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        self.frames_since_change += 1
        
        if self.prev_hist is None:
            self.prev_hist = hist
            self.scene_count = 1
            return True
        
        # Compare histograms
        diff = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CHISQR)
        
        is_change = (
            diff > self.threshold and 
            self.frames_since_change >= self.min_scene_frames
        )
        
        if is_change:
            self.prev_hist = hist
            self.frames_since_change = 0
            self.scene_count += 1
            return True
        
        # Update reference periodically
        if self.frames_since_change > self.min_scene_frames * 3:
            self.prev_hist = hist
        
        return False
    
    def reset(self):
        """Reset for new video"""
        self.prev_hist = None
        self.frames_since_change = 0
        self.scene_count = 0
