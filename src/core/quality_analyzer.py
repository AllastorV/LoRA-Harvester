"""
Quality Analyzer Module for LoRA-Harvester
Analyzes frame quality: blur, lighting, duplicates, composition
"""

import cv2
import logging
import numpy as np
from typing import Tuple, Dict
from collections import deque

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """
    Comprehensive frame quality analysis:
    - Blur detection (Laplacian + Sobel combined for motion-blur robustness)
    - Noise/grain detection (dark-scene grainy frames)
    - Lighting analysis (histogram)
    - Duplicate detection (perceptual hash)
    - Composition scoring
    """

    def __init__(self,
                 blur_threshold: float = 100.0,
                 brightness_range: Tuple[int, int] = (40, 220),
                 duplicate_threshold: float = 0.92,
                 history_size: int = 50,
                 noise_threshold: float = 12.0,
                 check_duplicates: bool = True):
        """
        Initialize quality analyzer

        Args:
            blur_threshold: Minimum sharpness score (higher = sharper required)
            brightness_range: Acceptable mean brightness range (0-255)
            duplicate_threshold: Similarity threshold for duplicate detection (0-1)
            history_size: Number of recent frames to check for duplicates
            noise_threshold: Maximum noise std-dev allowed on a uniform patch (higher = more tolerant)
        """
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.duplicate_threshold = duplicate_threshold
        self.history_size = history_size
        self.noise_threshold = noise_threshold
        self.check_duplicates = check_duplicates

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
            'low_contrast_rejected': 0,
            'noise_rejected': 0,
        }

        logger.info(
            "Quality Analyzer initialized - blur=%.1f brightness=%s "
            "dup_threshold=%.0f%% noise_threshold=%.1f",
            blur_threshold, brightness_range,
            duplicate_threshold * 100, noise_threshold,
        )

    # ─────────────────────────── Public API ───────────────────────────

    def check_frame_quality(self, frame: np.ndarray,
                            check_blur: bool = True,
                            check_lighting: bool = True,
                            check_duplicate: bool = True) -> Tuple[bool, Dict]:
        """
        Wrapper for analyze_frame() — compatibility method.
        Returns: (is_quality_ok, quality_info)
        """
        return self.analyze_frame(frame, check_blur, check_lighting, check_duplicate)

    def analyze_frame(self, frame: np.ndarray,
                      check_blur: bool = True,
                      check_lighting: bool = True,
                      check_duplicate: bool = True) -> Tuple[bool, Dict]:
        """
        Analyze frame quality.

        Downsamples once and reuses the result for all checks to avoid
        redundant resizing.

        Args:
            frame: Input frame (BGR)
            check_blur: Enable blur + noise detection
            check_lighting: Enable lighting analysis
            check_duplicate: Enable duplicate detection

        Returns:
            Tuple of (is_quality_ok, analysis_details)
        """
        self.stats['analyzed'] += 1

        analysis: Dict = {
            'blur_score': 0.0,
            'noise_score': 0.0,
            'brightness': 0,
            'contrast': 0.0,
            'is_duplicate': False,
            'duplicate_similarity': 0.0,
            'quality_score': 0.0,
            'rejection_reason': None,
        }

        # ── Downsample once, reuse for all checks ──────────────────────
        h, w = frame.shape[:2]
        if w > 512:
            scale = 512 / w
            small_frame = cv2.resize(frame, (512, int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
        else:
            small_frame = frame

        small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # 1. Blur + Noise detection
        if check_blur:
            blur_score = self._calculate_blur_score(small_gray)
            analysis['blur_score'] = blur_score

            if blur_score < self.blur_threshold:
                analysis['rejection_reason'] = 'blur'
                self.stats['blur_rejected'] += 1
                return False, analysis

            noise_score = self._calculate_noise_score(small_gray)
            analysis['noise_score'] = noise_score

            if noise_score > self.noise_threshold:
                analysis['rejection_reason'] = 'noise'
                self.stats['noise_rejected'] += 1
                return False, analysis

        # 2. Lighting analysis
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

            if contrast < 20:
                analysis['rejection_reason'] = 'low_contrast'
                self.stats['low_contrast_rejected'] += 1
                return False, analysis

        # 3. Duplicate detection
        if check_duplicate and self.check_duplicates:
            is_dup, similarity = self._check_duplicate(small_frame, small_gray)
            analysis['is_duplicate'] = is_dup
            analysis['duplicate_similarity'] = similarity

            if is_dup:
                analysis['rejection_reason'] = 'duplicate'
                self.stats['duplicate_rejected'] += 1
                return False, analysis

        analysis['quality_score'] = self._calculate_quality_score(analysis)
        return True, analysis

    # ─────────────────────── Blur / Sharpness ─────────────────────────

    def _calculate_blur_score(self, gray: np.ndarray) -> float:
        """
        Combined sharpness score: Laplacian variance + Sobel magnitude.

        - Laplacian variance detects overall blurriness.
        - Sobel magnitude detects directional (motion) blur that Laplacian
          alone can underestimate for horizontally/vertically blurred frames.

        The two scores are blended 60/40 and normalised so the result is
        comparable to the pure-Laplacian score used previously.
        """
        # Laplacian variance (classical method)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Sobel magnitude mean — captures edge strength in both directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = float(np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2)))
        # Normalise Sobel to roughly the same scale as Laplacian variance
        sobel_norm = sobel_mag ** 2

        return laplacian_var * 0.6 + sobel_norm * 0.4

    # ─────────────────────────── Noise ────────────────────────────────

    def _calculate_noise_score(self, gray: np.ndarray) -> float:
        """
        Estimate per-pixel noise level using local patch variance.

        Strategy: divide the image into a coarse grid and take the
        standard deviation of the darkest (flattest) patches.  A high
        score on a near-uniform region indicates sensor noise / film grain
        rather than real texture.

        Returns:
            Mean std-dev of the quietest 20 % of patches (lower = cleaner).
        """
        patch_h, patch_w = 16, 16
        h, w = gray.shape[:2]

        stds = []
        for y in range(0, h - patch_h, patch_h):
            for x in range(0, w - patch_w, patch_w):
                patch = gray[y:y + patch_h, x:x + patch_w]
                stds.append(float(np.std(patch)))

        if not stds:
            return 0.0

        stds.sort()
        # Take the quietest 20% of patches
        n_quiet = max(1, len(stds) // 5)
        return float(np.mean(stds[:n_quiet]))

    # ─────────────────────────── Lighting ─────────────────────────────

    def _analyze_lighting(self, gray: np.ndarray) -> Tuple[int, float]:
        """
        Analyze lighting conditions.
        Returns: (mean_brightness, contrast)
        """
        brightness = int(np.mean(gray))
        contrast = float(np.std(gray))
        return brightness, contrast

    # ─────────────────────────── Duplicates ───────────────────────────

    def _check_duplicate(self, frame: np.ndarray,
                         gray: np.ndarray) -> Tuple[bool, float]:
        """
        Check if frame is duplicate of recent frames.
        Uses perceptual hashing + histogram comparison.
        """
        frame_hash = self._perceptual_hash(gray)
        hist = self._calculate_histogram(frame)

        max_similarity = 0.0

        for old_hash, old_hist in zip(self.frame_hashes, self.frame_histograms):
            hash_sim = self._hash_similarity(frame_hash, old_hash)
            # L2-distance for concatenated 1-D histogram → convert to similarity
            l2 = float(np.linalg.norm(hist - old_hist))
            hist_sim = max(0.0, 1.0 - l2)  # 0 distance → 1.0 similarity
            similarity = hash_sim * 0.6 + hist_sim * 0.4
            max_similarity = max(max_similarity, similarity)

            if similarity > self.duplicate_threshold:
                return True, similarity

        self.frame_hashes.append(frame_hash)
        self.frame_histograms.append(hist)
        return False, max_similarity

    def _perceptual_hash(self, gray: np.ndarray, hash_size: int = 16) -> np.ndarray:
        """
        Calculate perceptual hash (pHash) of image.
        Returns a boolean numpy array for fast vectorised comparison.
        """
        resized = cv2.resize(gray, (hash_size, hash_size),
                             interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        return (dct_low > median).flatten()

    def _hash_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """
        Hamming distance based similarity using vectorised numpy comparison.
        """
        if hash1.shape != hash2.shape:
            return 0.0
        return float(np.mean(hash1 == hash2))

    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate H+S+V colour histogram for comparison.

        Uses separate 1-D histograms for H, S, and V channels concatenated
        into a single vector.  This captures brightness differences (V)
        which the old 2-D H×S histogram silently ignored.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [30], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [30], [0, 256])
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        cv2.normalize(v_hist, v_hist)
        return np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])

    # ──────────────────────── Quality Score ───────────────────────────

    def _calculate_quality_score(self, analysis: Dict) -> float:
        """
        Calculate overall quality score (0–100).

        Components:
          - Sharpness  (max +35): log-scaled blur score
          - Brightness (max +15): penalty for over/under-exposed frames
          - Contrast   (max +10): reward for rich tonal range
          - Noise      (max -20): penalty for noisy/grainy frames
        """
        score = 40.0  # baseline

        # Sharpness contribution (log scale to avoid extreme outliers)
        blur = analysis.get('blur_score', 0.0)
        blur_contrib = min(35.0, 35.0 * np.log1p(blur) / np.log1p(500.0))
        score += blur_contrib

        # Brightness contribution
        brightness = analysis.get('brightness', 128)
        if 80 <= brightness <= 180:
            score += 15
        elif 60 <= brightness <= 200:
            score += 8

        # Contrast contribution
        contrast = analysis.get('contrast', 0.0)
        if contrast > 50:
            score += 10
        elif contrast > 30:
            score += 5

        # Noise penalty
        noise = analysis.get('noise_score', 0.0)
        noise_penalty = min(20.0, noise * 2.0)
        score -= noise_penalty

        return float(np.clip(score, 0.0, 100.0))

    # ─────────────────────── History / Stats ──────────────────────────

    def clear_history(self):
        """Clear frame history (call between videos)."""
        self.frame_hashes.clear()
        self.frame_histograms.clear()

    def get_stats(self) -> Dict:
        """Get analysis statistics."""
        total_rejected = (
            self.stats['blur_rejected']
            + self.stats['dark_rejected']
            + self.stats['bright_rejected']
            + self.stats['duplicate_rejected']
            + self.stats['low_contrast_rejected']
            + self.stats['noise_rejected']
        )
        analyzed = self.stats['analyzed']
        return {
            **self.stats,
            'total_rejected': total_rejected,
            'acceptance_rate': (
                (analyzed - total_rejected) / analyzed * 100
                if analyzed > 0 else 0.0
            ),
        }

    def print_stats(self):
        """Print analysis statistics."""
        stats = self.get_stats()
        logger.info(
            "Quality stats — analyzed=%d blur=%d dark=%d bright=%d "
            "low_contrast=%d noise=%d dup=%d acceptance=%.1f%%",
            stats['analyzed'], stats['blur_rejected'], stats['dark_rejected'],
            stats['bright_rejected'], stats['low_contrast_rejected'],
            stats['noise_rejected'], stats['duplicate_rejected'],
            stats['acceptance_rate'],
        )


class SceneChangeDetector:
    """
    Detects scene changes for smart frame interval.
    Instead of fixed intervals, extract frames at scene changes.
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
        """Check if current frame is a scene change.

        Uses a combined H+S+V histogram so that brightness-only changes
        (e.g. fade to black / white) are also detected.  Previously only
        H and S channels were used, which caused missed detections when
        two frames had the same hue/saturation but very different values.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # H: 50 bins [0-180], S: 30 bins [0-256], V: 30 bins [0-256]
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [30], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [30], [0, 256])
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        cv2.normalize(v_hist, v_hist)
        # Concatenate into single descriptor
        hist = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])

        self.frames_since_change += 1

        if self.prev_hist is None:
            self.prev_hist = hist
            self.scene_count = 1
            return True

        # L2 distance (works better than chi-square for concatenated 1-D hists)
        diff = float(np.linalg.norm(hist - self.prev_hist))
        is_change = (
            diff > self.threshold
            and self.frames_since_change >= self.min_scene_frames
        )

        if is_change:
            self.prev_hist = hist
            self.frames_since_change = 0
            self.scene_count += 1
            return True

        # Slowly update reference to handle gradual lighting changes
        if self.frames_since_change > self.min_scene_frames * 3:
            self.prev_hist = hist

        return False

    def reset(self):
        """Reset for new video."""
        self.prev_hist = None
        self.frames_since_change = 0
        self.scene_count = 0
