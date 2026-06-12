"""
NsfwDetector — SFW/NSFW image classifier for LoRA-Harvester.

Supports three backends in priority order:
  1. 'falconsai' — Falconsai/nsfw_image_detection (ViT, ~350 MB, most accurate)
      pip install transformers
  2. 'wd14_tags' — reuse WD14 rating tags already produced by the captioner
      (zero extra cost when WD14 is already enabled)
  3. 'heuristic' — fast skin-tone heuristic (no extra deps, lower accuracy)

Usage:
    det = NsfwDetector(backend='auto', threshold=0.70)
    label, conf = det.classify(bgr_frame)
    # label ∈ {'sfw', 'nsfw', 'uncertain'}
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# WD14 Danbooru rating tags → SFW/NSFW mapping
_WD14_RATING_MAP = {
    'rating:general':       ('sfw',       1.00),
    'rating:sensitive':     ('uncertain', 0.60),
    'rating:questionable':  ('nsfw',      0.75),
    'rating:explicit':      ('nsfw',      1.00),
}


class NsfwDetector:
    """
    Classify an image as 'sfw', 'nsfw', or 'uncertain'.

    Parameters
    ----------
    backend : str
        One of 'auto', 'falconsai', 'wd14_tags', 'heuristic'.
        'auto' tries each backend in order until one is available.
    threshold : float
        Confidence threshold for hard sfw/nsfw decision.
        Predictions between (1-threshold) and threshold → 'uncertain'.
    device : str
        'cuda' or 'cpu' (for Falconsai backend only).
    """

    BACKENDS = ('falconsai', 'wd14_tags', 'heuristic')

    def __init__(
        self,
        backend: str = 'auto',
        threshold: float = 0.70,
        device: str = 'cpu',
    ) -> None:
        self.backend = backend
        self.threshold = max(0.5, min(0.99, threshold))
        self.device = device

        self._pipe = None          # Falconsai pipeline
        self._active_backend: Optional[str] = None
        self._checked = False

    # ── public API ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if at least one backend is usable."""
        self._lazy_init()
        return self._active_backend is not None

    def classify(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Classify a BGR frame.

        Returns
        -------
        (label, confidence)
            label  : 'sfw' | 'nsfw' | 'uncertain'
            confidence : float 0..1
        """
        self._lazy_init()
        if self._active_backend == 'falconsai':
            return self._classify_falconsai(image_bgr)
        if self._active_backend == 'heuristic':
            return self._classify_heuristic(image_bgr)
        # wd14_tags backend has no standalone classify — caller should use
        # classify_from_wd14_tags() directly.
        return 'uncertain', 0.5

    def classify_from_wd14_tags(self, tags: dict) -> Tuple[str, float]:
        """
        Derive SFW/NSFW from a dict of WD14/Danbooru tags.

        Parameters
        ----------
        tags : dict
            {tag_name: confidence} as returned by AdvancedCaptioner.caption_image()

        Returns
        -------
        (label, confidence)
        """
        best_label, best_conf = 'uncertain', 0.5
        for tag, (label, conf) in _WD14_RATING_MAP.items():
            if tag in tags:
                tag_conf = float(tags[tag])
                weighted = conf * max(0.5, tag_conf)
                if weighted > best_conf:
                    best_label, best_conf = label, weighted
        return best_label, best_conf

    # ── internals ───────────────────────────────────────────────────────────

    def _lazy_init(self) -> None:
        if self._checked:
            return
        self._checked = True

        order = self.BACKENDS if self.backend == 'auto' else [self.backend]
        for b in order:
            if b == 'falconsai' and self._try_init_falconsai():
                self._active_backend = 'falconsai'
                break
            if b == 'wd14_tags':
                # Always available — but classify() falls back to caller using
                # classify_from_wd14_tags() instead.
                self._active_backend = 'wd14_tags'
                logger.info("NsfwDetector: using wd14_tags backend")
                break
            if b == 'heuristic':
                self._active_backend = 'heuristic'
                logger.info("NsfwDetector: using heuristic backend")
                break

        if self._active_backend is None:
            logger.warning("NsfwDetector: no backend available")

    def _try_init_falconsai(self) -> bool:
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            dev = 0 if (self.device == 'cuda' and torch.cuda.is_available()) else -1
            self._pipe = hf_pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=dev,
            )
            logger.info("NsfwDetector: Falconsai/nsfw_image_detection loaded (device=%s)", dev)
            return True
        except Exception as exc:
            logger.debug("Falconsai backend unavailable: %s", exc)
            return False

    def _classify_falconsai(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        try:
            import cv2
            from PIL import Image
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            results = self._pipe(pil)
            # results: [{'label': 'nsfw'|'normal', 'score': float}, ...]
            for r in results:
                lbl = r['label'].lower()
                score = float(r['score'])
                if lbl in ('nsfw', 'explicit', 'unsafe'):
                    raw_nsfw = score
                    break
                if lbl in ('normal', 'safe', 'sfw', 'general'):
                    raw_nsfw = 1.0 - score
                    break
            else:
                raw_nsfw = 0.5

            return self._decision(raw_nsfw)
        except Exception as exc:
            logger.warning("Falconsai classify failed: %s", exc)
            return 'uncertain', 0.5

    def _classify_heuristic(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Fast skin-tone heuristic.  Counts pixels in the YCrCb skin range,
        returns 'nsfw' when the skin fraction exceeds the threshold.
        Rough but zero-dependency.
        """
        try:
            import cv2
            img = image_bgr
            if img is None or img.size == 0:
                return 'uncertain', 0.5
            # Resize to 64×64 for speed
            small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
            # Skin in YCrCb: Cr 133..173, Cb 77..127
            mask = (
                (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) &
                (ycrcb[:, :, 2] >= 77)  & (ycrcb[:, :, 2] <= 127)
            )
            skin_frac = float(mask.sum()) / (64 * 64)
            # Treat skin_frac as a proxy — high skin fraction → likely NSFW
            # This is intentionally conservative: only flags heavy exposure
            raw_nsfw = min(1.0, skin_frac * 2.5)
            return self._decision(raw_nsfw)
        except Exception as exc:
            logger.warning("Heuristic classify failed: %s", exc)
            return 'uncertain', 0.5

    def _decision(self, raw_nsfw: float) -> Tuple[str, float]:
        """Convert a raw NSFW probability to (label, confidence)."""
        t = self.threshold
        uncertain_lo = 1.0 - t
        if raw_nsfw >= t:
            return 'nsfw', raw_nsfw
        if raw_nsfw <= uncertain_lo:
            return 'sfw', 1.0 - raw_nsfw
        return 'uncertain', 0.5
