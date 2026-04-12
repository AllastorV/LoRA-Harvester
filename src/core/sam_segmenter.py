"""
SAM 2 (Segment Anything Model 2) integration for LoRA-Harvester.

Masks out unwanted backgrounds so the training data focuses on the
target character/subject. Given a bounding box from the detector,
SAM 2 produces a pixel-level mask for precise isolation.
"""

import logging
import threading
import numpy as np
import cv2
import torch
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class SAMSegmenter:
    """
    Segment Anything Model 2 wrapper.

    Uses a bounding box prompt (from YOLO) to produce a binary mask
    of the subject. The mask can then be used to:
      - Replace the background with white/transparent
      - Crop tightly around the subject contour
    """

    MODEL_IDS = {
        'sam2-tiny': 'facebook/sam2-hiera-tiny',
        'sam2-small': 'facebook/sam2-hiera-small',
        'sam2-base-plus': 'facebook/sam2-hiera-base-plus',
        'sam2-large': 'facebook/sam2-hiera-large',
    }

    def __init__(self,
                 model_type: str = 'sam2-tiny',
                 device: str = None):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.predictor = None
        self._loaded = False
        self._load_lock = threading.Lock()

        logger.info(
            "SAM2 Segmenter init (lazy) — model=%s device=%s",
            model_type, self.device,
        )

    # ── Lazy load ───────────────────────────────────────────────

    def _load_model(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            logger.info("Loading SAM 2 model...")
            try:
                from sam2.build_sam import build_sam2_hf
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                repo = self.MODEL_IDS.get(self.model_type, self.model_type)
                model = build_sam2_hf(repo, device=self.device)
                self.predictor = SAM2ImagePredictor(model)
                self._loaded = True
                logger.info("SAM 2 loaded successfully")
            except ImportError:
                raise ImportError(
                    "sam2 package required: pip install sam2"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM 2: {e}")

    def cleanup(self):
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Segmentation ────────────────────────────────────────────

    def segment_bbox(self,
                     image: np.ndarray,
                     bbox: List[int]) -> np.ndarray:
        """
        Segment the subject inside *bbox* and return a binary mask.

        Args:
            image: BGR OpenCV image.
            bbox: [x1, y1, x2, y2] bounding box.

        Returns:
            Binary mask (H, W) with 255 for foreground, 0 for background.
        """
        if not self._loaded:
            self._load_model()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        box_np = np.array(bbox, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            box=box_np,
            multimask_output=True,
        )
        # Pick the mask with the highest predicted IoU.
        best_idx = int(np.argmax(scores))
        mask = (masks[best_idx] * 255).astype(np.uint8)
        return mask

    def remove_background(self,
                          image: np.ndarray,
                          bbox: List[int],
                          bg_color: Tuple[int, int, int] = (255, 255, 255)
                          ) -> np.ndarray:
        """
        Replace the background with *bg_color*, keeping only the
        subject inside the bbox.

        Args:
            image: BGR OpenCV image.
            bbox: [x1, y1, x2, y2].
            bg_color: BGR fill color for background (default white).

        Returns:
            New BGR image with background replaced.
        """
        mask = self.segment_bbox(image, bbox)
        bg = np.full_like(image, bg_color, dtype=np.uint8)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (image.astype(np.float32) * mask3
                  + bg.astype(np.float32) * (1 - mask3))
        return result.astype(np.uint8)
