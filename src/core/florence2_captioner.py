"""
Florence-2 Dense Captioning Module for LoRA-Harvester.

Provides spatially-aware, detailed natural-language captions using
Microsoft's Florence-2 model. Replaces BLIP with higher quality
descriptions and optional region-level detail.
"""

import logging
import threading
import torch
import cv2
import numpy as np
from typing import Union, Optional

logger = logging.getLogger(__name__)

try:
    from PIL import Image as _PIL_Image
except ImportError:
    _PIL_Image = None


class Florence2Captioner:
    """
    Florence-2 based dense captioner.

    Supported tasks (passed as *task* kwarg):
      - '<CAPTION>'          — short one-line caption
      - '<DETAILED_CAPTION>' — paragraph-level description
      - '<MORE_DETAILED_CAPTION>' — exhaustive, dense description
    """

    MODEL_IDS = {
        'florence-2-base': 'microsoft/Florence-2-base',
        'florence-2-large': 'microsoft/Florence-2-large',
    }

    def __init__(self,
                 model_type: str = 'florence-2-base',
                 device: str = None):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.processor = None
        self.model = None
        self._loaded = False
        self._load_lock = threading.Lock()

        logger.info(
            "Florence-2 Captioner init (lazy) — model=%s device=%s",
            model_type, self.device,
        )

    # ── Lazy load ───────────────────────────────────────────────

    def _load_model(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            logger.info("Loading Florence-2 model...")
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM

                repo = self.MODEL_IDS.get(self.model_type, self.model_type)
                from src.core.model_paths import FLORENCE2_DIR, ensure_dirs
                ensure_dirs()
                _f2_cache = str(FLORENCE2_DIR)
                self.processor = AutoProcessor.from_pretrained(
                    repo, cache_dir=_f2_cache, trust_remote_code=True,
                )
                dtype = torch.float16 if self.device == 'cuda' else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    repo, cache_dir=_f2_cache,
                    torch_dtype=dtype, trust_remote_code=True,
                )
                self.model.to(self.device)
                self.model.eval()
                self._loaded = True
                logger.info("Florence-2 loaded successfully")
            except ImportError:
                raise ImportError(
                    "transformers>=4.38.0 required for Florence-2: "
                    "pip install transformers"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Florence-2: {e}")

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Generation ──────────────────────────────────────────────

    def generate(self,
                 image: Union[np.ndarray, object],
                 task: str = '<DETAILED_CAPTION>',
                 max_new_tokens: int = 256) -> str:
        """
        Generate a caption for *image*.

        Args:
            image: OpenCV BGR ndarray or PIL Image.
            task: Florence-2 task prompt.
            max_new_tokens: Generation length limit.

        Returns:
            Generated caption string.
        """
        if not self._loaded:
            self._load_model()

        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_image = _PIL_Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image

        inputs = self.processor(
            text=task, images=pil_image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True,
            )

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        # Florence-2 wraps the answer in task tags; extract it.
        parsed = self.processor.post_process_generation(
            text, task=task, image_size=pil_image.size,
        )
        # parsed is a dict like {'<DETAILED_CAPTION>': 'the caption text'}
        caption = parsed.get(task, text)
        if isinstance(caption, dict):
            caption = str(caption)

        del inputs, generated_ids
        return caption.strip()
