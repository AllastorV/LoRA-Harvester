"""Local SFW/NSFW classification with bounded, ordered GPU inference batches.

The Falconsai model, image processor and 0.70 decision rule are retained.
No frame skipping, approximate-result reuse or reduced-precision conversion is
used. CPU defaults to single-image inference. A CUDA OOM reduces the batch;
a persistent single-image CUDA OOM moves the same model to CPU, with a warning.

``wd14_tags`` and ``heuristic`` remain explicit legacy backends. Neither is a
silent fallback for standalone image classification when Falconsai fails.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_WD14_RATING_MAP = {
    'rating:general': ('sfw', 1.00),
    'rating:sensitive': ('uncertain', 0.60),
    'rating:questionable': ('nsfw', 0.75),
    'rating:explicit': ('nsfw', 1.00),
}
_UNCERTAIN = ('uncertain', 0.5)


class NsfwDetector:
    """One model per processing job; each result belongs to one final crop.

    ``batch_size`` is a maximum (1..32), not an assumption about free VRAM.
    Instances are owned by one video worker and are not shared across threads.
    """
    BACKENDS = ('falconsai', 'wd14_tags', 'heuristic')

    def __init__(self, backend: str = 'auto', threshold: float = 0.70,
                 device: str = 'cpu', batch_size: int = 8) -> None:
        if backend not in ('auto', *self.BACKENDS):
            raise ValueError(f'Unknown NSFW backend: {backend}')
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError('NSFW batch_size must be an integer in 1..32.')
        if not math.isfinite(float(threshold)):
            raise ValueError('NSFW threshold must be finite.')
        self.backend = backend
        self.threshold = max(0.5, min(0.99, float(threshold)))
        self.device = device
        self.batch_size = batch_size
        self.actual_device = 'uninitialized'
        self.last_error = ''
        self._effective_batch_size = 1
        self._pipe = None
        self._active_backend: Optional[str] = None
        self._checked = False
        self._metrics = {
            'pipeline_calls': 0, 'model_images': 0, 'failed_images': 0,
            'oom_retries': 0, 'cpu_fallbacks': 0, 'largest_batch': 0,
            'load_seconds': 0.0, 'preprocess_seconds': 0.0,
            'inference_seconds': 0.0,
        }

    @property
    def effective_batch_size(self) -> int:
        """Safe current batch ceiling. Never initializes a model from the UI."""
        return self._effective_batch_size

    def metrics(self) -> dict:
        return dict(self._metrics, backend=self._active_backend,
                    device=self.actual_device, batch_size=self.effective_batch_size)

    def is_available(self) -> bool:
        self._lazy_init()
        return self._active_backend is not None

    def classify(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        return self.classify_batch([image_bgr])[0]

    def classify_batch(self, images_bgr: Sequence[np.ndarray]) -> list[Tuple[str, float]]:
        """Ordered, cardinality-preserving output, including invalid images.

        Only at most ``effective_batch_size`` RGB copies exist at a time.
        Invalid inputs/model outputs are uncertain, never implicitly SFW.
        """
        if len(images_bgr) == 0:
            return []
        self._lazy_init()
        if self._active_backend == 'heuristic':
            return [self._classify_heuristic(image) for image in images_bgr]
        if self._active_backend != 'falconsai':
            self._metrics['failed_images'] += len(images_bgr)
            return [_UNCERTAIN for _ in images_bgr]

        import cv2
        from PIL import Image
        output = [_UNCERTAIN for _ in images_bgr]
        start = 0
        while start < len(images_bgr):
            stop = min(len(images_bgr), start + self.effective_batch_size)
            pil_images, indices = [], []
            tic = time.perf_counter()
            for index in range(start, stop):
                image = images_bgr[index]
                try:
                    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
                        raise ValueError('Expected a nonempty uint8 BGR image.')
                    pil_images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                    indices.append(index)
                except Exception as exc:
                    self._metrics['failed_images'] += 1
                    logger.warning('SFW/NSFW invalid image at batch index %s: %s', index, exc)
            self._metrics['preprocess_seconds'] += time.perf_counter() - tic
            try:
                if pil_images:
                    predictions = self._infer_pil(pil_images)
                    for index, result in zip(indices, predictions):
                        output[index] = result
            finally:
                for image in pil_images:
                    image.close()
            start = stop
        return output

    def classify_from_wd14_tags(self, tags: dict) -> Tuple[str, float]:
        """Explicit legacy tag route; never inferred from a caption string."""
        best_label, best_conf = _UNCERTAIN
        for tag, (label, conf) in _WD14_RATING_MAP.items():
            if tag in tags:
                try:
                    tag_conf = float(tags[tag])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(tag_conf) or not 0 <= tag_conf <= 1:
                    continue
                weighted = conf * max(0.5, tag_conf)
                if weighted > best_conf:
                    best_label, best_conf = label, weighted
        return best_label, best_conf

    def _lazy_init(self) -> None:
        if self._checked:
            return
        self._checked = True
        # There is no WD14 score at the video saving call site. Falling back
        # to wd14_tags there falsely announces a working classifier.
        if self.backend in ('auto', 'falconsai'):
            if self._try_init_falconsai():
                self._active_backend = 'falconsai'
        else:
            self._active_backend = self.backend
            self.actual_device = 'cpu'
            logger.info('NsfwDetector: explicit %s backend', self.backend)
        if self._active_backend is None:
            logger.error('SFW/NSFW classifier unavailable: %s', self.last_error)

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        return 'out of memory' in text or 'cuda_error_out_of_memory' in text

    def _try_init_falconsai(self) -> bool:
        tic = time.perf_counter()
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            wants_cuda = self.device == 'auto' or str(self.device).startswith('cuda')
            dev = 0 if wants_cuda and torch.cuda.is_available() else -1
            if wants_cuda and dev == -1:
                logger.warning('SFW/NSFW: CUDA unavailable; using CPU (batch 1).')
            try:
                self._pipe = hf_pipeline(
                    'image-classification', model='Falconsai/nsfw_image_detection',
                    device=dev,
                )
            except Exception as exc:
                if dev == -1 or not self._is_oom(exc):
                    raise
                self._clear_cuda_cache()
                logger.warning('SFW/NSFW GPU model load OOM; loading the same model on CPU.')
                self._metrics['cpu_fallbacks'] += 1
                dev = -1
                self._pipe = hf_pipeline(
                    'image-classification', model='Falconsai/nsfw_image_detection', device=-1,
                )
            # No half()/autocast/compile, custom resizing, or model replacement.
            self.actual_device = str(getattr(self._pipe, 'device', 'cuda:0' if dev == 0 else 'cpu'))
            self._effective_batch_size = self.batch_size if self.actual_device.startswith('cuda') else 1
            logger.info('SFW/NSFW: %s, batch <= %d', self.actual_device, self.effective_batch_size)
            return True
        except Exception as exc:
            self._pipe = None
            self.last_error = str(exc)
            return False
        finally:
            self._metrics['load_seconds'] += time.perf_counter() - tic

    def _infer_pil(self, images: list) -> list[Tuple[str, float]]:
        """Recover CUDA memory pressure without replaying any disk writes."""
        if not images:
            return []
        # A previous sibling may have reduced the safe batch ceiling.
        if len(images) > self.effective_batch_size:
            size = self.effective_batch_size
            results = []
            for start in range(0, len(images), size):
                results.extend(self._infer_pil(images[start:start + size]))
            return results
        self._metrics['pipeline_calls'] += 1
        tic = time.perf_counter()
        error = None
        try:
            # num_workers=0 is deliberate: never spawn a Windows process from
            # the Qt video worker. Batch collation still uses one GPU forward.
            if len(images) == 1:
                # Preserve the low-overhead original call for CPU and tails;
                # do not build a one-item DataLoader for every CPU image.
                rows = [self._pipe(images[0], top_k=2)]
            else:
                rows = self._pipe(images, batch_size=len(images), num_workers=0, top_k=2)
            if not isinstance(rows, list) or len(rows) != len(images):
                raise ValueError('SFW/NSFW model response count mismatch.')
        except Exception as exc:
            error = exc
        finally:
            self._metrics['inference_seconds'] += time.perf_counter() - tic
        if error is not None:
            is_oom = self._is_oom(error)
            message = str(error)
            # Drop exception tracebacks before retry to release failed tensors.
            error.__traceback__ = None
            del error
            if is_oom and self.actual_device.startswith('cuda'):
                self._metrics['oom_retries'] += 1
                self._clear_cuda_cache()
                if len(images) > 1:
                    self._effective_batch_size = max(1, len(images) // 2)
                    logger.warning('SFW/NSFW CUDA OOM; batch reduced to %d.', self.effective_batch_size)
                    return self._infer_pil(images)
                if self._move_to_cpu():
                    return self._infer_pil(images)
            self.last_error = message
            self._metrics['failed_images'] += len(images)
            logger.warning('SFW/NSFW inference failed; marking %d crops uncertain: %s', len(images), message)
            return [_UNCERTAIN for _ in images]
        self._metrics['model_images'] += len(images)
        self._metrics['largest_batch'] = max(self._metrics['largest_batch'], len(images))
        return [self._parse_prediction(row) for row in rows]

    def _parse_prediction(self, row: object) -> Tuple[str, float]:
        try:
            if not isinstance(row, list) or not row:
                raise ValueError('Empty or invalid prediction list.')
            # Same top-ranked recognized-label decision as the original code.
            # Validate all values before letting any result route an image.
            for item in row:
                if not isinstance(item, dict) or not isinstance(item.get('label'), str):
                    raise ValueError('Invalid prediction item.')
                score = float(item['score'])
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError('Non-finite/out-of-range probability.')
            for item in row:
                label, score = item['label'].lower(), float(item['score'])
                if label in ('nsfw', 'explicit', 'unsafe'):
                    return self._decision(score)
                if label in ('normal', 'safe', 'sfw', 'general'):
                    return self._decision(1.0 - score)
            raise ValueError('Unknown classifier labels.')
        except (TypeError, ValueError, KeyError) as exc:
            self._metrics['failed_images'] += 1
            logger.warning('SFW/NSFW response rejected: %s', exc)
            return _UNCERTAIN

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _move_to_cpu(self) -> bool:
        try:
            import torch
            self._pipe.model.to('cpu')
            self._pipe.device = torch.device('cpu')
            self.actual_device = 'cpu'
            self._effective_batch_size = 1
            self._metrics['cpu_fallbacks'] += 1
            self._clear_cuda_cache()
            logger.warning('SFW/NSFW: GPU memory insufficient at batch 1; same model now runs on CPU.')
            return True
        except Exception as exc:
            logger.warning('SFW/NSFW CPU recovery failed: %s', exc)
            return False

    def _classify_falconsai(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        """Compatibility for older callers."""
        return self.classify_batch([image_bgr])[0]

    def cleanup(self) -> None:
        """Explicit release before the optional clothing model post-pass."""
        self._pipe = None
        self._active_backend = None
        self._checked = False
        self._effective_batch_size = 1
        self.actual_device = 'uninitialized'
        self._clear_cuda_cache()

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
