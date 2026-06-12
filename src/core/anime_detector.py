"""
AnimeDetector — anime-optimised object detector for LoRA-Harvester.

Drop-in replacement for ObjectDetector: same public interface so
UnifiedVideoProcessor can use either without changes.

Backend priority (lazy-loaded on first use):
  1. yolov8_animeface  — Fuyucchi/yolov8_animeface (YOLOv8x6 .pt via ultralytics).
                         Best quality, GPU, works on any Python/numpy that
                         ultralytics supports (incl. 3.13 + numpy 2.x). Default.
  2. imgutils          — deepghs anime models (ONNX; needs onnxruntime + numpy<2)
  3. lbpcascade_animeface (OpenCV Haar, no extra deps, face-only)

Auto mode (AutoDetector):
  Classifies each frame as anime or real-photo using HSV saturation
  heuristic, then routes to the appropriate detector.

Dependencies (all optional — feature gracefully degrades):
  pip install imgutils>=0.4.0   # best quality
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.model_paths import MODELS_DIR, ensure_dirs

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Cascade download
# ──────────────────────────────────────────────────────────────

_CASCADE_URL = (
    "https://raw.githubusercontent.com/nagadomi/"
    "lbpcascade_animeface/master/lbpcascade_animeface.xml"
)
_CASCADE_PATH: Path = MODELS_DIR / "anime" / "lbpcascade_animeface.xml"

# YOLOv8x6 anime-face weights (Fuyucchi/yolov8_animeface). Downloaded by
# scripts/download_models.py / install. ~186 MB. Primary backend.
_YOLO_REPO = "Fuyucchi/yolov8_animeface"
_YOLO_FILE = "yolov8x6_animeface.pt"
_YOLO_PATH: Path = MODELS_DIR / "anime" / _YOLO_FILE


def _ensure_cascade() -> cv2.CascadeClassifier:
    """Download and load lbpcascade_animeface.xml."""
    ensure_dirs()
    _CASCADE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not _CASCADE_PATH.exists():
        logger.info("Downloading lbpcascade_animeface.xml …")
        tmp = _CASCADE_PATH.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(_CASCADE_URL, str(tmp))
            tmp.replace(_CASCADE_PATH)
            logger.info("Cascade downloaded → %s", _CASCADE_PATH)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download anime face cascade: {e}\n"
                f"Place lbpcascade_animeface.xml manually in {_CASCADE_PATH.parent}/"
            ) from e

    cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))
    if cascade.empty():
        _CASCADE_PATH.unlink(missing_ok=True)
        raise RuntimeError("Cascade file corrupt — deleted, will re-download next run.")
    return cascade


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _make_det(x1: int, y1: int, x2: int, y2: int,
              confidence: float = 0.8,
              class_id: int = 0, class_name: str = 'person') -> dict:
    """Build a detection dict matching ObjectDetector's format."""
    return {
        'bbox': [x1, y1, x2, y2],
        'confidence': float(confidence),
        'class_id': class_id,
        'class_name': class_name,
    }


def _empty_detections() -> Dict[str, List[dict]]:
    return {'person': [], 'animal': [], 'object': []}


# ──────────────────────────────────────────────────────────────
# AnimeDetector
# ──────────────────────────────────────────────────────────────

class AnimeDetector:
    """
    Anime-optimised object detector.
    Public interface mirrors ObjectDetector — direct drop-in for
    UnifiedVideoProcessor.
    """

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self._backend: Optional[str] = None    # set after first _ensure_loaded()
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._imgutils_ok: Optional[bool] = None
        self._yolo = None                      # ultralytics YOLO model
        self._yolo_device = None               # 'cuda:0' or 'cpu'

    # ── Public API (same as ObjectDetector) ────────────────────

    def detect(self, frame: np.ndarray) -> Dict[str, List[dict]]:
        """Detect anime characters in a single BGR frame."""
        self._ensure_loaded()
        return self._detect_impl(frame)

    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict[str, List[dict]]]:
        """Detect in a list of frames. YOLO runs the whole batch on the GPU
        at once; other backends fall back to per-frame."""
        if not frames:
            return []
        self._ensure_loaded()
        if self._backend == 'yolo' and self._yolo is not None:
            try:
                results = self._yolo.predict(
                    frames, conf=self.confidence, verbose=False,
                    device=self._yolo_device,
                )
                out = []
                for r in results:
                    det = _empty_detections()
                    persons = []
                    if r.boxes is not None and len(r.boxes):
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2), sc in zip(xyxy, confs):
                            persons.append(_make_det(int(x1), int(y1), int(x2), int(y2), float(sc)))
                    det['person'] = persons
                    out.append(det)
                return out
            except Exception as e:
                logger.debug("YOLO batch detect error: %s — falling back per-frame", e)
        return [self._detect_impl(f) for f in frames]

    def get_primary_subject(
        self, detections: Dict[str, List[dict]]
    ) -> Tuple[Optional[str], Optional[dict]]:
        """Return (category, best_detection) by highest area*confidence across all categories."""
        all_subjects = []
        for category in ('person', 'animal', 'object'):
            for d in detections.get(category, []):
                b = d['bbox']
                score = (b[2] - b[0]) * (b[3] - b[1]) * d['confidence']
                all_subjects.append((category, d, score))
        if not all_subjects:
            return None, None
        all_subjects.sort(key=lambda x: x[2], reverse=True)
        return all_subjects[0][0], all_subjects[0][1]

    def calculate_head_space(self, bbox: List[int], frame_height: int) -> float:
        """Return head-space ratio (top of bbox / frame height)."""
        if frame_height <= 0:
            return 0.0
        return float(bbox[1]) / float(frame_height)

    def cleanup(self):
        """Release resources."""
        self._cascade = None
        self._imgutils_ok = None
        self._yolo = None
        self._yolo_device = None
        self._backend = None

    # ── Backend loading ─────────────────────────────────────────

    def _ensure_loaded(self):
        if self._backend is not None:
            return

        # 1. YOLOv8 anime-face (ultralytics) — best, GPU, runs anywhere
        if self._try_yolo():
            self._backend = 'yolo'
            logger.info("AnimeDetector backend: yolov8x6_animeface (%s)", self._yolo_device)
            return

        # 2. imgutils (deepghs ONNX) — needs onnxruntime + numpy<2
        if self._try_imgutils():
            self._backend = 'imgutils'
            logger.info("AnimeDetector backend: imgutils (deepghs)")
            return

        # 3. Fallback: OpenCV Haar cascade (face-only, low quality)
        try:
            self._cascade = _ensure_cascade()
            self._backend = 'cascade'
            logger.info("AnimeDetector backend: lbpcascade_animeface (fallback)")
        except Exception as e:
            logger.error("AnimeDetector: all backends failed: %s", e)
            self._backend = 'none'

    def _try_yolo(self) -> bool:
        """Load the YOLOv8 anime-face model via ultralytics if weights exist."""
        if not _YOLO_PATH.exists():
            logger.info(
                "AnimeDetector: %s not found — download with "
                "scripts/download_models.py (or it falls back to cascade).",
                _YOLO_PATH,
            )
            return False
        try:
            import torch
            from ultralytics import YOLO
            self._yolo = YOLO(str(_YOLO_PATH))
            self._yolo_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            # Warm-up / smoke test so a broken load fails here, not silently.
            _probe = np.zeros((64, 64, 3), dtype=np.uint8)
            self._yolo.predict(_probe, conf=0.5, verbose=False, device=self._yolo_device)
            return True
        except Exception as e:
            logger.warning("AnimeDetector: YOLO backend unavailable (%s)", e)
            self._yolo = None
            return False

    def _try_imgutils(self) -> bool:
        """Return True only if imgutils can actually run inference.

        A bare ``import imgutils`` is NOT sufficient: under a wrong interpreter
        (e.g. Python 3.13 with numpy 2.x) or a broken onnxruntime DLL the import
        succeeds but every ``detect_person`` call throws — which the old code
        swallowed silently, yielding 0 detections and 0 saved frames forever.
        We run a tiny real inference probe so a broken backend falls back to the
        cascade cleanly and visibly.
        """
        try:
            import numpy as _np
            from imgutils.detect import detect_person  # noqa: F401
            from PIL import Image as _PILImage
            probe = _PILImage.fromarray(_np.zeros((64, 64, 3), dtype=_np.uint8))
            detect_person(probe, conf_threshold=0.5)  # downloads model on first run
            self._imgutils_ok = True
            return True
        except Exception as e:
            self._imgutils_ok = False
            logger.warning(
                "AnimeDetector: imgutils unavailable (%s) — falling back to "
                "lbpcascade_animeface (face-only, lower quality). Install/repair "
                "with: pip install dghs-imgutils 'numpy<2'", e,
            )
            return False

    # ── Detection implementations ───────────────────────────────

    def _detect_impl(self, frame: np.ndarray) -> Dict[str, List[dict]]:
        if self._backend == 'yolo':
            return self._detect_yolo(frame)
        elif self._backend == 'imgutils':
            return self._detect_imgutils(frame)
        elif self._backend == 'cascade':
            return self._detect_cascade(frame)
        return _empty_detections()

    def _detect_yolo(self, frame: np.ndarray) -> Dict[str, List[dict]]:
        """YOLOv8 anime-face detection (ultralytics). Boxes are faces; we
        return them under 'person' so the cropper builds head-space-aware
        character crops around each face."""
        result = _empty_detections()
        if self._yolo is None:
            return result
        try:
            r = self._yolo.predict(
                frame, conf=self.confidence, verbose=False,
                device=self._yolo_device,
            )[0]
            persons = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), sc in zip(xyxy, confs):
                    persons.append(_make_det(int(x1), int(y1), int(x2), int(y2), float(sc)))
            result['person'] = persons
        except Exception as e:
            logger.debug("YOLO anime detect error: %s", e)
        return result

    def _detect_imgutils(self, frame: np.ndarray) -> Dict[str, List[dict]]:
        """
        Use imgutils deepghs models.
        Tries detect_person (full body) then detect_faces as fallback.
        """
        result = _empty_detections()
        try:
            # Convert BGR → RGB for imgutils (PIL-based internally)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image as _PILImage
            pil = _PILImage.fromarray(rgb)

            persons = []

            # Full body detection
            try:
                from imgutils.detect import detect_person as _dp
                raw = _dp(pil, conf_threshold=self.confidence)
                # raw: list of ((x1,y1,x2,y2), label, score)
                for bbox_tuple, label, score in raw:
                    x1, y1, x2, y2 = (int(v) for v in bbox_tuple)
                    persons.append(_make_det(x1, y1, x2, y2, score))
            except Exception:
                pass

            # Face detection if no persons found
            if not persons:
                try:
                    from imgutils.detect import detect_faces as _df
                    raw = _df(pil, conf_threshold=self.confidence)
                    for bbox_tuple, label, score in raw:
                        x1, y1, x2, y2 = (int(v) for v in bbox_tuple)
                        persons.append(_make_det(x1, y1, x2, y2, score))
                except Exception:
                    pass

            result['person'] = persons

        except Exception as e:
            logger.debug("imgutils detect error: %s", e)

        return result

    def _detect_cascade(self, frame: np.ndarray) -> Dict[str, List[dict]]:
        """Haar cascade face detection — fallback, faces only."""
        result = _empty_detections()
        if self._cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        min_size = max(20, min(w, h) // 20)

        try:
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except Exception as e:
            logger.debug("Cascade detect error: %s", e)
            return result

        persons = []
        if len(faces):
            for (x, y, fw, fh) in faces:
                conf = min(0.95, 0.7 + (fw * fh) / (w * h) * 5)
                persons.append(_make_det(x, y, x + fw, y + fh, conf))

        result['person'] = persons
        return result


# ──────────────────────────────────────────────────────────────
# AutoDetector — routes to YOLO or AnimeDetector per frame
# ──────────────────────────────────────────────────────────────

class AutoDetector:
    """
    Automatically chooses between YOLO (real photo) and AnimeDetector
    (anime/illustration) based on a lightweight per-frame HSV saturation
    heuristic.

    High mean saturation + low hue variance → anime / illustration.
    Otherwise → real photo → YOLO.

    Same public interface as ObjectDetector / AnimeDetector.
    """

    # Saturation threshold: above this mean → likely anime
    _SAT_THRESHOLD = 85

    def __init__(self, yolo_detector, anime_detector: AnimeDetector):
        self._yolo = yolo_detector
        self._anime = anime_detector

    def _is_anime(self, frame: np.ndarray) -> bool:
        """Fast heuristic: mean HSV saturation > threshold."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mean_sat = float(np.mean(hsv[:, :, 1]))
            return mean_sat > self._SAT_THRESHOLD
        except Exception:
            return False

    def detect(self, frame: np.ndarray) -> Dict[str, List[dict]]:
        det = self._anime if self._is_anime(frame) else self._yolo
        return det.detect(frame)

    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict[str, List[dict]]]:
        return [self.detect(f) for f in frames]

    def get_primary_subject(self, detections):
        return self._yolo.get_primary_subject(detections)

    def calculate_head_space(self, bbox, frame_height):
        return self._anime.calculate_head_space(bbox, frame_height)

    def cleanup(self):
        self._anime.cleanup()
