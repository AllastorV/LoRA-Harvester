"""
Ensemble Object Detection Module
YOLOv8-only detection with NMS post-processing.
DETR and Faster R-CNN have been removed to reduce complexity and dependencies.
"""

import logging
import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# COCO 80-class names (index 0 = person, indices 14-23 = animals)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def _coco_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f'class_{class_id}'


@dataclass
class Detection:
    """Unified detection result."""
    bbox: List[int]        # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    model_source: str


class EnsembleDetector:
    """
    Detection wrapper around YOLOv8 with NMS post-processing.
    Keeps the same public API so callers (UnifiedVideoProcessor, main_window)
    don't need changes.
    """

    PERSON_CLASSES = [0]
    ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    def __init__(self,
                 models_to_use: List[str] = None,
                 confidence_threshold: float = 0.5,
                 voting_threshold: int = 1,
                 iou_threshold: float = 0.5,
                 nms_threshold: float = 0.45):
        # Accept legacy kwargs but only use YOLO.
        self.models_to_use = ['yolo']
        self.confidence_threshold = confidence_threshold
        self.voting_threshold = 1
        self.iou_threshold = iou_threshold
        self.nms_threshold = nms_threshold

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Initializing Detector on %s (YOLO-only)", self.device.upper())

        self.models: Dict = {}
        self._init_models()
        logger.info("Detector ready")

    def _init_models(self):
        try:
            from ultralytics import YOLO
            from src.core.model_paths import yolo_model_path
            logger.info("Loading YOLOv8...")
            _yp = yolo_model_path('yolov8n.pt')
            yolo_model = YOLO(str(_yp) if _yp.exists() else 'yolov8n.pt')
            yolo_model.to(self.device)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            yolo_model(dummy, verbose=False)
            self.models['yolo'] = yolo_model
            logger.info("YOLOv8 loaded")
        except Exception as e:
            logger.warning("YOLOv8 failed to load: %s", e)

    # ────────────────────── Detection ────────────────────────────

    def detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        if self.models.get('yolo') is None:
            return []
        results = self.models['yolo'](
            frame, conf=self.confidence_threshold, verbose=False
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            detections.append(Detection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=float(box.conf[0]),
                class_id=cls_id,
                class_name=results.names[cls_id],
                model_source='yolo',
            ))
        return detections

    # ─────────────────────── NMS ─────────────────────────────────

    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in detections:
            suppress = False
            for k in kept:
                if k.class_id == det.class_id:
                    if self.calculate_iou(det.bbox, k.bbox) > self.nms_threshold:
                        suppress = True
                        break
            if not suppress:
                kept.append(det)
        return kept

    def ensemble_voting(self, all_detections: List[Detection]) -> List[Detection]:
        """With a single model, voting is just NMS."""
        return self._apply_nms(all_detections)

    # ──────────────────────── Public API ─────────────────────────

    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        detections = self.detect_yolo(frame)
        consensus = self.ensemble_voting(detections)
        return self._categorize_detections(consensus)

    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict[str, List[Dict]]]:
        if not frames:
            return []
        yolo_batch: List[List[Detection]] = [[] for _ in frames]
        if self.models.get('yolo') is not None:
            raw = self.models['yolo'](
                frames, conf=self.confidence_threshold, verbose=False)
            for i, results in enumerate(raw):
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    yolo_batch[i].append(Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=float(box.conf[0]),
                        class_id=cls_id,
                        class_name=results.names[cls_id],
                        model_source='yolo',
                    ))
        return [
            self._categorize_detections(self.ensemble_voting(dets))
            for dets in yolo_batch
        ]

    # ───────────────────── Categorisation ────────────────────────

    def _categorize_detections(
        self, consensus_detections: List[Detection]
    ) -> Dict[str, List[Dict]]:
        categorized: Dict[str, List[Dict]] = {
            'person': [], 'animal': [], 'object': []}
        for det in consensus_detections:
            d = {
                'bbox': det.bbox, 'confidence': det.confidence,
                'class_id': det.class_id, 'class_name': det.class_name,
                'models': det.model_source,
            }
            if det.class_id in self.PERSON_CLASSES:
                categorized['person'].append(d)
            elif det.class_id in self.ANIMAL_CLASSES:
                categorized['animal'].append(d)
            else:
                categorized['object'].append(d)
        return categorized

    # ─────────────────── Shared interface ────────────────────────

    def get_primary_subject(
        self, detections: Dict[str, List[Dict]]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        all_subjects = []
        for category in ['person', 'animal', 'object']:
            for det in detections[category]:
                bbox = det['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                score = area * det['confidence']
                all_subjects.append((category, det, score))
        if not all_subjects:
            return None, None
        all_subjects.sort(key=lambda x: x[2], reverse=True)
        return all_subjects[0][0], all_subjects[0][1]

    def calculate_head_space(self, bbox: List[int], frame_height: int) -> float:
        return bbox[1] / frame_height if frame_height > 0 else 0.0

    def get_all_detections_bbox(
        self, detections: Dict[str, List[Dict]]
    ) -> Optional[List[int]]:
        all_boxes = [
            det['bbox']
            for cat in ('person', 'animal', 'object')
            for det in detections[cat]
        ]
        if not all_boxes:
            return None
        return [
            min(b[0] for b in all_boxes), min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes), max(b[3] for b in all_boxes),
        ]

    def cleanup(self):
        for name, model_data in self.models.items():
            if model_data and model_data is not False:
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        del model_data['model']
                else:
                    del model_data
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Detector cleaned up")

    def get_detection_stats(self) -> Dict:
        loaded = [k for k, v in self.models.items() if v and v is not False]
        return {
            'total_models': len(self.models),
            'loaded_models': loaded,
            'voting_threshold': self.voting_threshold,
            'nms_threshold': self.nms_threshold,
            'device': self.device,
        }
