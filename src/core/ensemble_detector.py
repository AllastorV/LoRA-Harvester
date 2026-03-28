"""
Ensemble Object Detection Module
Combines multiple AI models (YOLO, DETR, Faster R-CNN) for higher accuracy
Uses voting/consensus mechanism to verify detections
"""

import logging
import threading
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
    """Return COCO class name for a given 0-based class id."""
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
    model_source: str      # Which model detected this


class EnsembleDetector:
    """
    Ensemble detector combining multiple architectures:
    - YOLOv8 (Ultralytics) - Fast and accurate
    - DETR (Facebook/Meta) - Transformer-based
    - Faster R-CNN (Torchvision) - Traditional but reliable

    Improvements over v1:
    - Thread-safe lazy loading for DETR and Faster R-CNN
    - Confidence-weighted bounding-box merging (instead of simple average)
    - NMS post-processing to remove duplicate overlapping detections
    - Full COCO class names for DETR / Faster R-CNN
    """

    # COCO class mappings
    PERSON_CLASSES = [0]   # person
    ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # animals

    def __init__(self,
                 models_to_use: List[str] = ['yolo', 'detr', 'fasterrcnn'],
                 confidence_threshold: float = 0.5,
                 voting_threshold: int = 2,
                 iou_threshold: float = 0.5,
                 nms_threshold: float = 0.45):
        """
        Initialize ensemble detector.

        Args:
            models_to_use: List of models to use ['yolo', 'detr', 'fasterrcnn']
            confidence_threshold: Minimum confidence for individual models
            voting_threshold: Minimum votes needed for consensus (1-3)
            iou_threshold: IoU threshold for matching detections across models
            nms_threshold: IoU threshold for post-ensemble NMS
        """
        self.models_to_use = list(models_to_use)  # copy to avoid mutation issues
        self.confidence_threshold = confidence_threshold
        self.voting_threshold = min(voting_threshold, max(1, len(models_to_use)))
        self.iou_threshold = iou_threshold
        self.nms_threshold = nms_threshold

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(
            "Initializing Ensemble Detector on %s — models=%s voting=%d/%d",
            self.device.upper(), models_to_use,
            self.voting_threshold, len(models_to_use),
        )

        # Model storage and per-model lazy-load locks
        self.models: Dict = {}
        self._model_locks: Dict[str, threading.Lock] = {
            name: threading.Lock() for name in models_to_use
        }

        self._init_models()

        logger.info("Ensemble detector ready with %d models", len(self.models))

    # ─────────────────────── Model init / lazy load ───────────────────

    def _init_models(self):
        """
        Eagerly load YOLO (fast); mark DETR / Faster R-CNN for lazy loading.
        """
        if 'yolo' in self.models_to_use:
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8...")
                yolo_model = YOLO('yolov8n.pt')
                yolo_model.to(self.device)
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                yolo_model(dummy, verbose=False)
                self.models['yolo'] = yolo_model
                logger.info("YOLOv8 loaded")
            except Exception as e:
                logger.warning("YOLOv8 failed to load: %s", e)
                self.models_to_use.remove('yolo')

        # Mark others for lazy loading
        for name in ('detr', 'fasterrcnn'):
            if name in self.models_to_use:
                self.models[name] = None   # None = not yet loaded
                logger.info("%s marked for lazy loading", name.upper())

    def _lazy_load_detr(self):
        """Thread-safe lazy load of DETR."""
        with self._model_locks['detr']:
            if self.models.get('detr') is not None:
                return   # already loaded or failed (False)
            try:
                logger.info("Loading DETR (first use)...")
                from transformers import DetrImageProcessor, DetrForObjectDetection

                processor = DetrImageProcessor.from_pretrained(
                    "facebook/detr-resnet-50", cache_dir=".cache"
                )
                model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50", cache_dir=".cache"
                )
                model.to(self.device)
                model.eval()

                self.models['detr'] = {'processor': processor, 'model': model}
                logger.info("DETR loaded successfully")
            except Exception as e:
                logger.warning("DETR failed to load: %s", e)
                self.models['detr'] = False
                if 'detr' in self.models_to_use:
                    self.models_to_use.remove('detr')

    def _lazy_load_fasterrcnn(self):
        """Thread-safe lazy load of Faster R-CNN."""
        with self._model_locks['fasterrcnn']:
            if self.models.get('fasterrcnn') is not None:
                return
            try:
                logger.info("Loading Faster R-CNN (first use)...")
                from torchvision.models.detection import (
                    fasterrcnn_resnet50_fpn_v2,
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                )

                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                model = fasterrcnn_resnet50_fpn_v2(weights=weights)
                model.to(self.device)
                model.eval()

                self.models['fasterrcnn'] = {
                    'model': model,
                    'transforms': weights.transforms(),
                }
                logger.info("Faster R-CNN loaded successfully")
            except Exception as e:
                logger.warning("Faster R-CNN failed to load: %s", e)
                self.models['fasterrcnn'] = False
                if 'fasterrcnn' in self.models_to_use:
                    self.models_to_use.remove('fasterrcnn')

    # ────────────────────── Per-model detection ───────────────────────

    def detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 detection."""
        if self.models.get('yolo') is None:
            return []

        results = self.models['yolo'](
            frame, conf=self.confidence_threshold, verbose=False
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(Detection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=conf,
                class_id=cls_id,
                class_name=results.names[cls_id],
                model_source='yolo',
            ))
        return detections

    def detect_detr(self, frame: np.ndarray) -> List[Detection]:
        """Run DETR detection."""
        if 'detr' not in self.models_to_use:
            return []
        if self.models.get('detr') is None:
            self._lazy_load_detr()
        if not self.models.get('detr'):
            return []

        processor = self.models['detr']['processor']
        model = self.models['detr']['model']

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]
        del inputs, outputs

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = box.cpu().numpy()
            cls_id = int(label)
            detections.append(Detection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=float(score),
                class_id=cls_id,
                class_name=_coco_name(cls_id),
                model_source='detr',
            ))
        return detections

    def detect_fasterrcnn(self, frame: np.ndarray) -> List[Detection]:
        """Run Faster R-CNN detection."""
        if 'fasterrcnn' not in self.models_to_use:
            return []
        if self.models.get('fasterrcnn') is None:
            self._lazy_load_fasterrcnn()
        if not self.models.get('fasterrcnn'):
            return []

        model = self.models['fasterrcnn']['model']
        transforms = self.models['fasterrcnn']['transforms']

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            image_tensor = (
                torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            ).to(self.device)
            image_transformed = transforms(image_tensor)
            predictions = model([image_transformed])[0]
        del image_tensor, image_transformed

        detections = []
        for box, label, score in zip(
            predictions['boxes'], predictions['labels'], predictions['scores']
        ):
            if score < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            cls_id = int(label) - 1   # Faster R-CNN uses 1-based indexing
            detections.append(Detection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=float(score),
                class_id=cls_id,
                class_name=_coco_name(cls_id),
                model_source='fasterrcnn',
            ))
        return detections

    # ─────────────────────── IoU / NMS helpers ────────────────────────

    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two boxes."""
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
        """
        Non-Maximum Suppression to remove duplicate overlapping detections.

        Keeps the detection with the highest confidence when two boxes of
        the same class overlap more than nms_threshold.
        """
        if not detections:
            return []

        # Sort by confidence descending
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

    # ──────────────────────── Ensemble voting ─────────────────────────

    def ensemble_voting(self, all_detections: List[Detection]) -> List[Detection]:
        """
        Apply voting mechanism to combine detections from multiple models.

        Changes vs v1:
        - Confidence-weighted bounding box merging instead of simple mean
        - NMS applied after voting to remove residual overlaps

        Args:
            all_detections: All detections from all models

        Returns:
            Consensus detections that meet voting threshold (after NMS)
        """
        if not all_detections:
            return []

        # Group by spatial proximity and class
        detection_groups: List[List[Detection]] = []

        for detection in all_detections:
            matched = False
            for group in detection_groups:
                for group_det in group:
                    if (self.calculate_iou(detection.bbox, group_det.bbox) >= self.iou_threshold
                            and detection.class_id == group_det.class_id):
                        group.append(detection)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                detection_groups.append([detection])

        consensus: List[Detection] = []

        for group in detection_groups:
            model_votes = set(d.model_source for d in group)

            if len(model_votes) < self.voting_threshold:
                continue

            # Confidence-weighted bounding box merge
            total_conf = sum(d.confidence for d in group)
            if total_conf == 0:
                weights = [1.0 / len(group)] * len(group)
            else:
                weights = [d.confidence / total_conf for d in group]

            merged_bbox = [
                int(sum(w * d.bbox[i] for w, d in zip(weights, group)))
                for i in range(4)
            ]

            max_conf = max(d.confidence for d in group)
            # Pick class from the most confident detection
            best = max(group, key=lambda d: d.confidence)
            models_str = ','.join(sorted(model_votes))

            consensus.append(Detection(
                bbox=merged_bbox,
                confidence=max_conf,
                class_id=best.class_id,
                class_name=best.class_name,
                model_source=f"ensemble({models_str})",
            ))

        # Post-ensemble NMS
        return self._apply_nms(consensus)

    # ──────────────────────── Public detect API ───────────────────────

    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Run ensemble detection on frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Dictionary with categorized detections
        """
        all_detections: List[Detection] = []

        if 'yolo' in self.models:
            all_detections.extend(self.detect_yolo(frame))
        if 'detr' in self.models:
            all_detections.extend(self.detect_detr(frame))
        if 'fasterrcnn' in self.models:
            all_detections.extend(self.detect_fasterrcnn(frame))

        consensus = self.ensemble_voting(all_detections)
        return self._categorize_detections(consensus)

    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict[str, List[Dict]]]:
        """
        Run ensemble detection on multiple frames.
        YOLO uses native batch; DETR and Faster R-CNN run per-frame.

        Args:
            frames: List of input frames (BGR format)

        Returns:
            List of detection dictionaries
        """
        if not frames:
            return []

        # YOLO batch detection
        yolo_batch_results: List[List[Detection]] = [[] for _ in frames]
        if self.models.get('yolo') is not None:
            raw = self.models['yolo'](
                frames, conf=self.confidence_threshold, verbose=False
            )
            for i, results in enumerate(raw):
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    yolo_batch_results[i].append(Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=float(box.conf[0]),
                        class_id=cls_id,
                        class_name=results.names[cls_id],
                        model_source='yolo',
                    ))

        all_results = []
        for i, frame in enumerate(frames):
            all_detections = list(yolo_batch_results[i])

            if 'detr' in self.models:
                all_detections.extend(self.detect_detr(frame))
            if 'fasterrcnn' in self.models:
                all_detections.extend(self.detect_fasterrcnn(frame))

            consensus = self.ensemble_voting(all_detections)
            all_results.append(self._categorize_detections(consensus))

        return all_results

    # ───────────────────── Categorisation helpers ─────────────────────

    def _categorize_detections(
        self, consensus_detections: List[Detection]
    ) -> Dict[str, List[Dict]]:
        """Categorize detection results into person/animal/object."""
        categorized: Dict[str, List[Dict]] = {
            'person': [], 'animal': [], 'object': []
        }

        for det in consensus_detections:
            detection_dict = {
                'bbox': det.bbox,
                'confidence': det.confidence,
                'class_id': det.class_id,
                'class_name': det.class_name,
                'models': det.model_source,
            }
            if det.class_id in self.PERSON_CLASSES:
                categorized['person'].append(detection_dict)
            elif det.class_id in self.ANIMAL_CLASSES:
                categorized['animal'].append(detection_dict)
            else:
                categorized['object'].append(detection_dict)

        return categorized

    # ─────────────────── Shared detector interface ────────────────────

    def get_primary_subject(
        self, detections: Dict[str, List[Dict]]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Get primary subject from detections (same API as ObjectDetector)."""
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
        """Calculate head space ratio."""
        return bbox[1] / frame_height if frame_height > 0 else 0.0

    def get_all_detections_bbox(
        self, detections: Dict[str, List[Dict]]
    ) -> Optional[List[int]]:
        """Get bounding box encompassing all detections."""
        all_boxes = [
            det['bbox']
            for cat in ('person', 'animal', 'object')
            for det in detections[cat]
        ]
        if not all_boxes:
            return None

        return [
            min(b[0] for b in all_boxes),
            min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes),
            max(b[3] for b in all_boxes),
        ]

    def cleanup(self):
        """Release all models from memory"""
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
        logger.info("EnsembleDetector cleaned up")

    def get_detection_stats(self) -> Dict:
        """Get statistics about loaded models."""
        loaded = [k for k, v in self.models.items() if v and v is not False]
        return {
            'total_models': len(self.models),
            'loaded_models': loaded,
            'voting_threshold': self.voting_threshold,
            'nms_threshold': self.nms_threshold,
            'device': self.device,
        }
