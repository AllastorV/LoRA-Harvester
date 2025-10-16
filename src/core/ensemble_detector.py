"""
Ensemble Object Detection Module
Combines multiple AI models (YOLO, DETR, Faster R-CNN) for higher accuracy
Uses voting/consensus mechanism to verify detections
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Detection:
    """Unified detection result"""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    model_source: str  # Which model detected this


class EnsembleDetector:
    """
    Ensemble detector combining multiple architectures:
    - YOLOv8 (Ultralytics) - Fast and accurate
    - DETR (Facebook/Meta) - Transformer-based
    - Faster R-CNN (Torchvision) - Traditional but reliable
    """
    
    # COCO class mappings
    PERSON_CLASSES = [0]  # person
    ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # animals
    
    def __init__(self, 
                 models_to_use: List[str] = ['yolo', 'detr', 'fasterrcnn'],
                 confidence_threshold: float = 0.5,
                 voting_threshold: int = 2,
                 iou_threshold: float = 0.5):
        """
        Initialize ensemble detector
        
        Args:
            models_to_use: List of models to use ['yolo', 'detr', 'fasterrcnn']
            confidence_threshold: Minimum confidence for individual models
            voting_threshold: Minimum votes needed for consensus (1-3)
            iou_threshold: IoU threshold for matching detections across models
        """
        self.models_to_use = models_to_use
        self.confidence_threshold = confidence_threshold
        self.voting_threshold = min(voting_threshold, len(models_to_use))
        self.iou_threshold = iou_threshold
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing Ensemble Detector on {self.device.upper()}")
        print(f"📊 Models: {', '.join(models_to_use)}")
        print(f"🗳️  Voting threshold: {self.voting_threshold}/{len(models_to_use)}")
        
        # Initialize models
        self.models = {}
        self._init_models()
        
        print(f"✅ Ensemble detector ready with {len(self.models)} models")
    
    def _init_models(self):
        """Initialize selected models (lazy loading for better performance)"""
        
        # Only initialize YOLO immediately, others on-demand
        # 1. YOLOv8 (Ultralytics) - Fast to load
        if 'yolo' in self.models_to_use:
            try:
                from ultralytics import YOLO
                print("  📦 Loading YOLOv8...")
                yolo_model = YOLO('yolov8n.pt')
                yolo_model.to(self.device)
                # Warm up
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                yolo_model(dummy, verbose=False)
                self.models['yolo'] = yolo_model
                print("  ✅ YOLOv8 loaded")
            except Exception as e:
                print(f"  ⚠️  YOLOv8 failed to load: {e}")
                self.models_to_use.remove('yolo')
        
        # Mark other models for lazy loading
        if 'detr' in self.models_to_use:
            self.models['detr'] = None  # Will load on first use
            print("  ⏳ DETR marked for lazy loading...")
        
        if 'fasterrcnn' in self.models_to_use:
            self.models['fasterrcnn'] = None  # Will load on first use
            print("  ⏳ Faster R-CNN marked for lazy loading...")
    
    def _lazy_load_detr(self):
        """Lazy load DETR model"""
        if self.models.get('detr') is None and 'detr' in self.models_to_use:
            try:
                print("  📦 Loading DETR now (first use)...")
                from transformers import DetrImageProcessor, DetrForObjectDetection
                
                processor = DetrImageProcessor.from_pretrained(
                    "facebook/detr-resnet-50",
                    cache_dir=".cache"
                )
                model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50",
                    cache_dir=".cache"
                )
                model.to(self.device)
                model.eval()
                
                self.models['detr'] = {
                    'processor': processor,
                    'model': model
                }
                print("  ✅ DETR loaded successfully")
            except Exception as e:
                print(f"  ⚠️  DETR failed to load: {e}")
                self.models['detr'] = False  # Mark as failed
                self.models_to_use.remove('detr')
    
    def _lazy_load_fasterrcnn(self):
        """Lazy load Faster R-CNN model"""
        if self.models.get('fasterrcnn') is None and 'fasterrcnn' in self.models_to_use:
            try:
                print("  📦 Loading Faster R-CNN now (first use)...")
                from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
                
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                model = fasterrcnn_resnet50_fpn_v2(weights=weights)
                model.to(self.device)
                model.eval()
                
                self.models['fasterrcnn'] = {
                    'model': model,
                    'transforms': weights.transforms()
                }
                print("  ✅ Faster R-CNN loaded successfully")
            except Exception as e:
                print(f"  ⚠️  Faster R-CNN failed to load: {e}")
                self.models['fasterrcnn'] = False  # Mark as failed
                self.models_to_use.remove('fasterrcnn')
    
    def detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 detection"""
        if 'yolo' not in self.models:
            return []
        
        results = self.models['yolo'](frame, conf=self.confidence_threshold, verbose=False)[0]
        detections = []
        
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            detections.append(Detection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=conf,
                class_id=cls_id,
                class_name=results.names[cls_id],
                model_source='yolo'
            ))
        
        return detections
    
    def detect_detr(self, frame: np.ndarray) -> List[Detection]:
        """Run DETR detection"""
        if 'detr' not in self.models_to_use:
            return []
        
        # Lazy load if needed
        if self.models.get('detr') is None:
            self._lazy_load_detr()
        
        # Check if loading failed
        if self.models.get('detr') is False or self.models.get('detr') is None:
            return []
        
        processor = self.models['detr']['processor']
        model = self.models['detr']['model']
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare inputs
        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            conf = float(score)
            cls_id = int(label)
            
            # DETR uses COCO classes
            detections.append(Detection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=conf,
                class_id=cls_id,
                class_name=f"class_{cls_id}",
                model_source='detr'
            ))
        
        return detections
    
    def detect_fasterrcnn(self, frame: np.ndarray) -> List[Detection]:
        """Run Faster R-CNN detection"""
        if 'fasterrcnn' not in self.models_to_use:
            return []
        
        # Lazy load if needed
        if self.models.get('fasterrcnn') is None:
            self._lazy_load_fasterrcnn()
        
        # Check if loading failed
        if self.models.get('fasterrcnn') is False or self.models.get('fasterrcnn') is None:
            return []
        
        model = self.models['fasterrcnn']['model']
        transforms = self.models['fasterrcnn']['transforms']
        
        # Convert to RGB and tensor
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Apply transforms
        image_transformed = transforms(image_tensor)
        
        # Inference
        with torch.no_grad():
            predictions = model([image_transformed])[0]
        
        detections = []
        for box, label, score in zip(
            predictions['boxes'], 
            predictions['labels'], 
            predictions['scores']
        ):
            if score >= self.confidence_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                conf = float(score)
                cls_id = int(label) - 1  # Faster R-CNN uses 1-based indexing
                
                detections.append(Detection(
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    confidence=conf,
                    class_id=cls_id,
                    class_name=f"class_{cls_id}",
                    model_source='fasterrcnn'
                ))
        
        return detections
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def ensemble_voting(self, all_detections: List[Detection]) -> List[Detection]:
        """
        Apply voting mechanism to combine detections from multiple models
        
        Args:
            all_detections: All detections from all models
            
        Returns:
            Consensus detections that meet voting threshold
        """
        if not all_detections:
            return []
        
        # Group detections by spatial proximity and class
        detection_groups = []
        
        for detection in all_detections:
            # Try to find a matching group
            matched = False
            for group in detection_groups:
                # Check if this detection matches any in the group
                for group_det in group:
                    iou = self.calculate_iou(detection.bbox, group_det.bbox)
                    if iou >= self.iou_threshold and detection.class_id == group_det.class_id:
                        group.append(detection)
                        matched = True
                        break
                if matched:
                    break
            
            # Create new group if no match found
            if not matched:
                detection_groups.append([detection])
        
        # Apply voting threshold
        consensus_detections = []
        
        for group in detection_groups:
            # Count votes (unique models)
            model_votes = set(det.model_source for det in group)
            
            if len(model_votes) >= self.voting_threshold:
                # Merge detections in group (average bbox, max confidence)
                avg_bbox = [
                    int(np.mean([d.bbox[0] for d in group])),
                    int(np.mean([d.bbox[1] for d in group])),
                    int(np.mean([d.bbox[2] for d in group])),
                    int(np.mean([d.bbox[3] for d in group]))
                ]
                
                max_conf = max(d.confidence for d in group)
                class_id = group[0].class_id
                class_name = group[0].class_name
                models = ','.join(model_votes)
                
                consensus_detections.append(Detection(
                    bbox=avg_bbox,
                    confidence=max_conf,
                    class_id=class_id,
                    class_name=class_name,
                    model_source=f"ensemble({models})"
                ))
        
        return consensus_detections
    
    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Run ensemble detection on frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with categorized detections
        """
        all_detections = []
        
        # Run each model
        if 'yolo' in self.models:
            all_detections.extend(self.detect_yolo(frame))
        
        if 'detr' in self.models:
            all_detections.extend(self.detect_detr(frame))
        
        if 'fasterrcnn' in self.models:
            all_detections.extend(self.detect_fasterrcnn(frame))
        
        # Apply ensemble voting
        consensus_detections = self.ensemble_voting(all_detections)
        
        # Categorize detections
        categorized = {
            'person': [],
            'animal': [],
            'object': []
        }
        
        for det in consensus_detections:
            detection_dict = {
                'bbox': det.bbox,
                'confidence': det.confidence,
                'class_id': det.class_id,
                'class_name': det.class_name,
                'models': det.model_source
            }
            
            if det.class_id in self.PERSON_CLASSES:
                categorized['person'].append(detection_dict)
            elif det.class_id in self.ANIMAL_CLASSES:
                categorized['animal'].append(detection_dict)
            else:
                categorized['object'].append(detection_dict)
        
        return categorized
    
    def get_primary_subject(self, detections: Dict[str, List[Dict]]) -> Tuple[Optional[str], Optional[Dict]]:
        """Get primary subject from detections (same as original detector)"""
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
        """Calculate head space ratio"""
        y1 = bbox[1]
        head_space = y1 / frame_height if frame_height > 0 else 0
        return head_space
    
    def get_all_detections_bbox(self, detections: Dict[str, List[Dict]]) -> Optional[List[int]]:
        """Get bounding box encompassing all detections"""
        all_boxes = []
        
        for category in ['person', 'animal', 'object']:
            for det in detections[category]:
                all_boxes.append(det['bbox'])
        
        if not all_boxes:
            return None
        
        x1 = min(box[0] for box in all_boxes)
        y1 = min(box[1] for box in all_boxes)
        x2 = max(box[2] for box in all_boxes)
        y2 = max(box[3] for box in all_boxes)
        
        return [x1, y1, x2, y2]
    
    def get_detection_stats(self) -> Dict:
        """Get statistics about loaded models"""
        return {
            'total_models': len(self.models),
            'loaded_models': list(self.models.keys()),
            'voting_threshold': self.voting_threshold,
            'device': self.device
        }
