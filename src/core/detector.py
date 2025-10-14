"""
GPU-accelerated Object Detection Module using YOLOv8
Detects humans, animals, and objects in video frames
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple


class ObjectDetector:
    """YOLOv8-based object detector with GPU support"""
    
    # COCO dataset class mappings
    PERSON_CLASSES = [0]  # person
    ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    
    def __init__(self, model_size: str = 'yolov8n.pt', confidence: float = 0.5):
        """
        Initialize the detector
        
        Args:
            model_size: YOLO model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence: Detection confidence threshold
        """
        self.confidence = confidence
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🚀 Initializing YOLOv8 on {self.device.upper()}")
        
        # Load YOLO model
        self.model = YOLO(model_size)
        self.model.to(self.device)
        
        # Warm up the model
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)
        
        print(f"✅ Model loaded successfully on {self.device.upper()}")
    
    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with 'person', 'animal', and 'object' keys containing detected objects
        """
        # Run inference
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        detections = {
            'person': [],
            'animal': [],
            'object': []
        }
        
        # Process detections
        boxes = results.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            detection = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': results.names[class_id]
            }
            
            # Categorize detection
            if class_id in self.PERSON_CLASSES:
                detections['person'].append(detection)
            elif class_id in self.ANIMAL_CLASSES:
                detections['animal'].append(detection)
            else:
                detections['object'].append(detection)
        
        return detections
    
    def get_primary_subject(self, detections: Dict[str, List[Dict]]) -> Tuple[str, Dict]:
        """
        Get the primary subject from detections (largest and most confident)
        
        Args:
            detections: Detection results
            
        Returns:
            Tuple of (category, detection_dict) or (None, None) if no detections
        """
        all_subjects = []
        
        # Priority: person > animal > object
        for category in ['person', 'animal', 'object']:
            for det in detections[category]:
                bbox = det['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                score = area * det['confidence']  # Size + confidence scoring
                all_subjects.append((category, det, score))
        
        if not all_subjects:
            return None, None
        
        # Sort by score and return the best
        all_subjects.sort(key=lambda x: x[2], reverse=True)
        return all_subjects[0][0], all_subjects[0][1]
    
    def calculate_head_space(self, bbox: List[int], frame_height: int) -> float:
        """
        Calculate head space ratio for person detection
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_height: Height of the frame
            
        Returns:
            Head space ratio (0-1)
        """
        y1 = bbox[1]
        head_space = y1 / frame_height if frame_height > 0 else 0
        return head_space
    
    def get_all_detections_bbox(self, detections: Dict[str, List[Dict]]) -> List[int]:
        """
        Get bounding box that encompasses all detections
        
        Args:
            detections: Detection results
            
        Returns:
            Bounding box [x1, y1, x2, y2] or None
        """
        all_boxes = []
        
        for category in ['person', 'animal', 'object']:
            for det in detections[category]:
                all_boxes.append(det['bbox'])
        
        if not all_boxes:
            return None
        
        # Find min/max coordinates
        x1 = min(box[0] for box in all_boxes)
        y1 = min(box[1] for box in all_boxes)
        x2 = max(box[2] for box in all_boxes)
        y2 = max(box[3] for box in all_boxes)
        
        return [x1, y1, x2, y2]
