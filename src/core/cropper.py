"""
Smart Cropping Algorithm
Handles intelligent cropping with subject centering, head space, and zoom adjustment
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


class SmartCropper:
    """Intelligent cropping with subject awareness"""
    
    # Aspect ratios for vertical formats
    ASPECT_RATIOS = {
        '9:16': 9/16,
        '3:4': 3/4,
        '1:1': 1.0,
        '4:5': 4/5
    }
    
    def __init__(self, target_format: str = '9:16', min_padding: int = 500):
        """
        Initialize smart cropper
        
        Args:
            target_format: Target aspect ratio format
            min_padding: Minimum padding around detected objects (in pixels)
        """
        self.target_format = target_format
        self.aspect_ratio = self.ASPECT_RATIOS.get(target_format, 9/16)
        self.min_padding = min_padding
        
        # Head space parameters
        self.ideal_head_space = 0.15  # Ideal 15% space above head
        self.max_head_space = 0.25    # Maximum 25% space
        self.min_head_space = 0.05    # Minimum 5% space
    
    def calculate_crop_box(self, 
                          frame_shape: Tuple[int, int],
                          subject_bbox: List[int],
                          category: str,
                          head_space_ratio: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate optimal crop box for the frame
        
        Args:
            frame_shape: (height, width) of the frame
            subject_bbox: [x1, y1, x2, y2] of the subject
            category: 'person', 'animal', or 'object'
            head_space_ratio: Current head space ratio for persons
            
        Returns:
            Crop box (x, y, width, height) or None if not possible
        """
        frame_height, frame_width = frame_shape
        x1, y1, x2, y2 = subject_bbox
        
        # Calculate subject dimensions
        subject_width = x2 - x1
        subject_height = y2 - y1
        subject_center_x = (x1 + x2) // 2
        subject_center_y = (y1 + y2) // 2
        
        # Add padding
        padded_width = subject_width + 2 * self.min_padding
        padded_height = subject_height + 2 * self.min_padding
        
        # Calculate required crop dimensions based on aspect ratio
        if padded_width / padded_height > self.aspect_ratio:
            # Width-constrained
            crop_width = padded_width
            crop_height = int(crop_width / self.aspect_ratio)
        else:
            # Height-constrained
            crop_height = padded_height
            crop_width = int(crop_height * self.aspect_ratio)
        
        # Ensure crop doesn't exceed frame dimensions
        if crop_width > frame_width or crop_height > frame_height:
            # Scale down proportionally
            scale = min(frame_width / crop_width, frame_height / crop_height)
            crop_width = int(crop_width * scale)
            crop_height = int(crop_height * scale)
        
        # Adjust center for head space (only for persons)
        if category == 'person' and head_space_ratio > 0:
            # Calculate ideal position considering head space
            target_head_space = self.ideal_head_space
            
            # Adjust vertical position to achieve ideal head space
            if head_space_ratio < self.min_head_space:
                # Need more space above, shift up
                subject_center_y -= int(crop_height * 0.1)
            elif head_space_ratio > self.max_head_space:
                # Too much space above, shift down
                subject_center_y += int(crop_height * 0.1)
        
        # Calculate crop position (center on subject)
        crop_x = subject_center_x - crop_width // 2
        crop_y = subject_center_y - crop_height // 2
        
        # Ensure crop stays within frame bounds
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = max(0, min(crop_y, frame_height - crop_height))
        
        return (crop_x, crop_y, crop_width, crop_height)
    
    def apply_crop(self, frame: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply crop to frame
        
        Args:
            frame: Input frame
            crop_box: (x, y, width, height)
            
        Returns:
            Cropped frame
        """
        x, y, w, h = crop_box
        cropped = frame[y:y+h, x:x+w]
        return cropped
    
    def adaptive_zoom(self, 
                     frame: np.ndarray,
                     crop_box: Tuple[int, int, int, int],
                     zoom_factor: float = 1.2) -> np.ndarray:
        """
        Apply adaptive zoom to focus more on subject
        
        Args:
            frame: Input frame
            crop_box: Current crop box
            zoom_factor: Zoom factor (>1 to zoom in)
            
        Returns:
            Zoomed and cropped frame
        """
        x, y, w, h = crop_box
        
        # Calculate zoomed dimensions
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        
        # Calculate new position (center zoom)
        center_x = x + w // 2
        center_y = y + h // 2
        new_x = center_x - new_w // 2
        new_y = center_y - new_h // 2
        
        # Ensure bounds
        new_x = max(0, min(new_x, frame.shape[1] - new_w))
        new_y = max(0, min(new_y, frame.shape[0] - new_h))
        
        # Crop and resize back to target dimensions
        zoomed = frame[new_y:new_y+new_h, new_x:new_x+new_w]
        resized = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def calculate_quality_score(self, 
                                frame_shape: Tuple[int, int],
                                crop_box: Tuple[int, int, int, int],
                                subject_bbox: List[int]) -> float:
        """
        Calculate quality score for a crop
        
        Args:
            frame_shape: Original frame shape
            crop_box: Crop dimensions
            subject_bbox: Subject bounding box
            
        Returns:
            Quality score (0-1, higher is better)
        """
        frame_height, frame_width = frame_shape
        x, y, w, h = crop_box
        sx1, sy1, sx2, sy2 = subject_bbox
        
        # Calculate subject coverage in crop
        subject_area = (sx2 - sx1) * (sy2 - sy1)
        crop_area = w * h
        coverage = subject_area / crop_area if crop_area > 0 else 0
        
        # Penalize if crop is at frame edges (prefer centered crops)
        edge_penalty = 0
        if x <= 10 or y <= 10:
            edge_penalty += 0.1
        if x + w >= frame_width - 10 or y + h >= frame_height - 10:
            edge_penalty += 0.1
        
        # Ideal coverage is 30-60% of frame
        coverage_score = 1.0 - abs(0.45 - coverage)
        
        # Final score
        quality = max(0, coverage_score - edge_penalty)
        
        return quality
