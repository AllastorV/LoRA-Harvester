"""
Video Processing Module
Handles video frame extraction, processing, and saving
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, List
from datetime import datetime


class VideoProcessor:
    """Main video processing class"""
    
    def __init__(self, 
                 video_path: str,
                 output_dir: str,
                 detector,
                 text_detector,
                 cropper):
        """
        Initialize video processor
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for cropped frames
            detector: ObjectDetector instance
            text_detector: SubtitleDetector instance
            cropper: SmartCropper instance
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.detector = detector
        self.text_detector = text_detector
        self.cropper = cropper
        
        # Create output directory structure
        self.create_output_structure()
        
        # Video properties
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        
        # Processing stats
        self.stats = {
            'processed_frames': 0,
            'saved_frames': 0,
            'skipped_text': 0,
            'skipped_no_detection': 0,
            'person_frames': 0,
            'animal_frames': 0,
            'object_frames': 0
        }
    
    def create_output_structure(self):
        """Create output directory structure with subdirectories"""
        base_path = Path(self.output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.person_dir = base_path / 'persons'
        self.animal_dir = base_path / 'animals'
        self.object_dir = base_path / 'objects'
        
        self.person_dir.mkdir(exist_ok=True)
        self.animal_dir.mkdir(exist_ok=True)
        self.object_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output structure created at: {base_path}")
    
    def open_video(self) -> bool:
        """
        Open video file and get properties
        
        Returns:
            True if successful
        """
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open video: {self.video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎬 Video opened: {self.frame_width}x{self.frame_height} @ {self.fps:.2f}fps")
        print(f"📊 Total frames: {self.total_frames}")
        
        return True
    
    def process_video(self,
                     frame_interval: int = 30,
                     skip_text: bool = True,
                     use_quick_text_check: bool = True,
                     progress_callback: Optional[Callable] = None,
                     stop_callback: Optional[Callable] = None) -> Dict:
        """
        Process video and extract cropped frames
        
        Args:
            frame_interval: Process every Nth frame
            skip_text: Skip frames with subtitles/text
            use_quick_text_check: Use fast text detection first
            progress_callback: Callback function for progress updates
            stop_callback: Callback to check if processing should stop
            
        Returns:
            Processing statistics
        """
        if not self.open_video():
            return self.stats
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                # Check if we should stop
                if stop_callback and stop_callback():
                    print("\n⏹️  Processing stopped by user")
                    break
                
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Process only at specified intervals
                if frame_count % frame_interval != 0:
                    continue
                
                processed_count += 1
                self.stats['processed_frames'] = processed_count
                
                # Progress callback
                if progress_callback:
                    progress = (frame_count / self.total_frames) * 100
                    progress_callback(progress, self.stats)
                
                # Skip text frames if enabled
                if skip_text:
                    # Quick check first
                    if use_quick_text_check and self.text_detector.quick_text_check(frame):
                        self.stats['skipped_text'] += 1
                        continue
                    
                    # Deep check if quick check passed but we want to be sure
                    # (Optional: uncomment for more accuracy but slower processing)
                    # has_text, _ = self.text_detector.has_text(frame)
                    # if has_text:
                    #     self.stats['skipped_text'] += 1
                    #     continue
                
                # Detect objects
                detections = self.detector.detect(frame)
                
                # Get primary subject
                category, subject = self.detector.get_primary_subject(detections)
                
                if category is None:
                    self.stats['skipped_no_detection'] += 1
                    continue
                
                # Calculate head space for persons
                head_space = 0.0
                if category == 'person':
                    head_space = self.detector.calculate_head_space(
                        subject['bbox'], 
                        self.frame_height
                    )
                
                # Calculate crop box
                crop_box = self.cropper.calculate_crop_box(
                    (self.frame_height, self.frame_width),
                    subject['bbox'],
                    category,
                    head_space
                )
                
                if crop_box is None:
                    continue
                
                # Apply crop
                cropped_frame = self.cropper.apply_crop(frame, crop_box)
                
                # Calculate quality score
                quality = self.cropper.calculate_quality_score(
                    (self.frame_height, self.frame_width),
                    crop_box,
                    subject['bbox']
                )
                
                # Save frame only if quality is acceptable
                if quality > 0.3:
                    self.save_cropped_frame(cropped_frame, category, frame_count, quality)
                    self.stats['saved_frames'] += 1
                    self.stats[f'{category}_frames'] += 1
        
        finally:
            self.cap.release()
            print(f"\n✅ Processing complete!")
            self.print_stats()
        
        return self.stats
    
    def save_cropped_frame(self, 
                          frame: np.ndarray, 
                          category: str, 
                          frame_number: int,
                          quality: float):
        """
        Save cropped frame to appropriate directory
        
        Args:
            frame: Cropped frame
            category: Category ('person', 'animal', 'object')
            frame_number: Original frame number
            quality: Quality score
        """
        # Determine output directory
        if category == 'person':
            output_dir = self.person_dir
        elif category == 'animal':
            output_dir = self.animal_dir
        else:
            output_dir = self.object_dir
        
        # Generate filename with quality score
        filename = f"frame_{frame_number:06d}_q{int(quality*100)}.jpg"
        output_path = output_dir / filename
        
        # Save with high quality
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("📊 PROCESSING STATISTICS")
        print("="*50)
        print(f"Processed frames:    {self.stats['processed_frames']}")
        print(f"Saved frames:        {self.stats['saved_frames']}")
        print(f"  └─ Persons:        {self.stats['person_frames']}")
        print(f"  └─ Animals:        {self.stats['animal_frames']}")
        print(f"  └─ Objects:        {self.stats['object_frames']}")
        print(f"Skipped (text):      {self.stats['skipped_text']}")
        print(f"Skipped (no detect): {self.stats['skipped_no_detection']}")
        print("="*50)
    
    def extract_single_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame from video
        
        Args:
            frame_number: Frame number to extract
            
        Returns:
            Frame or None
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    def get_video_info(self) -> Dict:
        """
        Get video information
        
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(self.video_path)
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0
        }
        
        if info['fps'] > 0:
            info['duration'] = info['total_frames'] / info['fps']
        
        cap.release()
        
        return info
