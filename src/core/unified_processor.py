"""
Unified Video Processor v2.0
Combines all features: normal, ensemble, optimized, turbo, and batch processing
Now with Quality Analysis and Auto Captioning support
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List, Union, Any


class UnifiedVideoProcessor:
    """
    All-in-one video processor with:
    - Single model or ensemble detection
    - Standard or turbo mode processing
    - Batch video processing
    - GPU optimization
    - Progress tracking
    - V2.0: Quality analysis and filtering
    - V2.0: Auto captioning (BLIP + WD14)
    """
    
    def __init__(self, 
                 video_paths: Union[str, List[str]],
                 output_dir: str,
                 detector,
                 text_detector,
                 cropper,
                 use_turbo: bool = True,
                 batch_size: int = 8,
                 quality_analyzer: Any = None,
                 captioner: Any = None):
        """
        Initialize unified processor
        
        Args:
            video_paths: Single video path or list of video paths
            output_dir: Base output directory
            detector: ObjectDetector or EnsembleDetector instance
            text_detector: SubtitleDetector instance
            cropper: SmartCropper instance
            use_turbo: Enable turbo mode (batch frame processing)
            batch_size: Number of frames to process in parallel (default 8 for modern GPUs)
            quality_analyzer: V2.0 QualityAnalyzer instance (optional)
            captioner: V2.0 AdvancedCaptioner instance (optional)
        """
        # Handle single video or multiple videos
        if isinstance(video_paths, str):
            self.video_paths = [video_paths]
        else:
            self.video_paths = video_paths
        
        self.output_dir = output_dir
        self.detector = detector
        self.text_detector = text_detector
        self.cropper = cropper
        self.use_turbo = use_turbo
        self.batch_size = batch_size
        
        # V2.0 components
        self.quality_analyzer = quality_analyzer
        self.captioner = captioner
        
        # Check if using ensemble mode
        self.is_ensemble = hasattr(detector, 'models_to_use')
        
        # Video properties (will be set per video)
        self.cap = None
        self.current_video = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        
        # Overall stats for all videos
        self.overall_stats = {
            'total_videos': len(self.video_paths),
            'processed_videos': 0,
            'total_frames_saved': 0,
            'videos_stats': []
        }
        
        # Current video stats
        self.stats = self._create_empty_stats()
        
        # Performance tracking
        self.start_time = 0
        
        # FP16 support
        self.use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        if self.use_fp16 and self.use_turbo:
            print("🚀 FP16 mode enabled (faster inference)")
        
        print("="*60)
        print("🌾 UNIFIED VIDEO PROCESSOR v2.0")
        print("="*60)
        print(f"📹 Videos to process: {len(self.video_paths)}")
        print(f"🤖 Mode: {'Ensemble' if self.is_ensemble else 'Single Model'}")
        print(f"⚡ Turbo: {'Enabled' if self.use_turbo else 'Disabled'}")
        if self.use_turbo:
            print(f"📦 Batch size: {self.batch_size}")
        
        # V2.0 features
        if self.quality_analyzer:
            print(f"🔍 Quality Analysis: Enabled")
        if self.captioner:
            print(f"📝 Auto Captioning: Enabled")
        print("="*60)
    
    def _create_empty_stats(self) -> Dict:
        """Create empty stats dictionary"""
        return {
            'processed_frames': 0,
            'saved_frames': 0,
            'skipped_text': 0,
            'skipped_no_detection': 0,
            'skipped_quality': 0,  # V2.0
            'captioned_frames': 0,  # V2.0
            'person_frames': 0,
            'animal_frames': 0,
            'object_frames': 0,
            'processing_time': 0
        }
    
    def create_output_structure(self, video_name: str) -> Path:
        """Create output directory structure for a video"""
        mode_suffix = "ensemble" if self.is_ensemble else "yolo"
        turbo_suffix = "_turbo" if self.use_turbo else ""
        aspect_ratio = self.cropper.target_format.replace(':', 'x')
        
        base_path = Path(self.output_dir) / f"{video_name}_{aspect_ratio}_{mode_suffix}{turbo_suffix}"
        base_path.mkdir(parents=True, exist_ok=True)
        
        self.person_dir = base_path / 'persons'
        self.animal_dir = base_path / 'animals'
        self.object_dir = base_path / 'objects'
        
        self.person_dir.mkdir(exist_ok=True)
        self.animal_dir.mkdir(exist_ok=True)
        self.object_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output: {base_path}")
        return base_path
    
    def open_video(self, video_path: str) -> bool:
        """Open video file and get properties"""
        self.current_video = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open: {video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"\n🎬 Video: {Path(video_path).name}")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   FPS: {self.fps:.1f}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Total frames: {self.total_frames}")
        
        return True
    
    def process_all_videos(self,
                          frame_interval: int = 30,
                          skip_text: bool = True,
                          use_quick_text_check: bool = True,
                          progress_callback: Optional[Callable] = None,
                          stop_callback: Optional[Callable] = None) -> Dict:
        """
        Process all videos in the list
        
        Args:
            frame_interval: Process every Nth frame
            skip_text: Skip frames with subtitles
            use_quick_text_check: Use fast text detection
            progress_callback: Callback for progress updates
            stop_callback: Callback to check if processing should stop
            
        Returns:
            Overall statistics for all videos
        """
        total_start = time.time()
        
        for idx, video_path in enumerate(self.video_paths, 1):
            print(f"\n{'='*60}")
            print(f"Processing video {idx}/{len(self.video_paths)}")
            print(f"{'='*60}")
            
            # Check stop signal
            if stop_callback and stop_callback():
                print("\n⏹️  Batch processing stopped by user")
                break
            
            # Process single video
            video_stats = self.process_single_video(
                video_path,
                frame_interval,
                skip_text,
                use_quick_text_check,
                progress_callback,
                stop_callback
            )
            
            # Update overall stats
            self.overall_stats['processed_videos'] += 1
            self.overall_stats['total_frames_saved'] += video_stats['saved_frames']
            self.overall_stats['videos_stats'].append({
                'video_name': Path(video_path).name,
                'stats': video_stats
            })
        
        total_elapsed = time.time() - total_start
        self.overall_stats['total_time'] = total_elapsed
        
        # Print overall summary
        self.print_overall_summary()
        
        return self.overall_stats
    
    def process_single_video(self,
                            video_path: str,
                            frame_interval: int = 30,
                            skip_text: bool = True,
                            use_quick_text_check: bool = True,
                            progress_callback: Optional[Callable] = None,
                            stop_callback: Optional[Callable] = None) -> Dict:
        """Process a single video"""
        if not self.open_video(video_path):
            return self._create_empty_stats()
        
        # Create output structure for this video
        video_name = Path(video_path).stem
        self.create_output_structure(video_name)
        
        # Reset stats for this video
        self.stats = self._create_empty_stats()
        self.start_time = time.time()
        
        try:
            if self.use_turbo:
                self._process_video_turbo(
                    frame_interval, skip_text, use_quick_text_check,
                    progress_callback, stop_callback
                )
            else:
                self._process_video_standard(
                    frame_interval, skip_text, use_quick_text_check,
                    progress_callback, stop_callback
                )
        finally:
            if self.cap:
                self.cap.release()
            
            elapsed = time.time() - self.start_time
            self.stats['processing_time'] = elapsed
            fps = self.stats['processed_frames'] / elapsed if elapsed > 0 else 0
            
            print(f"\n✅ Video complete!")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Processing speed: {fps:.1f} FPS")
            self.print_video_stats()
        
        return self.stats
    
    def _process_video_standard(self,
                               frame_interval: int,
                               skip_text: bool,
                               use_quick_text: bool,
                               progress_callback: Optional[Callable],
                               stop_callback: Optional[Callable]):
        """Standard video processing (frame by frame)"""
        frame_count = 0
        
        while True:
            if stop_callback and stop_callback():
                break
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % frame_interval != 0:
                continue
            
            self.stats['processed_frames'] += 1
            
            # Progress callback
            if progress_callback and frame_count % 30 == 0:
                progress = (frame_count / self.total_frames) * 100
                progress_callback(progress, self.stats)
            
            # Process frame
            self._process_single_frame(frame, frame_count, skip_text, use_quick_text)
    
    def _process_video_turbo(self,
                            frame_interval: int,
                            skip_text: bool,
                            use_quick_text: bool,
                            progress_callback: Optional[Callable],
                            stop_callback: Optional[Callable]):
        """Turbo video processing (batch frames)"""
        frame_count = 0
        frame_batch = []
        frame_numbers = []
        
        while True:
            if stop_callback and stop_callback():
                if frame_batch:
                    self._process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                break
            
            ret, frame = self.cap.read()
            
            if not ret:
                if frame_batch:
                    self._process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                break
            
            frame_count += 1
            
            if frame_count % frame_interval != 0:
                continue
            
            frame_batch.append(frame)
            frame_numbers.append(frame_count)
            
            if len(frame_batch) >= self.batch_size:
                self._process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                frame_batch = []
                frame_numbers = []
            
            if progress_callback and frame_count % 30 == 0:
                progress = (frame_count / self.total_frames) * 100
                progress_callback(progress, self.stats)
    
    def _process_batch(self, frames: List[np.ndarray], frame_numbers: List[int],
                      skip_text: bool, use_quick_text: bool):
        """Process a batch of frames with GPU batch detection"""
        batch_size = len(frames)
        
        # Quick text check - filter first
        valid_frames = []
        valid_frame_nums = []
        
        if skip_text and self.text_detector:
            for i, (frame, frame_num) in enumerate(zip(frames, frame_numbers)):
                has_text = False
                if use_quick_text:
                    has_text = self.text_detector.quick_text_check(frame)
                else:
                    has_text, _ = self.text_detector.has_text(frame)
                
                if has_text:
                    self.stats['skipped_text'] += 1
                else:
                    valid_frames.append(frame)
                    valid_frame_nums.append(frame_num)
        else:
            valid_frames = frames
            valid_frame_nums = frame_numbers
        
        if not valid_frames:
            return
        
        # V2.0: Batch quality check if enabled
        quality_mask = [True] * len(valid_frames)
        if self.quality_analyzer:
            for i, frame in enumerate(valid_frames):
                is_quality_ok, _ = self.quality_analyzer.check_frame_quality(frame)
                if not is_quality_ok:
                    quality_mask[i] = False
                    self.stats['skipped_quality'] += 1
        
        # Filter by quality
        final_frames = [f for f, ok in zip(valid_frames, quality_mask) if ok]
        final_frame_nums = [n for n, ok in zip(valid_frame_nums, quality_mask) if ok]
        
        if not final_frames:
            return
        
        # GPU BATCH DETECTION - all frames at once!
        if hasattr(self.detector, 'detect_batch'):
            all_detections = self.detector.detect_batch(final_frames)
        else:
            all_detections = [self.detector.detect(f) for f in final_frames]
        
        # Process results
        for frame, frame_num, detections in zip(final_frames, final_frame_nums, all_detections):
            self.stats['processed_frames'] += 1
            self._process_frame_with_detection(frame, frame_num, detections)
    
    def _process_frame_with_detection(self, frame: np.ndarray, frame_number: int, 
                                       detections: Dict):
        """Process frame with pre-computed detections"""
        # Get primary subject
        category, subject = self.detector.get_primary_subject(detections)
        
        if category is None:
            self.stats['skipped_no_detection'] += 1
            return
        
        # Calculate head space
        head_space = 0.0
        if category == 'person':
            head_space = self.detector.calculate_head_space(
                subject['bbox'], 
                self.frame_height
            )
        
        # Calculate crop
        crop_box = self.cropper.calculate_crop_box(
            (self.frame_height, self.frame_width),
            subject['bbox'],
            category,
            head_space
        )
        
        if crop_box is None:
            return
        
        # Apply crop
        cropped = self.cropper.apply_crop(frame, crop_box)
        
        # Quality score
        quality = self.cropper.calculate_quality_score(
            (self.frame_height, self.frame_width),
            crop_box,
            subject['bbox']
        )
        
        if quality > 0.3:
            self.save_cropped_frame(cropped, category, frame_number, quality)
            self.stats['saved_frames'] += 1
            self.stats[f'{category}_frames'] += 1
    
    def _process_single_frame(self, frame: np.ndarray, frame_number: int,
                             skip_text: bool, use_quick_text: bool):
        """Process a single frame with V2.0 quality and captioning support"""
        # Skip text if needed (for non-batch processing)
        if skip_text and self.text_detector:
            has_text = False
            if use_quick_text:
                has_text = self.text_detector.quick_text_check(frame)
            else:
                has_text, _ = self.text_detector.has_text(frame)
            
            if has_text:
                self.stats['skipped_text'] += 1
                return
        
        # V2.0: Quality check before processing
        if self.quality_analyzer:
            is_quality_ok, quality_info = self.quality_analyzer.check_frame_quality(frame)
            if not is_quality_ok:
                self.stats['skipped_quality'] += 1
                return
        
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Get primary subject
        category, subject = self.detector.get_primary_subject(detections)
        
        if category is None:
            self.stats['skipped_no_detection'] += 1
            return
        
        # Calculate head space
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
            return
        
        # Apply crop
        cropped = self.cropper.apply_crop(frame, crop_box)
        
        # Quality check
        quality = self.cropper.calculate_quality_score(
            (self.frame_height, self.frame_width),
            crop_box,
            subject['bbox']
        )
        
        if quality > 0.3:
            saved_path = self.save_cropped_frame(cropped, category, frame_number, quality)
            self.stats['saved_frames'] += 1
            self.stats[f'{category}_frames'] += 1
            
            # V2.0: Auto captioning
            if self.captioner and saved_path:
                try:
                    # AdvancedCaptioner uses caption_image() method
                    result = self.captioner.caption_image(cropped)
                    caption = result.final_caption if hasattr(result, 'final_caption') else str(result)
                    caption_path = saved_path.with_suffix('.txt')
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    self.stats['captioned_frames'] += 1
                except Exception as e:
                    print(f"⚠️ Caption error for frame {frame_number}: {e}")
    
    def save_cropped_frame(self, frame: np.ndarray, category: str,
                          frame_number: int, quality: float) -> Optional[Path]:
        """Save cropped frame to appropriate directory and return path"""
        if category == 'person':
            output_dir = self.person_dir
        elif category == 'animal':
            output_dir = self.animal_dir
        else:
            output_dir = self.object_dir
        
        filename = f"frame_{frame_number:06d}_q{int(quality*100)}.jpg"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return output_path
    
    def print_video_stats(self):
        """Print statistics for current video"""
        print("\n" + "="*50)
        print("📊 VIDEO STATISTICS")
        print("="*50)
        print(f"Processed frames:    {self.stats['processed_frames']}")
        print(f"Saved frames:        {self.stats['saved_frames']}")
        print(f"  └─ Persons:        {self.stats['person_frames']}")
        print(f"  └─ Animals:        {self.stats['animal_frames']}")
        print(f"  └─ Objects:        {self.stats['object_frames']}")
        print(f"Skipped (text):      {self.stats['skipped_text']}")
        print(f"Skipped (no detect): {self.stats['skipped_no_detection']}")
        print(f"Skipped (quality):   {self.stats['skipped_quality']}")  # V2.0
        print(f"Captioned frames:    {self.stats['captioned_frames']}")  # V2.0
        print("="*50)
    
    def print_overall_summary(self):
        """Print overall summary for all videos"""
        print("\n\n" + "="*60)
        print("🎉 BATCH PROCESSING COMPLETE!")
        print("="*60)
        print(f"📹 Total videos processed: {self.overall_stats['processed_videos']}/{self.overall_stats['total_videos']}")
        print(f"💾 Total frames saved: {self.overall_stats['total_frames_saved']}")
        print(f"⏱️  Total time: {self.overall_stats.get('total_time', 0):.1f}s")
        print("\n📊 Per-Video Breakdown:")
        print("-"*60)
        
        for video_stat in self.overall_stats['videos_stats']:
            name = video_stat['video_name']
            stats = video_stat['stats']
            print(f"\n📹 {name}")
            print(f"   Saved: {stats['saved_frames']} frames")
            print(f"   Persons: {stats['person_frames']}, Animals: {stats['animal_frames']}, Objects: {stats['object_frames']}")
            print(f"   Time: {stats['processing_time']:.1f}s")
        
        print("="*60)
    
    def get_overall_stats(self) -> Dict:
        """Get overall statistics"""
        return self.overall_stats
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video information without opening for processing"""
        cap = cv2.VideoCapture(video_path)
        
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
