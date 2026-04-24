"""
Enhanced Video Processor with Advanced Features
Includes: Quality Analysis, Scene Detection, Checkpoint/Resume, Async I/O
"""

import cv2
import json
import logging
import numpy as np
import torch
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, List, Union
from queue import Queue, Empty
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

from src.core.quality_analyzer import QualityAnalyzer, SceneChangeDetector


@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for resume functionality"""
    video_path: str
    last_frame: int
    total_frames: int
    stats: Dict
    timestamp: str
    output_dir: str
    settings: Dict


class AsyncFrameSaver:
    """
    Asynchronous frame saver to avoid I/O blocking
    Uses a queue and background thread for saving
    """
    
    def __init__(self, num_workers: int = 4, jpeg_quality: int = 95):
        self.num_workers = num_workers
        self.jpeg_quality = jpeg_quality
        self.queue: Queue = Queue(maxsize=100)
        self.workers: List[threading.Thread] = []
        self.running = False
        self.saved_count = 0
        self.error_count = 0
        self._lock = threading.Lock()
    
    def start(self):
        """Start background save workers"""
        self.running = True
        self.saved_count = 0
        self.error_count = 0
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._save_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info("Async saver started with %d workers", self.num_workers)
    
    def stop(self):
        """Stop workers and wait for queue to empty"""
        # Wait for queue to empty
        self.queue.join()
        self.running = False
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
    
    def save(self, frame: np.ndarray, filepath: str):
        """Queue frame for saving"""
        if self.running:
            self.queue.put((frame.copy(), filepath))
    
    def _save_worker(self):
        """Background worker for saving frames"""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        
        while self.running or not self.queue.empty():
            try:
                frame, filepath = self.queue.get(timeout=0.5)
                
                try:
                    cv2.imwrite(filepath, frame, encode_params)
                    with self._lock:
                        self.saved_count += 1
                except Exception as e:
                    with self._lock:
                        self.error_count += 1
                finally:
                    self.queue.task_done()
                    
            except Empty:
                continue
    
    def get_stats(self) -> Dict:
        """Get save statistics"""
        return {
            'saved': self.saved_count,
            'errors': self.error_count,
            'pending': self.queue.qsize()
        }


class FramePrefetcher:
    """
    Prefetches frames in background thread to reduce video read latency
    """
    
    def __init__(self, cap: cv2.VideoCapture, buffer_size: int = 30, 
                 frame_interval: int = 1):
        self.cap = cap
        self.buffer_size = buffer_size
        self.frame_interval = frame_interval
        self.buffer: Queue = Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.current_frame_num = 0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def start(self, start_frame: int = 0):
        """Start prefetching from given frame"""
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        self.current_frame_num = start_frame
        self.running = True
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop prefetching"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Clear buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except Empty:
                break
    
    def get_frame(self, timeout: float = 5.0) -> Optional[tuple]:
        """
        Get next frame from buffer
        Returns: (frame_number, frame) or None if done
        """
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None
    
    def _prefetch_worker(self):
        """Background prefetch worker"""
        frame_count = self.current_frame_num
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                # Signal end of video
                self.buffer.put(None)
                break
            
            frame_count += 1
            
            # Only queue frames at interval
            if frame_count % self.frame_interval == 0:
                try:
                    self.buffer.put((frame_count, frame), timeout=1.0)
                except Exception:
                    if not self.running:
                        break


class EnhancedVideoProcessor:
    """
    Enhanced video processor with:
    - Quality analysis (blur, lighting, duplicates)
    - Scene change detection
    - Checkpoint/Resume support
    - Async frame saving
    - Frame prefetching
    - Detailed statistics
    """
    
    def __init__(self, 
                 video_paths: Union[str, List[str]],
                 output_dir: str,
                 detector,
                 text_detector,
                 cropper,
                 use_turbo: bool = True,
                 batch_size: int = 4,
                 enable_quality_check: bool = True,
                 enable_scene_detection: bool = False,
                 jpeg_quality: int = 95):
        """
        Initialize enhanced processor
        
        Args:
            video_paths: Single video or list of videos
            output_dir: Output directory
            detector: Object detector instance
            text_detector: Text detector instance
            cropper: SmartCropper instance
            use_turbo: Enable turbo mode
            batch_size: Batch size for turbo mode
            enable_quality_check: Enable quality analysis
            enable_scene_detection: Use scene detection instead of fixed interval
            jpeg_quality: JPEG quality (1-100)
        """
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
        self.enable_quality_check = enable_quality_check
        self.enable_scene_detection = enable_scene_detection
        self.jpeg_quality = jpeg_quality
        
        # Check if ensemble mode
        self.is_ensemble = hasattr(detector, 'models_to_use')
        
        # Initialize quality analyzer
        self.quality_analyzer = QualityAnalyzer(
            blur_threshold=80.0,
            brightness_range=(35, 225),
            duplicate_threshold=0.90
        ) if enable_quality_check else None
        
        # Scene detector
        self.scene_detector = SceneChangeDetector(
            threshold=25.0,
            min_scene_frames=10
        ) if enable_scene_detection else None
        
        # Async saver — 4 workers for better write throughput
        self.async_saver = AsyncFrameSaver(
            num_workers=4,
            jpeg_quality=jpeg_quality
        )
        
        # Video properties
        self.cap = None
        self.current_video = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        
        # Overall stats
        self.overall_stats = {
            'total_videos': len(self.video_paths),
            'processed_videos': 0,
            'total_frames_saved': 0,
            'videos_stats': []
        }
        
        # Current video stats
        self.stats = self._create_empty_stats()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(output_dir) / '.checkpoints'
        
        # FP16 support
        self.use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        # Print configuration
        print("="*60)
        print("🌾 ENHANCED VIDEO PROCESSOR")
        print("="*60)
        print(f"📹 Videos: {len(self.video_paths)}")
        print(f"🤖 Mode: {'Ensemble' if self.is_ensemble else 'Single Model'}")
        print(f"⚡ Turbo: {'✓' if self.use_turbo else '✗'}")
        print(f"🔍 Quality Check: {'✓' if enable_quality_check else '✗'}")
        print(f"🎬 Scene Detection: {'✓' if enable_scene_detection else '✗'}")
        print(f"💾 Async Save: ✓")
        print(f"🖼️ JPEG Quality: {jpeg_quality}%")
        print("="*60)
    
    def _create_empty_stats(self) -> Dict:
        """Create empty stats dictionary"""
        return {
            'processed_frames': 0,
            'saved_frames': 0,
            'skipped_text': 0,
            'skipped_no_detection': 0,
            'skipped_quality': 0,
            'skipped_duplicate': 0,
            'person_frames': 0,
            'animal_frames': 0,
            'object_frames': 0,
            'scene_changes': 0,
            'avg_quality_score': 0.0,
            'processing_time': 0
        }
    
    def create_output_structure(self, video_name: str) -> Path:
        """Create output directory structure"""
        mode_suffix = "ensemble" if self.is_ensemble else "yolo"
        quality_suffix = "_hq" if self.enable_quality_check else ""
        aspect_ratio = self.cropper.target_format.replace(':', 'x')
        
        base_path = Path(self.output_dir) / f"{video_name}_{aspect_ratio}_{mode_suffix}{quality_suffix}"
        base_path.mkdir(parents=True, exist_ok=True)
        
        self.person_dir = base_path / 'persons'
        self.animal_dir = base_path / 'animals'
        self.object_dir = base_path / 'objects'
        
        self.person_dir.mkdir(exist_ok=True)
        self.animal_dir.mkdir(exist_ok=True)
        self.object_dir.mkdir(exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output: {base_path}")
        return base_path
    
    def save_checkpoint(self, video_path: str, frame_num: int, settings: Dict):
        """Save processing checkpoint for resume"""
        checkpoint = ProcessingCheckpoint(
            video_path=video_path,
            last_frame=frame_num,
            total_frames=self.total_frames,
            stats=self.stats.copy(),
            timestamp=datetime.now().isoformat(),
            output_dir=self.output_dir,
            settings=settings
        )
        
        video_name = Path(video_path).stem
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.checkpoint_dir / f"{video_name}.checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2)
        except Exception as e:
            logger.warning("Checkpoint save error: %s", e)
    
    def load_checkpoint(self, video_path: str) -> Optional[ProcessingCheckpoint]:
        """Load checkpoint if exists"""
        video_name = Path(video_path).stem
        checkpoint_file = self.checkpoint_dir / f"{video_name}.checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            checkpoint = ProcessingCheckpoint(**data)
            
            # Check if same video
            if checkpoint.video_path == video_path:
                logger.info("Found checkpoint at frame %d/%d", checkpoint.last_frame, checkpoint.total_frames)
                return checkpoint

        except Exception as e:
            logger.warning("Checkpoint load error: %s", e)
        
        return None
    
    def clear_checkpoint(self, video_path: str):
        """Clear checkpoint after successful completion"""
        video_name = Path(video_path).stem
        checkpoint_file = self.checkpoint_dir / f"{video_name}.checkpoint.json"
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    def open_video(self, video_path: str) -> bool:
        """Open video file"""
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
        print(f"   Duration: {duration:.1f}s ({self.total_frames} frames)")
        
        return True
    
    def process_frame(self, frame: np.ndarray, frame_num: int, 
                      skip_text: bool = True) -> Optional[str]:
        """
        Process a single frame
        
        Returns:
            Category of saved frame or None if skipped
        """
        # 1. Text check
        if skip_text and self.text_detector:
            if self.text_detector.quick_text_check(frame):
                self.stats['skipped_text'] += 1
                return None
        
        # 2. Quality check
        if self.quality_analyzer:
            is_quality_ok, analysis = self.quality_analyzer.analyze_frame(frame)
            
            if not is_quality_ok:
                reason = analysis.get('rejection_reason', 'unknown')
                if reason == 'duplicate':
                    self.stats['skipped_duplicate'] += 1
                else:
                    self.stats['skipped_quality'] += 1
                return None
            
            # Track quality scores
            self._update_avg_quality(analysis['quality_score'])
        
        # 3. Object detection
        detections = self.detector.detect(frame)
        category, subject = self.detector.get_primary_subject(detections)
        
        if category is None:
            self.stats['skipped_no_detection'] += 1
            return None
        
        # 4. Calculate crop
        head_space = 0.0
        if category == 'person':
            head_space = self.detector.calculate_head_space(
                subject['bbox'], 
                self.frame_height
            )
        
        crop_box = self.cropper.calculate_crop_box(
            (self.frame_height, self.frame_width),
            subject['bbox'],
            category,
            head_space
        )
        
        if crop_box is None:
            return None
        
        # 5. Apply crop
        cropped = self.cropper.apply_crop(frame, crop_box)
        
        # 6. Save frame (async)
        self._save_frame(cropped, category, frame_num)
        
        return category
    
    def _update_avg_quality(self, score: float):
        """Update running average quality score"""
        n = self.stats['processed_frames']
        if n == 0:
            self.stats['avg_quality_score'] = score
        else:
            # Incremental average
            self.stats['avg_quality_score'] = (
                self.stats['avg_quality_score'] * n + score
            ) / (n + 1)
    
    def _save_frame(self, frame: np.ndarray, category: str, frame_num: int):
        """Save frame using async saver"""
        # Determine output directory
        if category == 'person':
            output_dir = self.person_dir
            self.stats['person_frames'] += 1
        elif category == 'animal':
            output_dir = self.animal_dir
            self.stats['animal_frames'] += 1
        else:
            output_dir = self.object_dir
            self.stats['object_frames'] += 1
        
        # Generate filename
        timestamp = datetime.now().strftime('%H%M%S%f')[:10]
        filename = f"frame_{frame_num:06d}_{timestamp}.jpg"
        filepath = str(output_dir / filename)
        
        # Queue for async save
        self.async_saver.save(frame, filepath)
        self.stats['saved_frames'] += 1
    
    def process_single_video(self,
                            video_path: str,
                            frame_interval: int = 30,
                            skip_text: bool = True,
                            progress_callback: Optional[Callable] = None,
                            stop_callback: Optional[Callable] = None,
                            resume: bool = True) -> Dict:
        """
        Process a single video with all enhancements
        """
        if not self.open_video(video_path):
            return self.stats
        
        # Reset stats
        self.stats = self._create_empty_stats()
        
        # Check for checkpoint
        start_frame = 0
        settings = {
            'frame_interval': frame_interval,
            'skip_text': skip_text,
            'quality_check': self.enable_quality_check,
            'scene_detection': self.enable_scene_detection
        }
        
        if resume:
            checkpoint = self.load_checkpoint(video_path)
            if checkpoint:
                response = input(f"Resume from frame {checkpoint.last_frame}? (y/n): ")
                if response.lower() == 'y':
                    start_frame = checkpoint.last_frame
                    self.stats = checkpoint.stats
                    print(f"▶️ Resuming from frame {start_frame}")
        
        # Create output structure
        video_name = Path(video_path).stem
        self.create_output_structure(video_name)
        
        # Clear quality analyzer history
        if self.quality_analyzer:
            self.quality_analyzer.clear_history()
        
        # Reset scene detector
        if self.scene_detector:
            self.scene_detector.reset()
        
        # Start async saver
        self.async_saver.start()
        
        # Start time
        start_time = time.time()
        
        # Seek to start position
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        checkpoint_interval = 500  # Save checkpoint every 500 frames
        
        try:
            while True:
                # Check stop
                if stop_callback and stop_callback():
                    print("\n⏹️ Stopped by user")
                    self.save_checkpoint(video_path, frame_count, settings)
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Scene detection mode
                if self.enable_scene_detection and self.scene_detector:
                    if not self.scene_detector.is_scene_change(frame):
                        continue
                    self.stats['scene_changes'] += 1
                else:
                    # Fixed interval mode
                    if frame_count % frame_interval != 0:
                        continue
                
                self.stats['processed_frames'] += 1
                
                # Process frame
                category = self.process_frame(frame, frame_count, skip_text)
                
                # Progress callback
                if progress_callback:
                    progress = (frame_count / self.total_frames) * 100
                    progress_callback(progress, self.stats)
                
                # Periodic checkpoint
                if frame_count % checkpoint_interval == 0:
                    self.save_checkpoint(video_path, frame_count, settings)
        
        finally:
            # Stop async saver
            self.async_saver.stop()
            
            # Release video
            self.cap.release()
            
            # Calculate time
            elapsed = time.time() - start_time
            self.stats['processing_time'] = elapsed
            
            # Clear checkpoint on successful completion
            if frame_count >= self.total_frames - 10:
                self.clear_checkpoint(video_path)
            
            # Print stats
            self._print_stats(elapsed)
        
        return self.stats
    
    def process_all_videos(self,
                          frame_interval: int = 30,
                          skip_text: bool = True,
                          progress_callback: Optional[Callable] = None,
                          stop_callback: Optional[Callable] = None) -> Dict:
        """Process all videos"""
        total_start = time.time()
        
        for idx, video_path in enumerate(self.video_paths, 1):
            print(f"\n{'='*60}")
            print(f"📹 Processing video {idx}/{len(self.video_paths)}")
            print(f"{'='*60}")
            
            if stop_callback and stop_callback():
                print("\n⏹️ Batch stopped")
                break
            
            # Process video
            video_stats = self.process_single_video(
                video_path,
                frame_interval,
                skip_text,
                progress_callback,
                stop_callback,
                resume=True
            )
            
            # Update overall stats
            self.overall_stats['processed_videos'] += 1
            self.overall_stats['total_frames_saved'] += video_stats['saved_frames']
            self.overall_stats['videos_stats'].append({
                'video': Path(video_path).name,
                'stats': video_stats
            })
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*60}")
        print("🎉 ALL VIDEOS COMPLETE!")
        print(f"{'='*60}")
        print(f"📹 Videos processed: {self.overall_stats['processed_videos']}/{len(self.video_paths)}")
        print(f"🖼️ Total frames saved: {self.overall_stats['total_frames_saved']}")
        print(f"⏱️ Total time: {total_time:.1f}s")
        
        return self.overall_stats
    
    def _print_stats(self, elapsed: float):
        """Print processing statistics"""
        fps = self.stats['processed_frames'] / elapsed if elapsed > 0 else 0
        
        print(f"\n{'─'*40}")
        print("📊 Processing Statistics:")
        print(f"   Processed frames: {self.stats['processed_frames']}")
        print(f"   Saved frames: {self.stats['saved_frames']}")
        print(f"   ├─ Persons: {self.stats['person_frames']}")
        print(f"   ├─ Animals: {self.stats['animal_frames']}")
        print(f"   └─ Objects: {self.stats['object_frames']}")
        print(f"   Skipped (text): {self.stats['skipped_text']}")
        print(f"   Skipped (no detection): {self.stats['skipped_no_detection']}")
        
        if self.enable_quality_check:
            print(f"   Skipped (quality): {self.stats['skipped_quality']}")
            print(f"   Skipped (duplicate): {self.stats['skipped_duplicate']}")
            print(f"   Avg quality score: {self.stats['avg_quality_score']:.1f}/100")
        
        if self.enable_scene_detection:
            print(f"   Scene changes: {self.stats['scene_changes']}")
        
        print(f"   Processing speed: {fps:.1f} frames/sec")
        print(f"   Total time: {elapsed:.1f}s")
        print(f"{'─'*40}")
        
        # Quality analyzer stats
        if self.quality_analyzer:
            self.quality_analyzer.print_stats()
