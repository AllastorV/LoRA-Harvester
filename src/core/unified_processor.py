"""
Unified Video Processor v3.0
Combines all features: normal, ensemble, optimized, turbo, and batch processing
Now with Quality Analysis and Auto Captioning support
"""

import cv2
import logging
import numpy as np
import torch
import time
import threading
import os
import tempfile
import hashlib
import copy
import math
from src.core.video_sampling import SelectiveVideoReader
from pathlib import Path
from typing import Optional, Callable, Dict, List, Union, Any

logger = logging.getLogger(__name__)


def _imwrite_unicode(path: str, frame: np.ndarray, ext: str = ".png") -> bool:
    """cv2.imwrite that survives non-ASCII Windows paths.

    Raw cv2.imwrite fails silently when the path contains non-ASCII chars
    on Windows. imencode + numpy.tofile sidesteps the C++ fopen() path.
    """
    try:
        ok, buf = cv2.imencode(ext, frame)
        if not ok:
            return False
        # Stage in the existing destination directory; failed writes never
        # expose a partial PNG and never create an accidental output directory.
        fd, tmp = tempfile.mkstemp(prefix='.lh-frame-', suffix='.tmp', dir=str(Path(path).parent))
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(buf.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return True
    except Exception as e:
        logger.warning("imwrite failed for %s: %s", path, e)
        return False


class UnifiedVideoProcessor:
    """
    All-in-one video processor with:
    - Single model or ensemble detection
    - Standard or turbo mode processing
    - Batch video processing
    - GPU optimization
    - Progress tracking
    - V2.0: Quality analysis and filtering
    - V2.0: Auto captioning (WD14)
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
                 captioner: Any = None,
                 caption_mode: str = "tags_only",
                 florence2: Any = None,
                 florence2_task: str = "<DETAILED_CAPTION>",
                 log_callback: Optional[Callable] = None,
                 jpeg_quality: int = 95,
                 upscaler: Any = None,
                 upscale_target: str = "crop",
                 upscale_min_resolution: int = 512,
                 upscale_max_resolution: int = 0,
                 subtitle_removal: bool = False,
                 nsfw_detector: Any = None,
                 nsfw_uncertain_folder: bool = True):
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
            caption_mode: Caption mode to use (tags_only)
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        # Handle single video or multiple videos
        if isinstance(video_paths, str):
            self.video_paths = [video_paths]
        else:
            self.video_paths = video_paths

        # Cap cv2 video decode threads to avoid stealing all CPU from UI.
        # Default 0 = OpenCV picks; we clamp to 4 as a sane ceiling.
        _cv2_threads = int(batch_size / 2) if batch_size > 2 else 2
        cv2.setNumThreads(min(_cv2_threads, 4))

        self.output_dir = output_dir
        self.detector = detector
        self.text_detector = text_detector
        self.cropper = cropper
        self.use_turbo = use_turbo
        self.batch_size = batch_size
        # Smallest batch size we'll auto-shrink down to on OOM before
        # giving up. 1 effectively disables turbo but still produces
        # correct output.
        self._min_batch_size = 1
        # Set by process_all_videos. Kept on the instance so the save
        # helpers don't need new arguments.
        self._frame_saved_callback: Optional[Callable[[str], None]] = None
        
        # V2.0 components
        self.quality_analyzer = quality_analyzer
        self.captioner = captioner
        self.caption_mode = caption_mode
        self.florence2 = florence2
        self.florence2_task = florence2_task
        self.log_callback = log_callback
        self.jpeg_quality = max(1, min(100, jpeg_quality))

        # V3.x upscaler
        self.upscaler = upscaler
        self.upscale_target = upscale_target      # "crop" or "frame"
        self.upscale_min_resolution = upscale_min_resolution
        # 0 = off. If an upscaled frame's longest side exceeds this, it is
        # downscaled back to this cap (keeps output within a resolution budget).
        self.upscale_max_resolution = max(0, int(upscale_max_resolution or 0))

        # V3.x subtitle removal (inpaint instead of skip)
        self.subtitle_removal = subtitle_removal

        # V3.x NSFW detection — separate saves into sfw/nsfw/(uncertain) subdirs
        self.nsfw_detector = nsfw_detector
        self.nsfw_uncertain_folder = nsfw_uncertain_folder
        # A bounded buffer of FINAL accepted crops, never raw video frames.
        # It is enabled only within a video run and drained before changing
        # output directories, parking on pause, or returning on stop/error.
        self._pending_crops = []
        self._pending_crop_bytes = 0
        self._pending_crop_max_bytes = 64 * 1024 * 1024
        self._defer_nsfw_saves = False
        
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
        self._stage_seconds = {}
        self._video_reader = None

        # Progress emit throttle — avoid flooding the UI thread with
        # progress signals faster than it can animate (causes freeze).
        self._last_progress_emit = 0.0
        self._progress_min_interval = 0.12   # seconds (~8 updates/sec max)
        
        # FP16 support
        self.use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        if self.use_fp16 and self.use_turbo:
            print("🚀 FP16 mode enabled (faster inference)")
        
        print("="*60)
        print("🌾 UNIFIED VIDEO PROCESSOR v3.0")
        print("="*60)
        print(f"📹 Videos to process: {len(self.video_paths)}")
        print(f"🤖 Mode: {'Ensemble' if self.is_ensemble else 'Single Model'}")
        print(f"⚡ Turbo: {'Enabled' if self.use_turbo else 'Disabled'}")
        if self.use_turbo:
            print(f"📦 Batch size: {self.batch_size}")
        
        # V2.0 features
        if self.quality_analyzer:
            print(f"🔍 Quality Analysis: Enabled")
        if self.captioner or self.florence2:
            print(f"📝 Auto Captioning: Enabled (mode={self.caption_mode})")
        print("="*60)
    
    def _emit_progress(self, progress_callback, progress: float, force: bool = False):
        """
        Throttled progress emission. Emits at most ~8 times/second so the
        UI thread is never flooded with signals it can't animate (freeze fix).
        Pass force=True for the final emit to guarantee 100% shows.
        """
        if not progress_callback:
            return
        now = time.monotonic()
        if force or (now - self._last_progress_emit) >= self._progress_min_interval:
            self._last_progress_emit = now
            progress_callback(progress, dict(self.stats))

    def _log(self, msg: str):
        """Log message to callback (GUI) and logger"""
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass
        logger.info(msg)

    def _timed(self, stage, function, *args, **kwargs):
        """Measure host wall time, including failed attempts, without CUDA sync.

        These are operational timings, not isolated GPU kernel measurements.
        No profiler, image sampling, or model/threshold change is required.
        """
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self._stage_seconds[stage] = (self._stage_seconds.get(stage, 0.0)
                                          + time.perf_counter() - start)

    def _create_empty_stats(self) -> Dict:
        """Create empty stats dictionary"""
        return {
            'processed_frames': 0,
            'saved_frames': 0,
            'skipped_text': 0,
            'skipped_no_detection': 0,
            'skipped_quality': 0,      # V2.0
            'captioned_frames': 0,     # V2.0
            'upscaled_frames': 0,      # V3.x: frames upscaled by Real-ESRGAN
            'upscale_oom': 0,          # V3.x: OOM events during upscale
            'subtitle_removed': 0,     # V3.x: subtitle inpainted out
            'overlay_crops': 0,        # frames where overlay exclusion was applied
            'nsfw_frames': 0,          # V3.x: classified as nsfw
            'sfw_frames': 0,           # V3.x: classified as sfw
            'nsfw_uncertain': 0,       # V3.x: uncertain classification
            'nsfw_classification_time': 0.0,
            'nsfw_crops_analyzed': 0,
            'nsfw_batches': 0,
            'nsfw_max_batch': 0,
            'person_frames': 0,
            'animal_frames': 0,
            'object_frames': 0,
            'processing_time': 0,
            'oom_dropped_frames': 0,   # frames dropped due to CUDA OOM
        }
    
    def create_output_structure(self, video_name: str) -> Path:
        """Create output directory structure for a video"""
        mode_suffix = "ensemble" if self.is_ensemble else "yolo"
        turbo_suffix = "_turbo" if self.use_turbo else ""
        aspect_ratio = self.cropper.target_format.replace(':', 'x')
        
        source = os.path.normcase(str(Path(self.current_video or video_name).resolve()))
        source_id = hashlib.sha256(source.encode('utf-8')).hexdigest()[:12]
        name = f"{video_name}_{source_id}_{aspect_ratio}_{mode_suffix}{turbo_suffix}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # Unified processing does not implement resume. Every run gets its own
        # namespace rather than overwriting existing images/captions.
        run = 1
        while True:
            base_path = Path(self.output_dir) / (name if run == 1 else f'{name}_run_{run}')
            try:
                base_path.mkdir()
                break
            except FileExistsError:
                run += 1
        
        self.person_dir = base_path / 'persons'
        self.animal_dir = base_path / 'animals'
        self.object_dir = base_path / 'objects'

        self.person_dir.mkdir(exist_ok=True)
        self.animal_dir.mkdir(exist_ok=True)
        self.object_dir.mkdir(exist_ok=True)

        # V3.x NSFW — create sfw/nsfw/(uncertain) subdirs only inside persons/
        if self.nsfw_detector is not None:
            (self.person_dir / 'sfw').mkdir(exist_ok=True)
            (self.person_dir / 'nsfw').mkdir(exist_ok=True)
            if self.nsfw_uncertain_folder:
                (self.person_dir / 'uncertain').mkdir(exist_ok=True)
            print("🔞 NSFW detection enabled — routing persons to sfw/nsfw subdirs")

        self.current_output_dir = base_path
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
                          stop_callback: Optional[Callable] = None,
                          skip_event: Optional[threading.Event] = None,
                          pause_event: Optional[threading.Event] = None,
                          start_skip_seconds: float = 0.0,
                          end_skip_seconds: float = 0.0,
                          frame_saved_callback: Optional[Callable] = None) -> Dict:
        """
        Process all videos in the list

        Args:
            frame_interval: Process every Nth frame
            skip_text: Skip frames with subtitles
            use_quick_text_check: Use fast text detection
            progress_callback: Callback for progress updates
            stop_callback: Callback to check if processing should stop
            skip_event: Optional threading.Event. When set, the *current*
                        video is abandoned and processing continues with
                        the next one. The event is cleared automatically
                        before each video starts, so it behaves as a
                        one-shot "skip current" signal.
            pause_event: Optional threading.Event. Semantics are inverted
                        for convenience: set() means "running", clear()
                        means "paused". The inner loops block on
                        pause_event.wait() at the top of every iteration,
                        so pausing is instantaneous *between* frames.
                        If None, processing never pauses.
            start_skip_seconds: Skip the first N seconds of every video.
                        Applied uniformly to the whole batch — handy for
                        dropping intros from a folder of episodes without
                        clicking through each one.
            end_skip_seconds: Skip the last N seconds of every video.
                        Same batch-wide semantics as ``start_skip_seconds``.
            frame_saved_callback: Optional zero-arg + path callable invoked
                        right after a frame is written to disk. Used by
                        the UI's live preview thumbnail grid.

        Returns:
            Overall statistics for all videos
        """
        total_start = time.time()
        self.overall_stats = {'total_videos': len(self.video_paths), 'processed_videos': 0,
                              'total_frames_saved': 0, 'total_frames_processed': 0,
                              'videos_stats': []}

        # Expose the frame-saved callback to the frame-saving helpers.
        # Stashing it on the instance keeps the signatures of the
        # deeply-nested save paths unchanged.
        self._frame_saved_callback = frame_saved_callback

        for idx, video_path in enumerate(self.video_paths, 1):
            print(f"\n{'='*60}")
            print(f"Processing video {idx}/{len(self.video_paths)}")
            print(f"{'='*60}")

            # Check stop signal
            if stop_callback and stop_callback():
                print("\n⏹️  Batch processing stopped by user")
                break

            # Clear any pre-existing skip flag so each video starts fresh.
            # Build a read-only callable for the inner loops — keeps the
            # signature consistent with ``stop_callback``.
            if skip_event is not None:
                skip_event.clear()
                skip_callback: Optional[Callable[[], bool]] = skip_event.is_set
            else:
                skip_callback = None

            # Process single video
            self.current_output_dir = None
            video_stats = self.process_single_video(
                video_path,
                frame_interval,
                skip_text,
                use_quick_text_check,
                progress_callback,
                stop_callback,
                skip_callback,
                pause_event,
                start_skip_seconds,
                end_skip_seconds,
            )

            # Log whether this video was skipped mid-flight so the UI can
            # reflect it in the per-video summary.
            if skip_event is not None and skip_event.is_set():
                video_stats['skipped_by_user'] = True
                print(f"\n⏭️  Video {idx}/{len(self.video_paths)} skipped by user "
                      f"— moving to next")

            # Update overall stats
            self.overall_stats['processed_videos'] += 1
            self.overall_stats['total_frames_saved'] += video_stats.get('saved_frames', 0)
            self.overall_stats['total_frames_processed'] = (
                self.overall_stats.get('total_frames_processed', 0)
                + video_stats.get('processed_frames', 0)
            )
            output_dir = getattr(self, 'current_output_dir', None)
            self.overall_stats['videos_stats'].append({
                'video_name': Path(video_path).name,
                'output_dir': str(output_dir) if output_dir else None,
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
                            stop_callback: Optional[Callable] = None,
                            skip_callback: Optional[Callable] = None,
                            pause_event: Optional[threading.Event] = None,
                            start_skip_seconds: float = 0.0,
                            end_skip_seconds: float = 0.0) -> Dict:
        """Process a single video

        Args:
            skip_callback: Optional zero-arg callable. When it returns True
                           the inner loop exits early and the current video
                           is abandoned. The outer batch loop is responsible
                           for clearing / resetting the underlying flag so
                           the next video starts fresh.
            pause_event: See ``process_all_videos`` — set() means running,
                           clear() means paused. The inner loop blocks on
                           ``wait()`` at the top of every iteration.
            start_skip_seconds / end_skip_seconds: Batch-wide trim applied
                           to this video as frame-count windows, computed
                           from ``self.fps`` and ``self.total_frames``.
        """
        if isinstance(frame_interval, bool) or not isinstance(frame_interval, int) or frame_interval < 1:
            raise ValueError('frame_interval must be a positive integer.')
        if not self.open_video(video_path):
            if self.cap:
                self.cap.release()
            return self._create_empty_stats()

        # Create output structure for this video
        video_name = Path(video_path).stem
        self.create_output_structure(video_name)

        # Reset stats for this video
        self.stats = self._create_empty_stats()
        self.start_time = time.time()
        self._stage_seconds = {}
        self._video_reader = None

        # Reset duplicate detection history for each video
        if self.quality_analyzer and hasattr(self.quality_analyzer, 'clear_history'):
            self.quality_analyzer.clear_history()

        # Resolve the trim window [start_frame, end_frame) in source frames.
        # Clamp to [0, total_frames]. If the window is empty (start >= end)
        # skip the whole video with a clear log line.
        start_frame = int(max(0.0, start_skip_seconds) * self.fps)
        end_frame = self.total_frames - int(max(0.0, end_skip_seconds) * self.fps)
        if end_frame <= start_frame:
            logger.warning(
                "Trim window [%.1fs .. -%.1fs] is empty for %s (fps=%.1f, "
                "total=%d). Skipping video entirely.",
                start_skip_seconds, end_skip_seconds, video_path,
                self.fps, self.total_frames,
            )
            self.cap.release()
            return self.stats
        if start_frame > 0 or end_frame < self.total_frames:
            print(f"   ✂  Trim: frames [{start_frame} .. {end_frame}) "
                  f"({start_skip_seconds:.1f}s head, {end_skip_seconds:.1f}s tail)")

        self._pending_crops = []
        self._pending_crop_bytes = 0
        self._defer_nsfw_saves = (
            self.nsfw_detector is not None
            and callable(getattr(self.nsfw_detector, 'classify_batch', None))
        )
        try:
            if self.use_turbo:
                self._process_video_turbo(
                    frame_interval, skip_text, use_quick_text_check,
                    progress_callback, stop_callback, skip_callback,
                    pause_event, start_frame, end_frame,
                )
            else:
                self._process_video_standard(
                    frame_interval, skip_text, use_quick_text_check,
                    progress_callback, stop_callback, skip_callback,
                    pause_event, start_frame, end_frame,
                )
        finally:
            try:
                # Drain already accepted crops even on stop/skip. At most one
                # bounded batch remains; never write a previous video's crops
                # into the next video's directory.
                self._flush_pending_crops()
            finally:
                self._defer_nsfw_saves = False
                self._pending_crops = []
                self._pending_crop_bytes = 0
                if self.cap:
                    self.cap.release()
            elapsed = time.time() - self.start_time
            self.stats['processing_time'] = elapsed
            stages = dict(self._stage_seconds)
            if self._video_reader is not None:
                self.stats['video_read'] = self._video_reader.snapshot()
                stages['video_read'] = self._video_reader.seconds
            stages['sfw_nsfw'] = self.stats['nsfw_classification_time']
            self.stats['stage_seconds'] = stages
            measured = sum(stages.values())
            self.stats['other_and_wait_seconds'] = max(0.0, elapsed - measured)
            self._log('PERF | ' + ' | '.join(f'{name}={seconds:.3f}s'
                      for name, seconds in sorted(stages.items(), key=lambda item: -item[1]))
                      + f' | total={elapsed:.3f}s')
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
                               stop_callback: Optional[Callable],
                               skip_callback: Optional[Callable] = None,
                               pause_event: Optional[threading.Event] = None,
                               start_frame: int = 0,
                               end_frame: Optional[int] = None):
        """Standard video processing (frame by frame)"""
        frame_count = 0
        self._video_reader = reader = SelectiveVideoReader(self.cap)

        while True:
            # Pause check: if the event is cleared, block here until
            # resume (or stop). wait() returns True immediately when the
            # event is already set — i.e. not paused — so the hot path
            # is effectively free.
            if pause_event is not None and not pause_event.is_set():
                self._flush_pending_crops()
                print("\n⏸  Paused — waiting for resume...")
                while not pause_event.wait(0.1):
                    if ((stop_callback and stop_callback())
                            or (skip_callback and skip_callback())):
                        return
                print("▶  Resumed")

            if stop_callback and stop_callback():
                break
            if skip_callback and skip_callback():
                # Caller asked to abandon the current video — the outer
                # batch loop will continue with the next one.
                print("\n⏭️  Skipping current video by user request")
                break

            # Keep the same one-based sample numbers and half-open trim.
            # Sequential grabs retain codec dependencies; do not random-seek.
            if end_frame is not None and frame_count >= end_frame:
                break
            selected = (frame_count >= start_frame
                        and (frame_count + 1) % frame_interval == 0)
            ret, frame = reader.read(selected)
            if not ret:
                break
            frame_count += 1
            if not selected:
                if frame_count % 64 == 0:
                    progress = frame_count / self.total_frames * 100 if self.total_frames else 0
                    self._emit_progress(progress_callback, progress)
                continue

            self.stats['processed_frames'] += 1

            # Progress callback — time-throttled to avoid UI freeze
            progress = (frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
            self._emit_progress(progress_callback, progress)

            # Process frame
            self._process_single_frame(frame, frame_count, skip_text, use_quick_text)

    def _process_video_turbo(self,
                            frame_interval: int,
                            skip_text: bool,
                            use_quick_text: bool,
                            progress_callback: Optional[Callable],
                            stop_callback: Optional[Callable],
                            skip_callback: Optional[Callable] = None,
                            pause_event: Optional[threading.Event] = None,
                            start_frame: int = 0,
                            end_frame: Optional[int] = None):
        """Turbo video processing (batch frames)"""
        frame_count = 0
        frame_batch = []
        frame_numbers = []
        self._video_reader = reader = SelectiveVideoReader(self.cap)

        while True:
            if pause_event is not None and not pause_event.is_set():
                # Flush whatever is already buffered before parking —
                # otherwise a long pause would hold onto VRAM/RAM for
                # no reason.
                if frame_batch:
                    self._safe_process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                    frame_batch = []
                    frame_numbers = []
                self._flush_pending_crops()
                print("\n⏸  Paused — waiting for resume...")
                while not pause_event.wait(0.1):
                    if ((stop_callback and stop_callback())
                            or (skip_callback and skip_callback())):
                        return
                print("▶  Resumed")

            if stop_callback and stop_callback():
                # User stopped — discard pending batch so we exit fast
                break
            if skip_callback and skip_callback():
                # Flush buffered frames for this video before moving on
                if frame_batch:
                    self._safe_process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                print("\n⏭️  Skipping current video by user request")
                break

            if end_frame is not None and frame_count >= end_frame:
                if frame_batch:
                    self._safe_process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                break
            selected = (frame_count >= start_frame
                        and (frame_count + 1) % frame_interval == 0)
            ret, frame = reader.read(selected)
            if not ret:
                if frame_batch:
                    self._safe_process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                break
            frame_count += 1
            if not selected:
                if frame_count % 64 == 0:
                    progress = frame_count / self.total_frames * 100 if self.total_frames else 0
                    self._emit_progress(progress_callback, progress)
                continue

            frame_batch.append(frame)
            frame_numbers.append(frame_count)

            if len(frame_batch) >= self.batch_size:
                self._safe_process_batch(frame_batch, frame_numbers, skip_text, use_quick_text)
                frame_batch = []
                frame_numbers = []

            if frame_count % max(frame_interval, 1) == 0:
                progress = (frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
                self._emit_progress(progress_callback, progress)

    # ─── VRAM-safe batch wrapper ──────────────────────────────────────────
    def _safe_process_batch(self,
                           frames: List[np.ndarray],
                           frame_numbers: List[int],
                           skip_text: bool,
                           use_quick_text: bool,
                           depth: int = 0) -> None:
        """
        Wrap ``_process_batch`` with CUDA OOM recovery. If the batch
        explodes we:

          1. Drain the CUDA caching allocator via ``empty_cache()``.
          2. Halve the instance-wide ``batch_size`` (down to
             ``self._min_batch_size``) so subsequent batches are smaller.
          3. Split the current batch in half and retry each half
             recursively. This way a single over-sized batch degrades
             gracefully instead of aborting the whole run.
          4. If we're already at size 1 and still blowing up the only
             sane thing is to drop the frame and log it — re-raising
             would kill the whole video.
        """
        if not frames:
            return
        saved_stats = dict(self.stats)
        quality_state = {}
        if self.quality_analyzer is not None:
            for attr in ('frame_hashes', 'frame_histograms', 'stats'):
                if hasattr(self.quality_analyzer, attr):
                    quality_state[attr] = copy.copy(getattr(self.quality_analyzer, attr))
        try:
            self._process_batch(frames, frame_numbers, skip_text, use_quick_text)
            return
        except RuntimeError as e:
            # Only intercept OOMs — everything else is a real bug.
            if 'out of memory' not in str(e).lower():
                raise
            # Detection/preprocessing failed before persistence. Restore both
            # duplicate history and counters before retrying these same frames.
            self.stats.clear()
            self.stats.update(saved_stats)
            for attr, value in quality_state.items():
                setattr(self.quality_analyzer, attr, value)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(
                "CUDA OOM on batch of %d frames (depth=%d). Shrinking "
                "batch_size %d → %d and retrying.",
                len(frames), depth, self.batch_size,
                max(self._min_batch_size, self.batch_size // 2),
            )
            # Permanently shrink the default batch size for subsequent calls.
            self.batch_size = max(self._min_batch_size, self.batch_size // 2)

            if len(frames) == 1:
                # Can't split further — drop with a warning. Counting
                # this lets us surface it in the final summary.
                self.stats['oom_dropped_frames'] = (
                    self.stats.get('oom_dropped_frames', 0) + 1
                )
                logger.error(
                    "Dropping frame %s: still OOM at batch_size=1",
                    frame_numbers[0],
                )
                return
            if depth > 6:
                # Pathological case — bail out rather than recursing forever.
                logger.error(
                    "OOM retry depth exceeded — dropping %d frames",
                    len(frames),
                )
                self.stats['oom_dropped_frames'] = (
                    self.stats.get('oom_dropped_frames', 0) + len(frames)
                )
                return
            mid = len(frames) // 2
            self._safe_process_batch(
                frames[:mid], frame_numbers[:mid], skip_text, use_quick_text, depth + 1,
            )
            self._safe_process_batch(
                frames[mid:], frame_numbers[mid:], skip_text, use_quick_text, depth + 1,
            )

    def _process_batch(self, frames: List[np.ndarray], frame_numbers: List[int],
                      skip_text: bool, use_quick_text: bool):
        """Process a batch of frames with GPU batch detection"""
        batch_size = len(frames)

        # Quick text check - filter first (with overlay-aware logic)
        valid_frames = []
        valid_frame_nums = []

        if skip_text and self.text_detector:
            for i, (frame, frame_num) in enumerate(zip(frames, frame_numbers)):
                if use_quick_text:
                    has_text = self._timed('text', self.text_detector.quick_text_check, frame)
                else:
                    has_text, _ = self._timed('text', self.text_detector.has_text, frame)

                if has_text:
                    if self.subtitle_removal:
                        # V3.x: remove subtitle via inpaint, keep the frame
                        regions = []
                        if hasattr(self.text_detector, 'detect_overlay_regions'):
                            regions = self._timed('text', self.text_detector.detect_overlay_regions, frame)
                        if regions:
                            frame = self._timed('text', self.text_detector.remove_subtitle_regions, frame, regions)
                            self.stats['subtitle_removed'] += 1
                        valid_frames.append(frame)
                        valid_frame_nums.append(frame_num)
                    else:
                        # Check if we can crop around overlays instead of skipping
                        if hasattr(self.text_detector, 'detect_overlay_regions'):
                            overlay_regions = self._timed('text', self.text_detector.detect_overlay_regions, frame)
                            if overlay_regions:
                                # Has overlay regions → keep frame, crop around them
                                valid_frames.append(frame)
                                valid_frame_nums.append(frame_num)
                                continue
                        # No overlay regions → skip
                        self.stats['skipped_text'] += 1
                else:
                    valid_frames.append(frame)
                    valid_frame_nums.append(frame_num)
        else:
            valid_frames = frames
            valid_frame_nums = frame_numbers
        
        if not valid_frames:
            return
        
        # V3.x: Upscale full frames before quality check if target="frame"
        if self.upscaler and self.upscale_target == "frame":
            valid_frames = [self._upscale_image(f) for f in valid_frames]

        # V2.0: Batch quality check if enabled
        # V3.x: Low-res frames are rescued by upscaler before quality check
        quality_mask = [True] * len(valid_frames)
        if self.quality_analyzer:
            for i, frame in enumerate(valid_frames):
                # V3.x: rescue low-res frames by upscaling before blur check
                if (self.upscaler and self.upscale_target == "crop"
                        and min(frame.shape[:2]) < self.upscale_min_resolution):
                    frame = self._upscale_image(frame)
                    valid_frames[i] = frame
                is_quality_ok, _ = self._timed('quality', self.quality_analyzer.check_frame_quality, frame)
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
            all_detections = self._timed('detection', self.detector.detect_batch, final_frames)
        else:
            all_detections = [self._timed('detection', self.detector.detect, f) for f in final_frames]
        
        # Process results
        for frame, frame_num, detections in zip(final_frames, final_frame_nums, all_detections):
            self.stats['processed_frames'] += 1
            try:
                self._process_frame_with_detection(frame, frame_num, detections)
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                # Do not replay already saved siblings on a post-detection OOM.
                self.stats['oom_dropped_frames'] += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning('Skipping frame %s after post-detection OOM: %s', frame_num, exc)
        self._flush_pending_crops()

    def _process_frame_with_detection(self, frame: np.ndarray, frame_number: int,
                                       detections: Dict):
        """Process frame with pre-computed detections"""
        # Frame may have been mutated (subtitle crop / upscale) — use its
        # CURRENT dimensions for all geometry, not the original video dims.
        fh, fw = frame.shape[:2]

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
                fh
            )

        # Detect overlay regions for batch frame (for exclusion-aware crop)
        excluded_zones = None
        if self.text_detector and hasattr(self.text_detector, 'detect_overlay_regions'):
            excluded_zones = self._timed('text', self.text_detector.detect_overlay_regions, frame) or None

        # Calculate crop
        crop_box = self.cropper.calculate_crop_box(
            (fh, fw),
            subject['bbox'],
            category,
            head_space,
            excluded_zones=excluded_zones,
        )

        if crop_box is None:
            return

        # Apply crop
        cropped = self.cropper.apply_crop(frame, crop_box)

        # V3.x: Upscale crop if enabled (target="crop")
        if self.upscaler and self.upscale_target == "crop":
            cropped = self._upscale_image(cropped)

        # Quality score — use current frame dims (may differ from original)
        quality = self.cropper.calculate_quality_score(
            (fh, fw),
            crop_box,
            subject['bbox']
        )

        if quality > 0.3:
            self._submit_crop(cropped, category, frame_number, quality, bool(excluded_zones))

    def _process_single_frame(self, frame: np.ndarray, frame_number: int,
                             skip_text: bool, use_quick_text: bool):
        """Process a single frame with V2.0 quality and captioning support"""
        # Text / overlay detection
        excluded_zones = None
        if skip_text and self.text_detector:
            # Step 1: Check for subtitle text
            if use_quick_text:
                has_subtitle = self._timed('text', self.text_detector.quick_text_check, frame)
            else:
                has_subtitle, _ = self._timed('text', self.text_detector.has_text, frame)

            if has_subtitle:
                if self.subtitle_removal:
                    # V3.x: remove subtitle regions via inpaint, keep the frame
                    regions = []
                    if hasattr(self.text_detector, 'detect_overlay_regions'):
                        regions = self._timed('text', self.text_detector.detect_overlay_regions, frame)
                    if regions:
                        frame = self._timed('text', self.text_detector.remove_subtitle_regions, frame, regions)
                        self.stats['subtitle_removed'] += 1
                    excluded_zones = None  # cleaned — no exclusion needed
                else:
                    # Step 2: Subtitle detected — check if we can crop around overlays
                    if hasattr(self.text_detector, 'detect_overlay_regions'):
                        excluded_zones = self._timed('text', self.text_detector.detect_overlay_regions, frame)
                        if excluded_zones:
                            # Overlay regions found → crop around them instead of skipping
                            pass
                        else:
                            # No specific overlay regions to crop around → skip frame
                            self.stats['skipped_text'] += 1
                            return
                    else:
                        # No overlay detection available → skip frame
                        self.stats['skipped_text'] += 1
                        return
            else:
                # No subtitle — still collect overlay regions for exclusion-aware cropping
                if hasattr(self.text_detector, 'detect_overlay_regions'):
                    excluded_zones = self._timed('text', self.text_detector.detect_overlay_regions, frame) or None
        
        # V3.x: Upscale full frame if target="frame"
        if self.upscaler and self.upscale_target == "frame":
            frame = self._upscale_image(frame)

        # V2.0: Quality check before processing
        # V3.x: Low-res frames rescued by upscaling before blur check
        if self.quality_analyzer:
            if (self.upscaler and self.upscale_target == "crop"
                    and min(frame.shape[:2]) < self.upscale_min_resolution):
                frame = self._upscale_image(frame)
            is_quality_ok, quality_info = self._timed('quality', self.quality_analyzer.check_frame_quality, frame)
            if not is_quality_ok:
                self.stats['skipped_quality'] += 1
                return

        # Frame may have been mutated (subtitle crop / upscale) — use its
        # CURRENT dimensions for all geometry, not the original video dims.
        fh, fw = frame.shape[:2]

        # Detect objects
        detections = self._timed('detection', self.detector.detect, frame)

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
                fh
            )

        # Calculate crop box — pass overlay regions so they're avoided
        crop_box = self.cropper.calculate_crop_box(
            (fh, fw),
            subject['bbox'],
            category,
            head_space,
            excluded_zones=excluded_zones,
        )

        if crop_box is None:
            return

        # Apply crop
        cropped = self.cropper.apply_crop(frame, crop_box)

        # V3.x: Upscale crop if enabled (target="crop")
        if self.upscaler and self.upscale_target == "crop":
            cropped = self._upscale_image(cropped)

        # Quality check — use current frame dims (may differ from original)
        quality = self.cropper.calculate_quality_score(
            (fh, fw),
            crop_box,
            subject['bbox']
        )

        if quality > 0.3:
            self._submit_crop(cropped, category, frame_number, quality, bool(excluded_zones))

    def _submit_crop(self, frame: np.ndarray, category: str, frame_number: int,
                     quality: float, overlay: bool) -> None:
        """Preserve accepted-crop order while amortizing GPU classifier calls."""
        record = (frame, category, frame_number, quality, overlay)
        if not self._defer_nsfw_saves or (category != 'person' and not self._pending_crops):
            self._persist_crop(record)
            return
        limit = min(32, max(1, int(getattr(self.nsfw_detector, 'effective_batch_size', 1))))
        if limit == 1:
            self._flush_pending_crops()
            self._persist_crop(record)
            return
        if self._pending_crops and (
                len(self._pending_crops) >= limit
                or self._pending_crop_bytes + frame.nbytes > self._pending_crop_max_bytes):
            self._flush_pending_crops()
        if frame.nbytes > self._pending_crop_max_bytes:
            self._persist_crop(record)
            return
        # A crop can be a view into a much larger source frame. A compact copy
        # makes the byte ceiling real and prevents later source reuse/mutation.
        owned = frame.copy()
        self._pending_crops.append((owned, category, frame_number, quality, overlay))
        self._pending_crop_bytes += owned.nbytes
        if len(self._pending_crops) >= limit:
            self._flush_pending_crops()

    @staticmethod
    def _valid_nsfw_result(result) -> tuple:
        try:
            label, confidence = result
            confidence = float(confidence)
            if label not in ('sfw', 'nsfw', 'uncertain'):
                raise ValueError('Unknown classifier label.')
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid classifier confidence.')
            return label, confidence
        except (TypeError, ValueError):
            return 'uncertain', 0.5

    def _classify_crops(self, frames: List[np.ndarray]) -> list:
        """Run only on final person crops; no full-frame rating propagation."""
        if not frames:
            return []
        start = time.perf_counter()
        try:
            batch = getattr(self.nsfw_detector, 'classify_batch', None)
            if callable(batch):
                results = batch(frames)
            else:
                results = [self.nsfw_detector.classify(frame) for frame in frames]
            if not isinstance(results, (list, tuple)) or len(results) != len(frames):
                raise ValueError('SFW/NSFW result count mismatch.')
            return [self._valid_nsfw_result(result) for result in results]
        except Exception as exc:
            # Never let a classifier OOM reach detection's replay wrapper:
            # sibling crops may have already been persisted.
            self._log(f'SFW/NSFW error; {len(frames)} crops marked uncertain: {exc}')
            return [('uncertain', 0.5) for _ in frames]
        finally:
            self.stats['nsfw_classification_time'] += time.perf_counter() - start
            self.stats['nsfw_crops_analyzed'] += len(frames)
            self.stats['nsfw_batches'] += 1
            self.stats['nsfw_max_batch'] = max(self.stats['nsfw_max_batch'], len(frames))

    def _flush_pending_crops(self) -> None:
        if not self._pending_crops:
            return
        pending = self._pending_crops
        # Detach before inference/persistence: a later cleanup must never
        # replay a partially saved batch, even after an exception.
        self._pending_crops = []
        self._pending_crop_bytes = 0
        person_frames = [record[0] for record in pending if record[1] == 'person']
        ratings = iter(self._classify_crops(person_frames))
        for record in pending:
            rating = next(ratings) if record[1] == 'person' else None
            self._persist_crop(record, rating)

    def _persist_crop(self, record: tuple, nsfw_result=None) -> None:
        frame, category, frame_number, quality, overlay = record
        saved_path = self.save_cropped_frame(
            frame, category, frame_number, quality, nsfw_result=nsfw_result)
        if saved_path is None:
            return
        self.stats['saved_frames'] += 1
        self.stats[f'{category}_frames'] += 1
        if overlay:
            self.stats['overlay_crops'] += 1
        self._timed('caption', self._caption_frame, frame, saved_path, frame_number)

    def _caption_frame(self, cropped: np.ndarray, saved_path: Path, frame_number: int):
        """
        Run auto-captioning on a saved frame and write the .txt file.

        Args:
            cropped:      The cropped frame image (BGR ndarray).
            saved_path:   Path where the frame JPEG was saved.
            frame_number: Original frame index (for logging).
        """
        if not saved_path:
            return
        if not self.captioner and not self.florence2:
            return
        try:
            mode = self.caption_mode
            use_wd14 = mode in ('tags_only', 'combined') and self.captioner is not None
            use_f2 = mode in ('florence2', 'combined') and self.florence2 is not None

            sep = ', '
            parts: list = []

            # Florence-2 NLP caption first (so tags come after in combined mode)
            if use_f2:
                try:
                    nlp = self.florence2.generate(cropped, task=self.florence2_task)
                    if nlp:
                        parts.append(nlp)
                except Exception as e:
                    self._log(f"⚠️ Florence-2 error frame {frame_number}: {e}")

            tag_count = 0
            if use_wd14:
                wd14_result = self.captioner.caption_image(cropped, mode='tags_only')
                tag_count = getattr(wd14_result, 'tag_count', 0)
                if mode == 'tags_only':
                    # Use the fully-formatted WD14 caption (includes trigger/suffix)
                    caption = wd14_result.final_caption
                else:
                    # combined: only take the raw tag body (no trigger/suffix yet)
                    tag_body = sep.join(getattr(wd14_result, 'filtered_tags', []) or [])
                    if tag_body:
                        parts.append(tag_body)
                    caption = None  # assembled below
            else:
                caption = None

            if caption is None:
                # florence2 or combined — assemble trigger + parts + suffix
                body = sep.join(parts) if parts else ''
                ts = getattr(self.captioner, 'tag_settings', None) if self.captioner else None
                trigger = (ts.trigger_word.strip() if ts and ts.trigger_word else '')
                suffix = (ts.caption_suffix.strip() if ts and ts.caption_suffix else '')
                caption = body
                if trigger:
                    caption = f"{trigger}{sep}{caption}" if caption else trigger
                if suffix:
                    caption = f"{caption}{sep}{suffix}" if caption else suffix
                caption = caption.strip()

            caption_path = saved_path.with_suffix('.txt')
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            self.stats['captioned_frames'] += 1
            if self.stats['captioned_frames'] == 1:
                preview = (caption[:60] + "...") if len(caption) > 60 else caption
                self._log(f"📝 First caption: {preview}")
            # Warn if WD14 produced 0 tags and we relied on it
            if use_wd14 and tag_count == 0 and not use_f2:
                if self.stats.get('_zero_tag_warned', 0) == 0:
                    self._log("⚠️ WD14 produced 0 tags — captions will only contain trigger word")
                    self.stats['_zero_tag_warned'] = 1
        except Exception as e:
            self._log(f"⚠️ Caption error frame {frame_number}: {e}")

    def _cap_resolution(self, frame: np.ndarray) -> np.ndarray:
        """Downscale *frame* so its longest side <= upscale_max_resolution.

        Aspect ratio preserved, alpha-safe (works on 3ch and 4ch). No-op when
        the cap is 0/off or the frame already fits.
        """
        cap = getattr(self, 'upscale_max_resolution', 0)
        if cap <= 0 or frame is None:
            return frame
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= cap:
            return frame
        scale = cap / float(longest)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _upscale_image(self, frame: np.ndarray) -> np.ndarray:
        """Upscale *frame* via the registered upscaler; returns original on failure."""
        if not self.upscaler or not self.upscaler.is_available():
            return self._cap_resolution(frame)
        result = self._timed('upscale', self.upscaler.upscale, frame)
        if result is not frame:
            self.stats['upscaled_frames'] += 1
        return self._cap_resolution(result)

    def save_cropped_frame(self, frame: np.ndarray, category: str,
                          frame_number: int, quality: float, nsfw_result=None) -> Optional[Path]:
        """Save cropped frame to appropriate directory and return path.

        When nsfw_detector is set, classifies the frame first and routes
        to category/sfw/, category/nsfw/, or category/uncertain/.
        """
        if category == 'person':
            base_dir = self.person_dir
        elif category == 'animal':
            base_dir = self.animal_dir
        else:
            base_dir = self.object_dir

        # Precomputed batch predictions never trigger a second inference.
        rating_label = None
        if self.nsfw_detector is not None and category == 'person':
            rating = (self._classify_crops([frame])[0] if nsfw_result is None
                      else self._valid_nsfw_result(nsfw_result))
            rating_label, _ = rating
            folder_label = rating_label
            if rating_label == 'uncertain' and not self.nsfw_uncertain_folder:
                # Retain the existing explicit legacy folder preference. The
                # counter remains uncertain, never counted as confirmed SFW.
                folder_label = 'sfw'
            output_dir = base_dir / folder_label
        else:
            output_dir = base_dir

        filename = f"frame_{frame_number:06d}_q{int(quality*100)}.png"
        from src.core.dataset_files import unique_image_path
        output_path = self._timed('output_naming', unique_image_path, output_dir / filename)

        # PNG: lossless, unicode-safe. compression 3 = good size/speed balance.
        if not self._timed('image_write', _imwrite_unicode, str(output_path), frame, ext=".png"):
            self.stats['save_errors'] = self.stats.get('save_errors', 0) + 1
            self._log(f'Image write failed: {output_path}')
            return None

        # Count folder classifications only after a successful durable write.
        if rating_label is not None:
            key = {'sfw': 'sfw_frames', 'nsfw': 'nsfw_frames', 'uncertain': 'nsfw_uncertain'}[rating_label]
            self.stats[key] += 1

        # Notify the UI's live preview grid (throttled / no-op if None).
        cb = self._frame_saved_callback
        if cb is not None:
            try:
                cb(str(output_path))
            except Exception as e:
                logger.debug("frame_saved_callback raised: %s", e)

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
        print(f"Skipped (quality):   {self.stats['skipped_quality']}")   # V2.0
        print(f"Overlay crops:       {self.stats['overlay_crops']}")     # logo/watermark aware
        print(f"Captioned frames:    {self.stats['captioned_frames']}")  # V2.0
        if self.nsfw_detector is not None:
            print(f"  |- SFW:           {self.stats['sfw_frames']}")
            print(f"  |- NSFW:          {self.stats['nsfw_frames']}")
            print(f"  |- Uncertain:     {self.stats['nsfw_uncertain']}")
            count = self.stats['nsfw_crops_analyzed']
            elapsed = self.stats['nsfw_classification_time']
            milliseconds = elapsed * 1000 / count if count else 0.0
            device = getattr(self.nsfw_detector, 'actual_device', 'unknown')
            self._log(
                f"SFW/NSFW: {device}, {count} crops, {elapsed:.2f}s "
                f"({milliseconds:.1f} ms/crop), max batch={self.stats['nsfw_max_batch']}. "
                "Classification time excludes video decode, captioning and saving.")


    def print_overall_summary(self):
        """Summarize completion without relying on a missing legacy method."""
        stats = self.overall_stats
        self._log('Completed: {videos} videos, {frames} saved frames, {seconds:.1f}s'.format(
            videos=stats.get('processed_videos', 0),
            frames=stats.get('total_frames_saved', 0),
            seconds=stats.get('total_time', 0)))
