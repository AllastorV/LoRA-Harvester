"""
Command Line Interface for LoRA-Harvester v3.0
AI-Powered Dataset Collection Tool for LoRA Training
Supports: Batch processing, Quality analysis, Auto-captioning, Resume
"""

import argparse
import sys
import os
import glob
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# ── Early lightweight commands (no cv2/torch needed) ────────────────────────
if '--list-upscale-models' in sys.argv:
    try:
        from src.core.upscale_models import list_models
        models = list_models()
        print(f"\nAvailable upscale models ({len(models)} total):")
        print(f"  {'Name':<35} {'Scale':>5}  {'Arch':<20}  Status")
        print(f"  {'-'*35} {'-----':>5}  {'-'*20}  ------")
        for name, cfg in models.items():
            status = "ready" if cfg.get('available') else "not downloaded"
            print(f"  {name:<35} {cfg.get('scale','?'):>5}x  "
                  f"{cfg.get('arch','?'):<20}  {status}")
        print(f"\nDownload: python scripts/download_models.py --upscale-models <name>")
    except Exception as e:
        print(f"Error loading model registry: {e}")
    sys.exit(0)
# ─────────────────────────────────────────────────────────────────────────────

from src.core.detector import ObjectDetector
from src.core.text_detector import SubtitleDetector
from src.core.cropper import SmartCropper
from src.core.enhanced_processor import EnhancedVideoProcessor
from src.core.unified_processor import UnifiedVideoProcessor
from src.core.advanced_captioner import (
    AdvancedCaptioner, TagSettings, CAPTIONER_PRESETS
)
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def parse_negative_tags(tags_str: str) -> list:
    """Parse comma-separated negative tags"""
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split(',') if t.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="🌾 LoRA-Harvester v3.0 - AI Powered Dataset Collection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py input.mp4 -o output_folder
  
  # Batch processing
  python cli.py *.mp4 -o output --turbo
  
  # With quality filtering
  python cli.py input.mp4 -o output --quality --no-blur --no-duplicates
  
  # With auto-captioning (WD14 tags)
  python cli.py input.mp4 -o output --caption --caption-mode tags_only
  
  # With trigger word and custom settings
  python cli.py input.mp4 -o output --caption --trigger "my_character" --max-tags 25
  
  # Using preset
  python cli.py input.mp4 -o output --caption --preset anime_character
  
  # Full example with all features
  python cli.py videos/*.mp4 -o dataset -f 9:16 -i 15 \\
      --quality --turbo --ensemble \\
      --caption --trigger "sks person" --max-tags 30 \\
      --negative-tags "watermark,signature,text"
        """
    )
    
    # ==================== VIDEO INPUT ====================
    parser.add_argument('videos', nargs='+', 
                       help='Input video file path(s) - supports wildcards')
    
    # ==================== OUTPUT ====================
    parser.add_argument('-o', '--output', default='output',
                       help='Output directory (default: output)')
    
    # ==================== FORMAT & CROPPING ====================
    parser.add_argument('-f', '--format', default='9:16',
                       choices=['9:16', '3:4', '1:1', '4:5', '16:9', '4:3'],
                       help='Output aspect ratio (default: 9:16)')
    parser.add_argument('-p', '--padding', type=int, default=500,
                       help='Minimum padding around objects in pixels (default: 500)')
    
    # ==================== PROCESSING ====================
    parser.add_argument('-i', '--interval', type=int, default=30,
                       help='Frame interval - process every N frames (default: 30)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Detection confidence threshold 0-1 (default: 0.5)')
    parser.add_argument('--no-skip-text', action='store_true',
                       help='Process frames with text/subtitles')
    
    # ==================== MODEL SELECTION ====================
    parser.add_argument('-m', '--model', default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLO model size (default: yolov8n.pt)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Enable ensemble detection mode with NMS post-processing')
    parser.add_argument('--ensemble-models', nargs='+',
                       choices=['yolo'],
                       default=['yolo'],
                       help='Models to use in ensemble mode')
    parser.add_argument('--voting-threshold', type=int, default=1,
                       help='Minimum model agreements for ensemble (default: 1)')
    
    # ==================== PERFORMANCE ====================
    parser.add_argument('--turbo', action='store_true', default=True,
                       help='Enable turbo mode [DEFAULT]')
    parser.add_argument('--no-turbo', action='store_true',
                       help='Disable turbo mode')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for turbo mode (default: 4)')
    
    # ==================== QUALITY ANALYSIS (NEW) ====================
    quality_group = parser.add_argument_group('Quality Analysis')
    quality_group.add_argument('--quality', action='store_true',
                              help='Enable quality analysis (blur, lighting, duplicates)')
    quality_group.add_argument('--blur-threshold', type=float, default=80.0,
                              help='Minimum sharpness score (default: 80.0)')
    quality_group.add_argument('--brightness-min', type=int, default=35,
                              help='Minimum brightness 0-255 (default: 35)')
    quality_group.add_argument('--brightness-max', type=int, default=225,
                              help='Maximum brightness 0-255 (default: 225)')
    quality_group.add_argument('--no-duplicates', action='store_true',
                              help='Skip duplicate/similar frames')
    quality_group.add_argument('--duplicate-threshold', type=float, default=0.90,
                              help='Similarity threshold for duplicates (default: 0.90)')
    
    # ==================== UPSCALE (NEW v3.x) ====================
    upscale_group = parser.add_argument_group('Upscale (Real-ESRGAN)')
    upscale_group.add_argument('--upscale', action='store_true',
                               help='Enable Real-ESRGAN upscaling (requires: pip install realesrgan basicsr)')
    upscale_group.add_argument('--upscale-model', default='RealESRGAN_x4plus_anime_6B',
                               help='Upscale model name from registry (default: RealESRGAN_x4plus_anime_6B). '
                                    'Use --list-upscale-models to see all.')
    upscale_group.add_argument('--upscale-target', default='crop', choices=['crop', 'frame'],
                               help='What to upscale: crop (after detection, faster) or '
                                    'frame (full frame before detection). Default: crop')
    upscale_group.add_argument('--upscale-tile', type=int, default=0,
                               help='Tile size for low-VRAM GPUs (0=no tiling, default: 0)')
    upscale_group.add_argument('--upscale-min-res', type=int, default=512,
                               help='Rescue frames below this min dimension (px) before quality filter. '
                                    'Default: 512. Set 0 to disable rescue logic.')
    upscale_group.add_argument('--face-enhance', action='store_true',
                               help='GFPGAN face restoration on top of ESRGAN (requires: pip install gfpgan)')
    upscale_group.add_argument('--list-upscale-models', action='store_true',
                               help='List available upscale models and exit')

    # ==================== SCENE DETECTION (NEW) ====================
    scene_group = parser.add_argument_group('Scene Detection')
    scene_group.add_argument('--scene-detection', action='store_true',
                            help='Use scene detection instead of fixed interval')
    scene_group.add_argument('--scene-threshold', type=float, default=25.0,
                            help='Scene change detection threshold (default: 25.0)')
    
    # ==================== CAPTIONING (NEW) ====================
    caption_group = parser.add_argument_group('Auto Captioning (WD14/Danbooru)')
    caption_group.add_argument('--caption', action='store_true',
                              help='Enable auto-captioning')
    caption_group.add_argument('--caption-mode', default='tags_only',
                              choices=['tags_only', 'tag_first', 'florence2', 'combined'],
                              help='Caption mode: tags_only, tag_first, florence2, combined (default: tags_only)')
    caption_group.add_argument('--wd14-model', default='wd-v1-4-vit-tagger-v2',
                              choices=['wd-v1-4-vit-tagger-v2', 'wd-v1-4-convnext-tagger-v2', 'wd-v1-4-swinv2-tagger-v2'],
                              help='WD14 model (default: wd-v1-4-vit-tagger-v2)')
    caption_group.add_argument('--no-wd14', action='store_true',
                              help='Disable WD14 tagging')
    
    # ==================== TAG SETTINGS (NEW) ====================
    tag_group = parser.add_argument_group('Tag Settings')
    tag_group.add_argument('--trigger', '--trigger-word', dest='trigger_word', default='',
                          help='Trigger word added at beginning of every caption')
    tag_group.add_argument('--max-tags', type=int, default=30,
                          help='Maximum number of tags (default: 30)')
    tag_group.add_argument('--min-confidence', type=float, default=0.35,
                          help='Minimum tag confidence 0-1 (default: 0.35)')
    tag_group.add_argument('--negative-tags', type=str, default='',
                          help='Comma-separated tags to exclude (e.g., "watermark,signature,text")')
    tag_group.add_argument('--priority-tags', type=str, default='',
                          help='Comma-separated priority tags (always included)')
    tag_group.add_argument('--preset', choices=['anime_character', 'style_lora', 'realistic_photo', 'concept_art'],
                          help='Use preset tag configuration')
    tag_group.add_argument('--keep-character', action='store_true', default=True,
                          help='Keep character name tags [DEFAULT]')
    tag_group.add_argument('--no-character', action='store_true',
                          help='Remove character name tags')
    tag_group.add_argument('--keep-series', action='store_true',
                          help='Keep series/copyright tags')
    tag_group.add_argument('--include-quality', action='store_true',
                          help='Include quality tags (masterpiece, etc.)')
    tag_group.add_argument('--include-rating', action='store_true',
                          help='Include rating tags')
    tag_group.add_argument('--tag-separator', default=', ',
                          help='Tag separator (default: ", ")')
    tag_group.add_argument('--no-underscores', action='store_true',
                          help='Use spaces instead of underscores in tags')
    tag_group.add_argument('--caption-prefix', default='',
                          help='Prefix added before caption')
    tag_group.add_argument('--caption-suffix', default='',
                          help='Suffix added after caption')
    tag_group.add_argument('--save-json', action='store_true',
                          help='Also save detailed JSON with tags')
    
    # ==================== CHECKPOINT/RESUME (NEW) ====================
    resume_group = parser.add_argument_group('Checkpoint/Resume')
    resume_group.add_argument('--resume', action='store_true', default=True,
                             help='Resume from checkpoint if available [DEFAULT]')
    resume_group.add_argument('--no-resume', action='store_true',
                             help='Start fresh, ignore checkpoints')
    
    # ==================== OTHER ====================
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--jpeg-quality', type=int, default=95,
                       help='JPEG output quality 1-100 (default: 95)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)
    
    # Handle flags
    use_turbo = args.turbo and not args.no_turbo
    use_resume = args.resume and not args.no_resume
    keep_character = args.keep_character and not args.no_character
    
    # Expand wildcards and validate video files
    video_files = []
    for pattern in args.videos:
        if os.path.exists(pattern) and os.path.isfile(pattern):
            video_files.append(pattern)
        else:
            matches = glob.glob(pattern)
            if matches:
                video_files.extend([f for f in matches if os.path.isfile(f)])
            else:
                print(f"⚠️  Warning: No files found matching: {pattern}")
    
    # Filter valid video files
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    video_files = list(set([f for f in video_files if any(f.lower().endswith(ext) for ext in valid_extensions)]))
    
    if not video_files:
        print(f"❌ Error: No valid video files found")
        print(f"   Supported formats: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    # Print configuration
    print("="*60)
    print("🌾 LORA-HARVESTER v3.0 - CLI Mode")
    print("="*60)
    
    if len(video_files) == 1:
        print(f"📹 Single video mode")
        print(f"   Input: {video_files[0]}")
    else:
        print(f"📹 Batch processing mode")
        print(f"   Videos to process: {len(video_files)}")
        for i, vf in enumerate(video_files[:5], 1):
            print(f"   {i}. {Path(vf).name}")
        if len(video_files) > 5:
            print(f"   ... and {len(video_files) - 5} more")
    
    print(f"\n📁 Output: {args.output}")
    print(f"📐 Format: {args.format}")
    print(f"⏱️  Frame Interval: {args.interval}")
    print(f"🎯 Confidence: {args.confidence}")
    print(f"📏 Padding: {args.padding}px")
    print(f"🔤 Skip Text: {not args.no_skip_text}")
    print(f"⚡ Turbo Mode: {use_turbo}")
    
    if args.quality:
        print(f"\n🔍 Quality Analysis: ENABLED")
        print(f"   Blur threshold: {args.blur_threshold}")
        print(f"   Brightness: {args.brightness_min}-{args.brightness_max}")
        print(f"   Skip duplicates: {args.no_duplicates}")
    
    if args.scene_detection:
        print(f"\n🎬 Scene Detection: ENABLED")
        print(f"   Threshold: {args.scene_threshold}")
    
    if args.caption:
        print(f"\n📝 Auto Captioning: ENABLED")
        print(f"   Mode: {args.caption_mode}")
        print(f"   WD14: {'✓' if not args.no_wd14 else '✗'} ({args.wd14_model})")
        if args.trigger_word:
            print(f"   Trigger: '{args.trigger_word}'")
        print(f"   Max tags: {args.max_tags}")
        if args.preset:
            print(f"   Preset: {args.preset}")
        if args.negative_tags:
            neg_list = parse_negative_tags(args.negative_tags)
            print(f"   Negative tags: {len(neg_list)} tags")
    
    if args.ensemble:
        print(f"\n🤖 Ensemble Mode: ENABLED")
        print(f"   Models: {', '.join(args.ensemble_models)}")
        print(f"   Voting: {args.voting_threshold}/{len(args.ensemble_models)}")
    else:
        print(f"\n🤖 Single Model: {args.model}")
    
    print("="*60)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  Running on CPU (slower)")
    except ImportError:
        print("❌ PyTorch not found. Please install requirements.")
        sys.exit(1)
    
    print()
    
    # Initialize components
    print("🔄 Initializing AI models...")
    
    try:
        # Initialize detector
        if args.ensemble:
            from src.core.ensemble_detector import EnsembleDetector
            detector = EnsembleDetector(
                models_to_use=args.ensemble_models,
                confidence_threshold=args.confidence,
                voting_threshold=args.voting_threshold
            )
        else:
            detector = ObjectDetector(model_size=args.model, confidence=args.confidence)
        
        # Initialize text detector
        text_detector = SubtitleDetector() if not args.no_skip_text else None
        
        # Initialize cropper
        cropper = SmartCropper(target_format=args.format, min_padding=args.padding)
        
        # Initialize captioner if enabled
        captioner = None
        if args.caption:
            # Build tag settings
            if args.preset:
                tag_settings = CAPTIONER_PRESETS[args.preset]
                # Override with command line args
                tag_settings.trigger_word = args.trigger_word or tag_settings.trigger_word
                tag_settings.max_tags = args.max_tags
                tag_settings.min_confidence = args.min_confidence
            else:
                tag_settings = TagSettings(
                    trigger_word=args.trigger_word,
                    max_tags=args.max_tags,
                    min_confidence=args.min_confidence,
                    negative_tags=parse_negative_tags(args.negative_tags),
                    priority_tags=parse_negative_tags(args.priority_tags),
                    keep_character_tags=keep_character,
                    keep_series_tags=args.keep_series,
                    include_quality_tags=args.include_quality,
                    include_rating_tags=args.include_rating,
                    tag_separator=args.tag_separator,
                    use_underscores=not args.no_underscores,
                    caption_prefix=args.caption_prefix,
                    caption_suffix=args.caption_suffix
                )
            
            captioner = AdvancedCaptioner(
                wd14_model=args.wd14_model,
                tag_settings=tag_settings,
                enable_wd14=not args.no_wd14,
            )
        
        # Build upscaler if requested (v3.x)
        upscaler = None
        if getattr(args, 'upscale', False):
            try:
                from src.core.upscaler import FrameUpscaler
                upscaler = FrameUpscaler(
                    model_name=args.upscale_model,
                    tile=args.upscale_tile,
                    use_gpu=True,
                    face_enhance=args.face_enhance,
                )
                if upscaler.is_available():
                    print(f"✅ Upscaler ready: {args.upscale_model} ({upscaler.get_scale()}x)")
                else:
                    print("⚠️  Upscaler deps missing — upscale disabled.")
                    print("    Install: pip install realesrgan basicsr")
                    upscaler = None
            except Exception as e:
                print(f"⚠️  Upscaler init failed: {e} — upscale disabled.")
                upscaler = None

        print("✅ Models loaded successfully!")
        print()

        # Use UnifiedVideoProcessor when upscale is active (has upscaler support).
        # Fall back to EnhancedVideoProcessor for the legacy path.
        if upscaler is not None:
            quality_analyzer = None
            if args.quality:
                from src.core.quality_analyzer import QualityAnalyzer
                quality_analyzer = QualityAnalyzer(
                    blur_threshold=args.blur_threshold,
                    brightness_range=(args.brightness_min, args.brightness_max),
                    duplicate_threshold=args.duplicate_threshold,
                    check_duplicates=args.no_duplicates,
                )
            processor = UnifiedVideoProcessor(
                video_paths=video_files,
                output_dir=args.output,
                detector=detector,
                text_detector=text_detector,
                cropper=cropper,
                use_turbo=use_turbo,
                batch_size=args.batch_size,
                quality_analyzer=quality_analyzer,
                jpeg_quality=args.jpeg_quality,
                upscaler=upscaler,
                upscale_target=args.upscale_target,
                upscale_min_resolution=args.upscale_min_res,
            )
        else:
            # Create enhanced processor (legacy path — no upscale)
            processor = EnhancedVideoProcessor(
                video_paths=video_files,
                output_dir=args.output,
                detector=detector,
                text_detector=text_detector,
                cropper=cropper,
                use_turbo=use_turbo,
                batch_size=args.batch_size,
                enable_quality_check=args.quality,
                enable_scene_detection=args.scene_detection,
                jpeg_quality=args.jpeg_quality
            )
        
        # Update quality analyzer settings if enabled
        if args.quality and processor.quality_analyzer:
            processor.quality_analyzer.blur_threshold = args.blur_threshold
            processor.quality_analyzer.brightness_range = (args.brightness_min, args.brightness_max)
            if args.no_duplicates:
                processor.quality_analyzer.duplicate_threshold = args.duplicate_threshold

        # Update scene detector settings if enabled
        # (UnifiedVideoProcessor has no scene_detector — guard with getattr)
        if args.scene_detection and getattr(processor, 'scene_detector', None):
            processor.scene_detector.threshold = args.scene_threshold
        
        print("🎬 Starting video processing...")
        print()
        
        # Process all videos
        # (resume/checkpoint is only supported by EnhancedVideoProcessor)
        process_kwargs = {
            'frame_interval': args.interval,
            'skip_text': not args.no_skip_text,
        }
        if isinstance(processor, EnhancedVideoProcessor):
            process_kwargs['resume'] = use_resume
        overall_stats = processor.process_all_videos(**process_kwargs)
        
        # Run captioning on output if enabled
        if captioner and overall_stats['total_frames_saved'] > 0:
            print()
            print("="*60)
            print("📝 Running Auto-Captioning on saved frames...")
            print("="*60)
            
            # Caption each output directory
            for video_stat in overall_stats.get('videos_stats', []):
                # EnhancedVideoProcessor uses 'video', UnifiedVideoProcessor uses 'video_name'
                raw_name = video_stat.get('video') or video_stat.get('video_name') or ''
                video_name = Path(raw_name).stem if raw_name else ''
                if not video_name:
                    continue
                
                # Find output directories
                output_base = Path(args.output)
                for subdir in output_base.iterdir():
                    if subdir.is_dir() and video_name in subdir.name:
                        for category_dir in ['persons', 'animals', 'objects']:
                            cat_path = subdir / category_dir
                            if cat_path.exists():
                                print(f"\n📂 Captioning: {cat_path}")
                                captioner.caption_directory(
                                    str(cat_path),
                                    mode=args.caption_mode,
                                    overwrite=False,
                                    save_json=args.save_json
                                )
        
        print()
        print("="*60)
        print("✅ All processing complete!")
        print(f"📊 Results saved to: {args.output}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
