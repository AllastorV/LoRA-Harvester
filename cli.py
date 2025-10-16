"""
Command Line Interface for LoRA-Harvester
AI-Powered Dataset Collection Tool for LoRA Training
Supports single video and batch processing
"""

import argparse
import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.detector import ObjectDetector
from src.core.text_detector import SubtitleDetector
from src.core.cropper import SmartCropper
from src.core.unified_processor import UnifiedVideoProcessor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="🌾 LoRA-Harvester - AI Powered Dataset Collection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python cli.py input.mp4 -o output_folder
  
  # Batch processing (multiple videos)
  python cli.py video1.mp4 video2.mp4 video3.mp4
  python cli.py *.mp4
  python cli.py folder/*.mp4
  
  # Custom settings
  python cli.py input.mp4 -o output -f 9:16 -i 30 -c 0.6 -p 250
  
  # High quality mode with ensemble
  python cli.py input.mp4 -o output -i 15 --ensemble --turbo
  
  # Batch process all videos in a folder
  python cli.py videos/*.mp4 -f 1:1 -i 20 --turbo
        """
    )
    
    # Required arguments - now accepts multiple videos
    parser.add_argument('videos', nargs='+', help='Input video file path(s) - supports wildcards')
    
    # Optional arguments
    parser.add_argument('-o', '--output', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('-f', '--format', default='9:16',
                       choices=['9:16', '3:4', '1:1', '4:5', '16:9', '4:3'],
                       help='Output aspect ratio (default: 9:16)')
    parser.add_argument('-i', '--interval', type=int, default=30,
                       help='Frame interval - process every N frames (default: 30)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Detection confidence threshold 0-1 (default: 0.5)')
    parser.add_argument('-p', '--padding', type=int, default=500,
                       help='Minimum padding around objects in pixels (default: 500)')
    parser.add_argument('--no-skip-text', action='store_true',
                       help='Process frames with text/subtitles (default: skip them)')
    parser.add_argument('-m', '--model', default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLO model size (default: yolov8n.pt, larger=more accurate but slower)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Enable ensemble mode (YOLO + DETR + Faster R-CNN for higher accuracy)')
    parser.add_argument('--ensemble-models', nargs='+', 
                       choices=['yolo', 'detr', 'fasterrcnn'],
                       default=['yolo', 'detr', 'fasterrcnn'],
                       help='Models to use in ensemble mode')
    parser.add_argument('--voting-threshold', type=int, default=2,
                       help='Minimum model agreements for ensemble (default: 2)')
    parser.add_argument('--turbo', action='store_true', default=True,
                       help='Enable turbo mode (optimized batch processing, 2-3x faster) [DEFAULT]')
    parser.add_argument('--no-turbo', action='store_true',
                       help='Disable turbo mode (use standard frame-by-frame processing)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for turbo mode (default: 4)')
    
    args = parser.parse_args()
    
    # Handle turbo flag
    use_turbo = args.turbo and not args.no_turbo
    
    # Expand wildcards and validate video files
    video_files = []
    for pattern in args.videos:
        # Try direct path first
        if os.path.exists(pattern) and os.path.isfile(pattern):
            video_files.append(pattern)
        else:
            # Try glob pattern
            matches = glob.glob(pattern)
            if matches:
                video_files.extend([f for f in matches if os.path.isfile(f)])
            else:
                print(f"⚠️  Warning: No files found matching: {pattern}")
    
    # Remove duplicates and validate
    video_files = list(set(video_files))
    
    # Filter to only video files
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    video_files = [f for f in video_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    if not video_files:
        print(f"❌ Error: No valid video files found")
        print(f"   Supported formats: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    # Print configuration
    print("="*60)
    print("🎬 LORA-HARVESTER - CLI Mode")
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
    if use_turbo:
        print(f"   Batch size: {args.batch_size}")
    
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
        # Initialize detector (ensemble or single)
        if args.ensemble:
            from src.core.ensemble_detector import EnsembleDetector
            detector = EnsembleDetector(
                models_to_use=args.ensemble_models,
                confidence_threshold=args.confidence,
                voting_threshold=args.voting_threshold
            )
        else:
            detector = ObjectDetector(model_size=args.model, confidence=args.confidence)
        
        text_detector = SubtitleDetector() if not args.no_skip_text else None
        cropper = SmartCropper(target_format=args.format, min_padding=args.padding)
        
        print("✅ Models loaded successfully!")
        print()
        
        # Create unified processor (handles both single and batch)
        processor = UnifiedVideoProcessor(
            video_paths=video_files,
            output_dir=args.output,
            detector=detector,
            text_detector=text_detector,
            cropper=cropper,
            use_turbo=use_turbo,
            batch_size=args.batch_size
        )
        
        print("🎬 Starting video processing...")
        print()
        
        # Process all videos
        overall_stats = processor.process_all_videos(
            frame_interval=args.interval,
            skip_text=not args.no_skip_text
        )
        
        print()
        print("✅ All processing complete!")
        print(f"📊 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
