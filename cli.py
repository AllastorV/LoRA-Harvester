"""
Command Line Interface for LoRA-Harvester
AI-Powered Dataset Collection Tool for LoRA Training
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.detector import ObjectDetector
from src.core.text_detector import SubtitleDetector
from src.core.cropper import SmartCropper
from src.core.video_processor import VideoProcessor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="🌾 LoRA-Harvester - AI Powered Dataset Collection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py input.mp4 -o output_folder
  
  # Custom settings
  python cli.py input.mp4 -o output -f 9:16 -i 30 -c 0.6 -p 250
  
  # Skip subtitle detection (faster)
  python cli.py input.mp4 -o output --no-skip-text
  
  # High quality mode (process more frames)
  python cli.py input.mp4 -o output -i 15
        """
    )
    
    # Required arguments
    parser.add_argument('video', help='Input video file path')
    
    # Optional arguments
    parser.add_argument('-o', '--output', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('-f', '--format', default='9:16',
                       choices=['9:16', '3:4', '1:1', '4:5'],
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
    parser.add_argument('--turbo', action='store_true',
                       help='Enable turbo mode (optimized batch processing, 2-3x faster)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for turbo mode (default: 4)')
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print("="*60)
    print("🎬 VIDEO SMART CROPPER - CLI Mode")
    print("="*60)
    print(f"📹 Input: {args.video}")
    print(f"📁 Output: {args.output}")
    print(f"📐 Format: {args.format}")
    print(f"⏱️  Frame Interval: {args.interval}")
    print(f"🎯 Confidence: {args.confidence}")
    print(f"📏 Padding: {args.padding}px")
    print(f"🔤 Skip Text: {not args.no_skip_text}")
    print(f"🤖 Model: {args.model}")
    
    if args.ensemble:
        print(f"🤖 Ensemble Mode: ENABLED")
        print(f"   Models: {', '.join(args.ensemble_models)}")
        print(f"   Voting: {args.voting_threshold}/{len(args.ensemble_models)}")
    
    print("="*60)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
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
            from src.core.detector import ObjectDetector
            detector = ObjectDetector(model_size=args.model, confidence=args.confidence)
        text_detector = SubtitleDetector() if not args.no_skip_text else None
        cropper = SmartCropper(target_format=args.format, min_padding=args.padding)
        
        # Create output directory
        video_name = Path(args.video).stem
        mode_suffix = "ensemble" if args.ensemble else "yolo"
        turbo_suffix = "_turbo" if args.turbo else ""
        output_dir = Path(args.output) / f"{video_name}_{args.format.replace(':', 'x')}_{mode_suffix}{turbo_suffix}"
        
        # Choose processor
        if args.turbo:
            from src.core.optimized_processor import TurboVideoProcessor
            print(f"⚡ TURBO MODE: Batch size = {args.batch_size}")
            processor = TurboVideoProcessor(
                args.video,
                str(output_dir),
                detector,
                text_detector,
                cropper,
                batch_size=args.batch_size
            )
        else:
            from src.core.video_processor import VideoProcessor
            processor = VideoProcessor(
                args.video,
                str(output_dir),
                detector,
                text_detector,
                cropper
            )
        
        print("✅ Models loaded successfully!")
        print()
        print("🎬 Starting video processing...")
        print()
        
        # Process video
        stats = processor.process_video(
            frame_interval=args.interval,
            skip_text=not args.no_skip_text
        )
        
        print()
        print("✅ Processing complete!")
        print(f"📊 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
