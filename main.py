"""
LoRA-Harvester - Main Application
AI-Powered Dataset Collection Tool for LoRA Training
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.main_window import create_app


def main():
    """Main application entry point"""
    print("="*60)
    print("🌾 LORA-HARVESTER - AI Powered Dataset Collection")
    print("="*60)
    print("🚀 Starting application...")
    print()
    
    # Check CUDA availability
    try:
        import torch
        print("🔍 Checking GPU availability...")
        print(f"   PyTorch version: {torch.__version__}")
        
        # Check CUDA support
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")
        
        # Try to get CUDA version
        try:
            import torch.version
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"   CUDA compiled version: {torch.version.cuda}")
        except:
            pass
        
        if cuda_available:
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   Current Device: {torch.cuda.current_device()}")
        else:
            print("⚠️  No GPU detected. Possible reasons:")
            print("   1. NVIDIA GPU driver not installed")
            print("   2. CUDA toolkit not installed")
            print("   3. PyTorch installed without CUDA support")
            print("   4. GPU not compatible with CUDA")
            print()
            print("🔧 To fix:")
            print("   1. Install NVIDIA GPU drivers")
            print("   2. Install CUDA toolkit 11.8+")
            print("   3. Reinstall PyTorch with CUDA:")
            print("      pip uninstall torch torchvision")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print()
            print("   Running on CPU (slower but functional)")
    except ImportError:
        print("⚠️  PyTorch not installed. Please install requirements first.")
        print("   Run: pip install -r requirements.txt")
        return
    
    print()
    print("📱 Features:")
    print("   • AI-powered object detection (YOLOv8)")
    print("   • Smart cropping with head space awareness")
    print("   • Automatic subtitle detection & skipping")
    print("   • Multiple vertical formats (9:16, 3:4, 1:1, 4:5)")
    print("   • GPU acceleration support")
    print("   • Drag & drop interface")
    print()
    print("="*60)
    
    # Create and run application
    app, window = create_app()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
