#!/usr/bin/env python3
"""
GPU Detection and Diagnostics Script for LoRA-Harvester
Comprehensive GPU and CUDA testing
"""

import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_gpu():
    print("="*60)
    print("🔍 GPU DIAGNOSTICS - LoRA-Harvester")
    print("="*60)
    
    # 1. Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed!")
        print("   Install with: pip install torch torchvision")
        return
    
    # 2. Check CUDA availability
    print(f"\n🔧 CUDA Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n📱 GPU {i}: {props.name}")
            print(f"   Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"   CUDA capability: {props.major}.{props.minor}")
            print(f"   Multiprocessors: {props.multi_processor_count}")
    else:
        print("❌ CUDA not available")
    
    # 3. Test tensor operations
    print(f"\n🧪 Testing Operations:")
    try:
        # CPU test
        x = torch.randn(1000, 1000)
        y = torch.mm(x, x)
        print("✅ CPU operations working")
        
        # GPU test
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x_gpu = x.to(device)
            y_gpu = torch.mm(x_gpu, x_gpu)
            print("✅ GPU operations working")
            
            # Memory info
            print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"   GPU memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        else:
            print("⚠️  GPU operations skipped (no CUDA)")
            
    except Exception as e:
        print(f"❌ Operation failed: {e}")
    
    # 4. Check if models can use GPU
    print(f"\n🤖 AI Model GPU Support:")
    try:
        # Test YOLOv8
        from ultralytics import YOLO
        model = YOLO(os.path.join(_PROJECT_ROOT, 'yolov8n.pt'))
        if torch.cuda.is_available():
            model.to('cuda')
            print("✅ YOLOv8 can use GPU")
        else:
            print("⚠️  YOLOv8 will use CPU")
    except Exception as e:
        print(f"⚠️  YOLOv8 test failed: {e}")
    
    # 5. Recommendations
    print(f"\n💡 Recommendations:")
    if not torch.cuda.is_available():
        print("🔧 To enable GPU acceleration:")
        print("   1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("   2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("   3. Reinstall PyTorch with CUDA:")
        print("      pip uninstall torch torchvision")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   4. Restart your computer")
    else:
        print("✅ GPU acceleration ready!")
        print("   Your system can process videos 5-10x faster")
        print("   Ensemble mode will work smoothly")
    
    print("="*60)

if __name__ == "__main__":
    check_gpu()