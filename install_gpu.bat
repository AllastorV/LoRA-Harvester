@echo off
echo ============================================================
echo 🔧 LoRA-Harvester GPU Setup Script
echo ============================================================
echo.

echo 🔍 Checking current PyTorch installation...
python -c "import torch; print(f'Current PyTorch: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch not found
) else (
    echo ✅ PyTorch found
)

echo.
echo 🗑️  Uninstalling CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo 📥 Installing PyTorch with CUDA support...
echo    This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 🧪 Testing GPU installation...
python check_gpu.py

echo.
echo ============================================================
echo 🎉 Installation complete!
echo ============================================================
echo.
echo Next steps:
echo 1. If you see CUDA available: True above, you're ready!
echo 2. If CUDA is still False, you may need to:
echo    - Install NVIDIA GPU drivers
echo    - Install CUDA toolkit 11.8+
echo    - Restart your computer
echo.
pause