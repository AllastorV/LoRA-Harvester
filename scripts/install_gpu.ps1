#!/usr/bin/env powershell
# LoRA-Harvester GPU Setup Script (PowerShell)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "[*] LoRA-Harvester GPU Setup Script" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

Write-Host "[*] Checking current PyTorch installation..." -ForegroundColor Cyan
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>$null
    Write-Host "Current PyTorch: $torchVersion" -ForegroundColor Green
} catch {
    Write-Host "[X] PyTorch not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "[*] Uninstalling CPU-only PyTorch..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

Write-Host ""
Write-Host "[*] Installing PyTorch with CUDA support..." -ForegroundColor Cyan
Write-Host "   This may take a few minutes..." -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Write-Host ""
Write-Host "[*] Testing GPU installation..." -ForegroundColor Cyan
python check_gpu.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "[OK] Installation complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. If you see CUDA available: True above, you're ready!" -ForegroundColor Green
Write-Host "2. If CUDA is still False, you may need to:" -ForegroundColor Yellow
Write-Host "   - Install NVIDIA GPU drivers" -ForegroundColor Gray
Write-Host "   - Install CUDA toolkit 11.8+" -ForegroundColor Gray
Write-Host "   - Restart your computer" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to continue"