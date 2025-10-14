@echo off
echo ================================================
echo Video Smart Cropper - ENSEMBLE MODE
echo ================================================
echo.

echo This script will run the app with ensemble mode enabled
echo for highest accuracy detection using 3 AI models.
echo.

call venv\Scripts\activate.bat

echo Starting application with ensemble mode...
echo.
echo Note: First run will download additional models:
echo   - DETR (Facebook/Meta Transformer)
echo   - Faster R-CNN (Torchvision)
echo.
echo This may take a few minutes...
echo.

python main.py

pause
