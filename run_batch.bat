@echo off
REM LoRA-Harvester - Batch Video Processing
REM Process multiple videos at once

echo ============================================================
echo        LORA-HARVESTER - Batch Video Processor
echo ============================================================
echo.
echo This script allows you to process multiple videos at once.
echo.
echo Usage:
echo   1. Place all videos in a folder
echo   2. Run this script
echo   3. Select the folder or videos
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Examples:
echo Choose processing mode:
echo   1. Process all MP4 files in current folder
echo   2. Process specific videos
echo   3. High quality mode (Ensemble)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Processing all MP4 files in current folder...
    python cli.py *.mp4 -f 1:1 -i 30 --turbo
) else if "%choice%"=="2" (
    echo.
    echo Enter video paths separated by space:
    set /p videos="Video paths: "
    python cli.py %videos% -f 1:1 -i 30 --turbo
) else if "%choice%"=="3" (
    echo.
    echo Processing with Ensemble mode (higher accuracy)...
    python cli.py *.mp4 -f 1:1 -i 20 --ensemble --turbo
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Processing complete! Check the 'output' folder.
echo ============================================================
pause
