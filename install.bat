@echo off
echo ================================================
echo Video Smart Cropper - Installation Script
echo ================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo To run the application:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Run the app: python main.py
echo.
echo For CLI mode: python cli.py --help
echo.
echo Note: For GPU support, ensure you have CUDA installed
echo ================================================
pause
