@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Project environment missing. Starting install.bat.
    call "%~dp0install.bat"
)
if not exist "%PYTHON%" (
    echo [ERROR] Complete install.bat, then retry.
    pause
    exit /b 1
)
"%PYTHON%" -c "from src.core.setup_manager import verified_venv; verified_venv(require_supported=False)"
if errorlevel 1 (
    echo [ERROR] Environment check failed. Open install.bat and select repair.
    pause
    exit /b 1
)
"%PYTHON%" main.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
