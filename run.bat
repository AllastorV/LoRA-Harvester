@echo off
cd /d "%~dp0"
echo ============================================
echo    LoRA-Harvester - Video Smart Cropper
echo ============================================
echo.

REM Find Python
set PYTHON_CMD=
for %%p in (python python3 python.exe) do (
    where %%p >nul 2>&1 && set PYTHON_CMD=%%p && goto :found
)

:found
if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo [*] Using: %PYTHON_CMD%
echo [*] Checking dependencies...

REM Check PyQt5
%PYTHON_CMD% -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo [*] Installing missing dependencies...
    %PYTHON_CMD% -m pip install -r requirements.txt
)

echo [*] Starting application...
echo.
%PYTHON_CMD% main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed. Check the error above.
)
pause
