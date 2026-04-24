@echo off
cd /d "%~dp0"
echo ============================================
echo    LoRA-Harvester - Video Smart Cropper
echo ============================================
echo.

REM ── Try venv first ───────────────────────────
if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
    goto :run
)

REM ── Fall back to system Python ───────────────
set PYTHON_CMD=
for %%p in (python python3) do (
    where %%p >nul 2>&1 && set PYTHON_CMD=%%p && goto :run
)

echo [ERROR] Python not found! Install Python 3.10+ or run install.bat first.
pause
exit /b 1

:run
echo [*] Python: %PYTHON_CMD%
echo [*] Starting...
echo.

%PYTHON_CMD% main.py

REM If Python exits with an error code, keep window open so user can read the message
if errorlevel 1 (
    echo.
    echo ============================================
    echo [ERROR] Application crashed - see above
    echo ============================================
    pause
)
