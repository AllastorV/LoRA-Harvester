@echo off
cd /d "%~dp0"
echo ============================================
echo    LoRA-Harvester - Video Smart Cropper
echo ============================================
echo.

REM ── 1. Project venv wins if present ──────────
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
    goto :run
)

REM ── 2. Force Python 3.10 via the py launcher ─
REM    CRITICAL: the full dependency stack (dghs-imgutils for anime
REM    detection, realesrgan for upscale, numpy<2) is installed ONLY in
REM    Python 3.10. Running under another version (e.g. the Store 3.13 that
REM    "py main.py" picks by default) silently disables anime detection and
REM    upscaling -> 0 frames saved. So we pin 3.10 explicitly.
py -3.10 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    goto :run
)

REM ── 3. Fall back to PATH python (usually 3.10) ─
set "PYTHON_CMD="
for %%p in (python python3) do (
    where %%p >nul 2>&1 && set "PYTHON_CMD=%%p" && goto :run
)

echo [ERROR] Python 3.10 not found! Install Python 3.10 or run install.bat first.
pause
exit /b 1

:run
echo [*] Python: %PYTHON_CMD%
%PYTHON_CMD% -c "import sys;print('[*] Interpreter:',sys.executable);print('[*] Version:',sys.version.split()[0])"
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
