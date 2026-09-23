@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set "COMPONENTS=core,upscale,anime,faces"
set "CHANNEL=cpu"
nvidia-smi -L >nul 2>&1
if not errorlevel 1 (
    set "COMPONENTS=core,gpu,upscale,anime,faces"
    set "CHANNEL=cu124"
)
echo Selected PyTorch channel for this hardware: %CHANNEL%
for %%V in (3.10 3.11 3.12) do (
    py -%%V -c "import sys,struct;sys.exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1
    if not errorlevel 1 (
        py -%%V scripts\setup_wizard.py --cli --yes --components "%COMPONENTS%" --channel "%CHANNEL%"
        goto :done
    )
)
python -c "import sys,struct;sys.exit(0 if (3,10)<=sys.version_info[:2]<(3,13) and struct.calcsize('P')==8 else 1)" >nul 2>&1
if not errorlevel 1 (
    python scripts\setup_wizard.py --cli --yes --components "%COMPONENTS%" --channel "%CHANNEL%"
    goto :done
)
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" scripts\setup_wizard.py --cli --yes --components "%COMPONENTS%" --channel "%CHANNEL%"
    goto :done
)
echo [ERROR] Install 64-bit Python 3.10, 3.11 or 3.12 from python.org, then run this file again.
pause
exit /b 1
:done
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (
    echo Setup complete. Start the program with run.bat.
) else (
    echo [ERROR] Setup failed. Check the latest setup log in the logs folder.
)
pause
exit /b %RC%
