@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo [ERROR] Run install.bat to create the project environment first.
  pause
  exit /b 1
)
"venv\Scripts\python.exe" "scripts\ai_control_cli.py" config
pause
