@echo off
cd /d "%~dp0"
echo ============================================
echo    LoRA-Harvester - Batch Processing
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
echo.
echo ============================================
echo   Batch Video Processing Wizard
echo ============================================
echo.
echo This script will help you process multiple videos at once.
echo.
echo Usage Examples:
echo   1. Process all MP4 files in current folder
echo   2. Process videos in a specific folder
echo   3. Custom settings with high quality
echo.

:menu
echo ============================================
echo Select an option:
echo ============================================
echo.
echo [1] Process all *.mp4 files in current folder
echo [2] Process all videos in 'videos' folder
echo [3] Custom: Enter folder path
echo [4] Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto option1
if "%choice%"=="2" goto option2
if "%choice%"=="3" goto option3
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
goto menu

:option1
echo.
echo Processing all *.mp4 files in current folder...
echo.
set /p format="Enter format (9:16, 3:4, 1:1, 4:5, 16:9, 4:3) [default: 1:1]: "
if "%format%"=="" set format=1:1
set /p interval="Enter frame interval [default: 30]: "
if "%interval%"=="" set interval=30
set /p quality="Use high quality mode? (y/n) [default: y]: "
if "%quality%"=="" set quality=y

if /i "%quality%"=="y" (
    echo Running in HIGH QUALITY mode...
    %PYTHON_CMD% cli.py *.mp4 -f %format% -i %interval% --ensemble --turbo
) else (
    echo Running in STANDARD mode...
    %PYTHON_CMD% cli.py *.mp4 -f %format% -i %interval% --turbo
)
goto end

:option2
echo.
echo Processing all videos in 'videos' folder...
echo.
if not exist "videos" (
    echo ERROR: 'videos' folder not found!
    pause
    goto menu
)
set /p format="Enter format (9:16, 3:4, 1:1, 4:5, 16:9, 4:3) [default: 1:1]: "
if "%format%"=="" set format=1:1
set /p interval="Enter frame interval [default: 30]: "
if "%interval%"=="" set interval=30
set /p quality="Use high quality mode? (y/n) [default: y]: "
if "%quality%"=="" set quality=y

if /i "%quality%"=="y" (
    echo Running in HIGH QUALITY mode...
    %PYTHON_CMD% cli.py videos/*.mp4 -f %format% -i %interval% --ensemble --turbo
) else (
    echo Running in STANDARD mode...
    %PYTHON_CMD% cli.py videos/*.mp4 -f %format% -i %interval% --turbo
)
goto end

:option3
echo.
set /p folder="Enter folder path: "
if not exist "%folder%" (
    echo ERROR: Folder not found!
    pause
    goto menu
)
set /p format="Enter format (9:16, 3:4, 1:1, 4:5, 16:9, 4:3) [default: 1:1]: "
if "%format%"=="" set format=1:1
set /p interval="Enter frame interval [default: 30]: "
if "%interval%"=="" set interval=30
set /p confidence="Enter confidence threshold (0.1-0.9) [default: 0.5]: "
if "%confidence%"=="" set confidence=0.5
set /p quality="Use high quality mode? (y/n) [default: y]: "
if "%quality%"=="" set quality=y

if /i "%quality%"=="y" (
    echo Running in HIGH QUALITY mode...
    %PYTHON_CMD% cli.py "%folder%"/*.mp4 -f %format% -i %interval% -c %confidence% --ensemble --turbo
) else (
    echo Running in STANDARD mode...
    %PYTHON_CMD% cli.py "%folder%"/*.mp4 -f %format% -i %interval% -c %confidence% --turbo
)
goto end

:end
echo.
echo ============================================
echo Processing Complete!
echo ============================================
echo.
echo Check the 'output' folder for results.
echo.
pause
goto exit

:exit
exit /b 0
