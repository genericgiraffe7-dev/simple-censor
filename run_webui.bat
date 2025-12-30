@echo off
title SimpleCensor WebUI Launcher
setlocal

:: Ensure the script runs from the current directory
cd /d "%~dp0"

:: 1. Check if the environment exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run 'setup_env.bat' first to install requirements.
    echo.
    pause
    exit /b
)

:: 2. Launch the WebUI
echo Launching SimpleCensor WebUI...
venv\Scripts\python.exe editgui.py

:: 3. Keep window open if the app crashes
if %ERRORLEVEL% neq 0 (
    echo.
    echo [CRASH] The application closed with an error.
    pause
)

endlocal