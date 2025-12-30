@echo off
title SimpleCensor Live Overlay
setlocal

cd /d "%~dp0"

:: 1. Check for venv
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found! Run setup_env.bat first.
    pause
    exit /b
)

:: 2. Launch
echo Initializing Live Screen Censor...
echo Ensure your models are downloaded in the \Models folder.
venv\Scripts\python.exe desktop.py

pause