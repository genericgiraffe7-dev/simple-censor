@echo off
setlocal enabledelayedexpansion
title SimpleCensor - Automated Venv Setup

echo ======================================================
echo         SIMPLECENSOR ISOLATED VENV SETUP
echo ======================================================
echo.

:: 1. Download Base Portable Python
if not exist "python_base" (
    echo [1/4] Downloading Base Portable Python 3.11...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile 'python_base.zip'"
    echo [1/4] Extracting Python...
    powershell -Command "Expand-Archive -Path 'python_base.zip' -DestinationPath 'python_base' -Force"
    del python_base.zip
    
    :: Enable site-packages for the base so we can install virtualenv
    echo [1/4] Configuring base environment...
    cd python_base
    powershell -Command "(Get-Content python311._pth) -replace '#import site', 'import site' | Set-Content python311._pth"
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
    .\python.exe get-pip.py --quiet
    .\python.exe -m pip install virtualenv --quiet
    del get-pip.py
    cd ..
) else (
    echo [1/4] Base Python already exists.
)

:: 2. Create the VENV
if not exist "venv" (
    echo [2/4] Creating Virtual Environment (venv)...
    python_base\python.exe -m virtualenv venv
) else (
    echo [2/4] Virtual environment already exists.
)

:: 3. Install Requirements into VENV
echo [3/4] Installing AI libraries into venv...
echo Note: This may take several minutes depending on your internet speed.
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\python.exe -m pip install ultralytics gradio opencv-python numpy moviepy huggingface_hub onnxruntime-gpu --quiet

:: 4. Final Folder Prep
if not exist "Models" mkdir Models
if not exist "output" mkdir output

echo.
echo ======================================================
echo            SETUP COMPLETE SUCCESSFULLY
echo ======================================================
echo.
echo Environment is ready. You can now use run_app.bat to start.
echo.
pause