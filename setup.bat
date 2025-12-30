@echo off
setlocal enabledelayedexpansion
title SimpleCensor - Full Automated Setup

echo ======================================================
echo         SIMPLECENSOR FULL SYSTEM SETUP
echo ======================================================
echo.

:: 1. Initial Folders
mkdir Models 2>nul
mkdir output 2>nul

:: 2. Download and Extract Portable Python
if exist "python_portable\python.exe" goto :CHECKVENV
echo [1/4] Downloading Portable Python...
curl -L "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -o python_portable.zip
mkdir python_portable
tar -xf python_portable.zip -C python_portable
del python_portable.zip
cd python_portable
echo python311.zip > python311._pth
echo . >> python311._pth
echo import site >> python311._pth
curl -L "https://bootstrap.pypa.io/get-pip.py" -o get-pip.py
.\python.exe get-pip.py --quiet
.\python.exe -m pip install virtualenv --quiet
del get-pip.py
cd ..

:CHECKVENV
:: 3. Create the VENV
if exist "venv\Scripts\python.exe" goto :INSTALLREQS
echo [2/4] Creating Virtual Environment...
python_portable\python.exe -m virtualenv venv

:INSTALLREQS
:: 4. Install Requirements (Specifically including onnx and onnxruntime)
echo [3/4] Installing AI libraries and Runtimes...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install ultralytics gradio opencv-python numpy moviepy huggingface_hub mss PyQt5 onnx onnxruntime --only-binary :all:

:: 5. Download Models from Hugging Face
echo [4/4] Syncing models from genericgiraffe/censorship...
venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='genericgiraffe/censorship', local_dir='Models', local_dir_use_symlinks=False)"

echo.
echo ======================================================
echo            SETUP COMPLETE - CLOSING IN 3s
echo ======================================================
echo.
timeout /t 3
exit