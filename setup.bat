@echo off
setlocal enabledelayedexpansion
title SimpleCensor - RTX GPU Automated Setup

echo ======================================================
echo         SIMPLECENSOR RTX GPU SYSTEM SETUP
echo ======================================================
echo.

:: 1. Initial Folders
mkdir Models 2>nul
mkdir output 2>nul

:: 2. Download and Extract Portable Python
if exist "python_portable\python.exe" goto :CHECKVENV
echo [1/4] Downloading Portable Python (3.11.9)...
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
:: 4. Install Requirements (FORCING CUDA GPU)
echo [3/4] Installing RTX GPU Libraries...
venv\Scripts\python.exe -m pip install --upgrade pip

echo Installing CUDA-enabled PyTorch...
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo Installing ONNX GPU Runtime and Ultralytics...
venv\Scripts\python.exe -m pip uninstall onnxruntime -y 2>nul
venv\Scripts\python.exe -m pip install ultralytics mss PyQt5 onnx onnxruntime-gpu numpy==1.26.4 huggingface_hub

:: 5. Download Models (REFINED SYNC)
echo [4/4] Syncing models from genericgiraffe/censorship...
:: This refined command forces the download directly to the Models folder
venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='genericgiraffe/censorship', local_dir='Models', local_dir_use_symlinks=False, allow_patterns=['*.onnx', '*.pt', '*.yaml', 'config.json'])"

cls
echo ======================================================
echo           		 SETUP COMPLETE 
echo ======================================================
echo.
echo 
echo.
pause
exit