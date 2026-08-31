@echo off
REM horos installer - Windows.
REM Detects CUDA and installs the matching torch + horos into .\.venv.
REM Linux / macOS / Jetson: use install.sh instead.
setlocal enabledelayedexpansion

REM ============================================================
REM Python >= 3.10
REM ============================================================
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python not found on PATH. Install Python ^>= 3.10 first.
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: horos needs Python ^>= 3.10.
    python -V
    exit /b 1
)

REM ============================================================
REM Virtual environment
REM ============================================================
if defined VIRTUAL_ENV (
    echo Using the already-activated virtualenv: %VIRTUAL_ENV%
    set "VPY=python"
) else (
    if not exist .venv (
        echo Creating .venv ...
        python -m venv .venv
        if errorlevel 1 goto error
    )
    set "VPY=.venv\Scripts\python.exe"
)

%VPY% -m pip install --upgrade pip wheel >nul
if errorlevel 1 goto error

REM ============================================================
REM Detect CUDA (nvcc first, then nvidia-smi)
REM ============================================================
set "CUDA_VERSION=none"
where nvcc >nul 2>nul
if %errorlevel%==0 (
    for /f "tokens=* usebackq" %%i in (`powershell -NoProfile -Command "nvcc --version | Select-String 'release ([0-9]+\.[0-9]+)' | ForEach-Object { $_.Matches[0].Groups[1].Value }"`) do (
        set "CUDA_VERSION=%%i"
    )
) else (
    where nvidia-smi >nul 2>nul
    if %errorlevel%==0 (
        for /f "tokens=* usebackq" %%i in (`powershell -NoProfile -Command "nvidia-smi | Select-String 'CUDA Version:\s*([0-9]+\.[0-9]+)' | ForEach-Object { $_.Matches[0].Groups[1].Value }"`) do (
            set "CUDA_VERSION=%%i"
        )
    )
)
if "%CUDA_VERSION%"=="" set "CUDA_VERSION=none"
echo Detected CUDA version: %CUDA_VERSION%

REM ============================================================
REM torch - on Windows the default PyPI wheel is CPU-only, so a
REM CUDA machine must install from the matching PyTorch index.
REM Installed BEFORE horos so pip sees the requirement satisfied.
REM ============================================================
set "TORCH_INDEX="
if "%CUDA_VERSION%"=="none" goto torch_cpu

for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION%") do (
    set "CUDA_MAJOR=%%a"
    set "CUDA_MINOR=%%b"
)
if !CUDA_MAJOR! GEQ 13 set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
if !CUDA_MAJOR!==12 (
    if !CUDA_MINOR! GEQ 6 (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    ) else (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
    )
)
if !CUDA_MAJOR!==11 set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"

if defined TORCH_INDEX (
    echo Installing torch from !TORCH_INDEX! ...
    %VPY% -m pip install torch torchvision --index-url !TORCH_INDEX!
    if errorlevel 1 goto error
    goto torch_done
)

:torch_cpu
echo No usable CUDA detected. Installing the CPU-only torch build ...
%VPY% -m pip install torch torchvision
if errorlevel 1 goto error

:torch_done

REM ============================================================
REM horos itself (pulls rfdetr==1.9.4 and transformers)
REM ============================================================
echo Installing horos with its dependencies ...
%VPY% -m pip install -e .
if errorlevel 1 goto error

REM ============================================================
REM Verify
REM ============================================================
%VPY% -c "import sys, time; t0=time.time(); import horos; dt=time.time()-t0; assert 'torch' not in sys.modules, 'R1b violated'; print('import horos OK (%%.2fs, lazy backends intact)' %% dt)"
if errorlevel 1 goto error
%VPY% -c "import torch; print('torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"

echo.
echo horos installed.
echo Next steps:
if not defined VIRTUAL_ENV echo   .venv\Scripts\activate
echo   horos init .\my_project
echo   horos import ^<dataset dir^> --project .\my_project
echo   horos ui --project .\my_project    (open http://localhost:5000)
exit /b 0

:error
echo.
echo Installation failed - see the error above.
exit /b 1
