@echo off
REM horos installer - Windows.
REM Creates .\.venv, installs the horos core, then runs `horos install`, which
REM detects the GPU / CUDA version and installs the matching ML stack (on a
REM CUDA machine torch comes from the matching PyTorch index - the plain PyPI
REM Windows wheel is CPU-only). All platform logic lives in horos itself
REM (horos/api/install.py) - this script only bootstraps the venv.
REM Linux / macOS / Jetson: use install.sh instead.
setlocal

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
REM horos core (torch-free by design), then the ML stack
REM ============================================================
echo Installing the horos core ...
%VPY% -m pip install -e .
if errorlevel 1 goto error

echo Installing the ML stack (horos install) ...
%VPY% -m horos.cli install
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
echo   horos doctor                   (verify the environment)
echo   horos init .\my_project
echo   horos import ^<dataset dir^> --project .\my_project
echo   horos ui .\my_project          (open http://localhost:5000)
exit /b 0

:error
echo.
echo Installation failed - see the error above.
exit /b 1
