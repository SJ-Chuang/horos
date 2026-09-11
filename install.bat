@echo off
REM horos installer - Windows.
REM Creates .\.venv, installs the horos core, then runs `horos install`, which
REM detects the GPU / CUDA version and installs the matching ML stack (on a
REM CUDA machine torch comes from the matching PyTorch index - the plain PyPI
REM Windows wheel is CPU-only). All platform logic lives in horos itself
REM (horos/api/install.py) - this script only bootstraps the venv.
REM If no usable Python is present, the script offers to install one (asking
REM first) and to add it to the user PATH.
REM Linux / macOS / Jetson: use install.sh instead.
REM
REM Non-interactive use (CI): set HOROS_AUTO_INSTALL_PYTHON=1 to answer "yes"
REM to both questions without prompting.
setlocal

REM ============================================================
REM Python >= 3.10
REM ============================================================
REM Do not trust `where python`: on a fresh Windows install `python` resolves
REM to the Microsoft Store placeholder (an App Execution Alias) that is on PATH
REM but is not an interpreter - it prints "Python was not found" and exits
REM non-zero. Probe the candidates by actually running them, and also accept
REM the `py` launcher, which python.org installers put on PATH.
call :find_python
if defined PY goto have_python

echo.
if defined PY_TOO_OLD (
    echo horos needs Python ^>= 3.10, but the Python on PATH is older:
    python -V
) else (
    echo No working Python found. Any `python` on PATH is the Microsoft Store
    echo placeholder ^(an App Execution Alias^), not an interpreter.
)
echo.

REM ---------- Question 1: install Python? ----------
REM PYVER and PYDIR below must stay in sync: a per-user 3.12.x install always
REM lands in Programs\Python\Python312 regardless of the patch level.
set "PYVER=3.12.10"
if defined HOROS_AUTO_INSTALL_PYTHON goto ask_path
choice /C YN /N /M "Install Python %PYVER% now (per-user, no admin rights needed)? [Y/N] "
if errorlevel 2 (
    echo.
    echo Install Python ^>= 3.10 yourself, then run install.bat again:
    echo     winget install --id Python.Python.3.12
    echo   or https://www.python.org/downloads/windows/ - tick "Add python.exe to PATH".
    exit /b 1
)

REM ---------- Question 2: add to PATH? ----------
:ask_path
set "PREPEND=1"
if defined HOROS_AUTO_INSTALL_PYTHON goto do_install
choice /C YN /N /M "Add Python to your user PATH so `python` works in every new terminal? [Y/N] "
if errorlevel 2 set "PREPEND=0"

:do_install
set "PYDIR=%LOCALAPPDATA%\Programs\Python\Python312"
set "PYARGS=/quiet InstallAllUsers=0 PrependPath=%PREPEND% Include_launcher=1 Include_test=0"
set "PYARCH=amd64"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "PYARCH=arm64"

REM Prefer winget (ships with Windows 10 1709+ / 11); it verifies the
REM installer hash. --override replaces the manifest's silent switches, so
REM /quiet must be repeated.
where winget >nul 2>nul
if errorlevel 1 goto install_from_python_org
echo.
echo Installing Python %PYVER% with winget ...
winget install --id Python.Python.3.12 --exact --source winget --accept-source-agreements --accept-package-agreements --override "%PYARGS%"
if exist "%PYDIR%\python.exe" goto python_installed
echo winget did not produce a Python install - falling back to python.org.

:install_from_python_org
set "PYEXE=%TEMP%\python-%PYVER%-%PYARCH%.exe"
echo.
echo Downloading https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-%PYARCH%.exe ...
curl.exe -L -# -o "%PYEXE%" "https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-%PYARCH%.exe"
if errorlevel 1 goto error
echo Running the installer ...
"%PYEXE%" %PYARGS%
if errorlevel 1 goto error
del "%PYEXE%" >nul 2>nul

:python_installed
REM The installer edited the registry PATH; this cmd session still has the old
REM one, so make the new interpreter visible here as well.
set "PATH=%PYDIR%;%PYDIR%\Scripts;%PATH%"
call :find_python
if not defined PY (
    echo ERROR: Python was installed but could not be found at %PYDIR%.
    echo Open a new terminal and run install.bat again.
    exit /b 1
)
echo.
echo Python installed:
%PY% -V
if "%PREPEND%"=="1" (
    echo Added to your user PATH - takes effect in terminals opened from now on.
) else (
    echo Not added to PATH. It lives in %PYDIR%; horos itself is used through
    echo .venv\Scripts\activate, which does not need it on PATH.
)

:have_python
echo.
echo Using Python:
%PY% -c "import sys; print('  ' + sys.executable + '  (' + sys.version.split()[0] + ')')"

REM ============================================================
REM Virtual environment
REM ============================================================
if defined VIRTUAL_ENV (
    echo Using the already-activated virtualenv: %VIRTUAL_ENV%
    set "VPY=%PY%"
) else (
    if not exist .venv (
        echo Creating .venv ...
        %PY% -m venv .venv
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

REM ============================================================
REM :find_python - sets PY to a working interpreter command (>= 3.10), or
REM leaves it empty. Sets PY_TOO_OLD when an interpreter exists but is too old.
REM ============================================================
:find_python
set "PY="
set "PY_TOO_OLD="
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 3)" >nul 2>nul
if not errorlevel 1 ( set "PY=python" & goto :eof )
REM exit code 3 = real interpreter, too old; 9009 = the Store placeholder
if errorlevel 3 if not errorlevel 4 set "PY_TOO_OLD=1"
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 3)" >nul 2>nul
if not errorlevel 1 ( set "PY=py -3" & set "PY_TOO_OLD=" & goto :eof )
if errorlevel 3 if not errorlevel 4 set "PY_TOO_OLD=1"
goto :eof
