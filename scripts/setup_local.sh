#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------------------------------------------------
# setup_local.sh
#
# Set up a local development virtual environment, clean build artifacts,
# and install horos in editable mode.
#
# The default (full) mode delegates the install to the user-facing installer
# — ./install.sh (Ubuntu / macOS / Jetson) or install.bat (Windows, when run
# from Git Bash / MSYS) — so developers exercise exactly the path users take:
# .venv, the horos core, then `horos install` picking the platform-correct ML
# stack (torch source, rfdetr pin, albumentations, transformers). Afterwards
# `horos doctor` runs as the installation check; a missing or mis-built
# dependency fails this script. Plain `pip install -e .` is NOT used for the
# full mode: it would pull a PyPI torch, which is wrong on Jetson and on
# Windows-with-CUDA.
#
# --light keeps a torch-free environment for dataset/annotation development
# and skips the installer and doctor entirely.
#
# Usage:
#   bash scripts/setup_local.sh                       # install.sh/.bat + doctor check
#   bash scripts/setup_local.sh --recreate            # Delete and recreate .venv
#   bash scripts/setup_local.sh --dev                 # Also install [dev] extras (pytest, ruff)
#   bash scripts/setup_local.sh --light --dev         # No ML deps: annotation-only dev loop
#   bash scripts/setup_local.sh --clean-only          # Only clean caches, no venv/install
#   bash scripts/setup_local.sh --python python3.11   # Use specific Python version
#   bash scripts/setup_local.sh --no-cache            # pip installs without the wheel cache
# ----------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RECREATE="false"
DEV="false"
LIGHT="false"
CLEAN_ONLY="false"
NO_CACHE="false"

usage() {
  cat << 'EOF'
Usage:
  bash scripts/setup_local.sh [options]

Options:
  --recreate          Delete existing .venv and recreate from scratch.
  --dev               Also install the [dev] extras (pytest, ruff).
  --light             Install the torch-free core only (no torch/rfdetr/
                      transformers) — dataset + annotation development
                      without the ~3 GB ML stack. Skips install.sh/.bat and
                      the doctor check.
  --clean-only        Only clean __pycache__, .pytest_cache, dist, build, *.egg-info.
  --python <path>     Use specified Python executable (e.g., python3.11).
  --no-cache          Run pip without its wheel cache (PIP_NO_CACHE_DIR=1).
  -h, --help          Show this help message.

Full mode runs ./install.sh (or install.bat under Git Bash / MSYS on Windows)
and then `horos doctor`; the script fails if doctor reports a missing or
mis-built dependency.

Environment variables:
  PYTHON_BIN          Python executable fallback. Default: python3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE="true"
      shift
      ;;
    --dev)
      DEV="true"
      shift
      ;;
    --light)
      LIGHT="true"
      shift
      ;;
    --clean-only)
      CLEAN_ONLY="true"
      shift
      ;;
    --python)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --python requires a value."
        exit 1
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# ------------------------------------------------------------------
# Clean build artifacts and caches
# ------------------------------------------------------------------
echo "==> Cleaning caches and build artifacts..."
find . -type d -name "__pycache__" -not -path "./$VENV_DIR/*" -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache .ruff_cache dist build ./*.egg-info

if [[ "$CLEAN_ONLY" == "true" ]]; then
  echo ""
  echo "Clean completed. (--clean-only mode, no environment setup)"
  exit 0
fi

# ------------------------------------------------------------------
# Validate Python
# ------------------------------------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN"
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: horos needs Python >= 3.10 (found $("$PYTHON_BIN" --version 2>&1))."
  exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 || echo "unknown")
echo "==> Project root: $PROJECT_ROOT"
echo "==> Python:       $PYTHON_BIN ($PYTHON_VERSION)"
echo "==> Venv dir:     $VENV_DIR"
echo "==> Recreate:     $RECREATE"
echo "==> Dev extras:   $DEV"
echo "==> Light mode:   $LIGHT"
echo "==> No cache:     $NO_CACHE"

# ------------------------------------------------------------------
# Recreate virtual environment on request
# ------------------------------------------------------------------
if [[ "$RECREATE" == "true" && -d "$VENV_DIR" ]]; then
  echo "==> Removing existing virtual environment..."
  rm -rf "$VENV_DIR"
fi

if [[ "$NO_CACHE" == "true" ]]; then
  # honoured by every pip invocation below, including those inside install.sh/.bat
  export PIP_NO_CACHE_DIR=1
fi

# The installers must create ./.venv themselves; an activated foreign venv
# would otherwise be reused, and this script is about the project venv.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "==> Ignoring the activated virtualenv ($VIRTUAL_ENV); using ./$VENV_DIR"
  unset VIRTUAL_ENV
fi

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ON_WINDOWS="true" ;;
  *) ON_WINDOWS="false" ;;
esac

if [[ "$ON_WINDOWS" == "true" ]]; then
  VENV_PY="$VENV_DIR/Scripts/python.exe"
else
  VENV_PY="$VENV_DIR/bin/python"
fi

if [[ "$LIGHT" == "true" ]]; then
  # --------------------------------------------------------------
  # Light mode: torch-free environment, no installer, no doctor
  # --------------------------------------------------------------
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    echo "==> Using existing virtual environment: $VENV_DIR"
  fi
  echo "==> Python in venv: $VENV_PY ($("$VENV_PY" --version 2>&1))"
  echo "==> Upgrading pip / setuptools / wheel..."
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
  echo "==> Installing horos in editable mode (--light: core only, no ML stack)..."
  # the core's declared dependencies are torch-free by design, so a plain
  # editable install IS the light environment — no hand-maintained dep list
  "$VENV_PY" -m pip install -e .
else
  # --------------------------------------------------------------
  # Full mode: the user-facing installer builds the environment
  # --------------------------------------------------------------
  if [[ "$ON_WINDOWS" == "true" ]]; then
    echo "==> Running install.bat (Windows installer)..."
    # install.bat finds `python` on PATH; put the requested interpreter first
    PY_DIR="$(dirname "$(command -v "$PYTHON_BIN")")"
    PATH="$PY_DIR:$PATH" cmd.exe //c install.bat
  else
    echo "==> Running install.sh (Ubuntu / macOS / Jetson installer)..."
    PYTHON="$PYTHON_BIN" bash ./install.sh
  fi
  echo "==> Python in venv: $VENV_PY ($("$VENV_PY" --version 2>&1))"
fi

# ------------------------------------------------------------------
# Dev extras (pytest, ruff) on top of whatever the installer built
# ------------------------------------------------------------------
if [[ "$DEV" == "true" ]]; then
  echo "==> Installing [dev] extras (pytest, ruff)..."
  # [dev] adds only pytest/ruff on top of the core deps; the ML stack is not in
  # the base requirements, so this cannot disturb what horos install chose
  "$VENV_PY" -m pip install -e ".[dev]"
fi

# ------------------------------------------------------------------
# Installation check: horos doctor (full mode only)
# ------------------------------------------------------------------
if [[ "$LIGHT" != "true" ]]; then
  echo ""
  echo "==> Installation check: horos doctor"
  if ! "$VENV_PY" -m horos.cli doctor; then
    echo ""
    echo "ERROR: horos doctor reports a missing or mis-built dependency (see above)."
    echo "Follow the listed fix/manual steps, then re-run this script."
    exit 1
  fi
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "==> Installed package info:"
"$VENV_PY" -m pip show horos || true

echo ""
echo "Setup completed successfully."
if [[ "$ON_WINDOWS" == "true" ]]; then
  echo "  Activate with: source $VENV_DIR/Scripts/activate"
else
  echo "  Activate with: source $VENV_DIR/bin/activate"
fi
if [[ "$DEV" == "true" ]]; then
  echo "  Test with:     bash scripts/local_test.sh"
else
  echo "  Quick check:   bash scripts/local_test.sh --quick"
  echo "  Full tests:    re-run with --dev first (installs pytest, ruff)"
fi
