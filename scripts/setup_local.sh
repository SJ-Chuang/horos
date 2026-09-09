#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------------------------------------------------
# setup_local.sh
#
# Set up a local development virtual environment, clean build artifacts,
# and install horos in editable mode.
#
# NOTE: plain `pip install -e .` pulls torch from PyPI, which is wrong on
# Jetson and on Windows-with-CUDA. For a platform-aware torch install use
# ./install.sh at the repo root; this script is the developer-loop tool.
#
# Usage:
#   bash scripts/setup_local.sh                       # Create .venv (if missing), install package
#   bash scripts/setup_local.sh --recreate            # Delete and recreate .venv
#   bash scripts/setup_local.sh --dev                 # Install with [dev] extras (pytest, ruff)
#   bash scripts/setup_local.sh --light --dev         # No ML deps: annotation-only dev loop
#   bash scripts/setup_local.sh --clean-only          # Only clean caches, no venv/install
#   bash scripts/setup_local.sh --python python3.11   # Use specific Python version
#   bash scripts/setup_local.sh --no-cache            # pip install with --no-cache-dir
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
  --dev               Install with [dev] extras (pip install -e ".[dev]").
  --light             Install with --no-deps plus the light runtime deps only
                      (no torch/rfdetr/transformers) — dataset + annotation
                      development without the ~3 GB ML stack.
  --clean-only        Only clean __pycache__, .pytest_cache, dist, build, *.egg-info.
  --python <path>     Use specified Python executable (e.g., python3.11).
  --no-cache          Pass --no-cache-dir to pip install.
  -h, --help          Show this help message.

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
# Create / recreate virtual environment
# ------------------------------------------------------------------
if [[ "$RECREATE" == "true" && -d "$VENV_DIR" ]]; then
  echo "==> Removing existing virtual environment..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "==> Using existing virtual environment: $VENV_DIR"
fi

# ------------------------------------------------------------------
# Activate
# ------------------------------------------------------------------
echo "==> Activating virtual environment..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Python in venv: $(which python) ($(python --version 2>&1))"

# ------------------------------------------------------------------
# Upgrade base tools
# ------------------------------------------------------------------
echo "==> Upgrading pip / setuptools / wheel..."
python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------
# Install package
# ------------------------------------------------------------------
PIP_EXTRA_ARGS=""
if [[ "$NO_CACHE" == "true" ]]; then
  PIP_EXTRA_ARGS="--no-cache-dir"
fi

if [[ "$LIGHT" == "true" ]]; then
  echo "==> Installing horos in editable mode (--light: no ML dependencies)..."
  # shellcheck disable=SC2086
  python -m pip install -e . --no-deps $PIP_EXTRA_ARGS
  # shellcheck disable=SC2086
  python -m pip install "pydantic>=2.6,<3" "flask>=3.0,<4" "pyyaml>=6.0" "pillow>=10.0" $PIP_EXTRA_ARGS
  if [[ "$DEV" == "true" ]]; then
    # shellcheck disable=SC2086
    python -m pip install "pytest>=8" "ruff>=0.4" $PIP_EXTRA_ARGS
  fi
elif [[ "$DEV" == "true" ]]; then
  echo "==> Installing horos in editable mode with [dev] extras..."
  # shellcheck disable=SC2086
  python -m pip install -e ".[dev]" $PIP_EXTRA_ARGS
else
  echo "==> Installing horos in editable mode..."
  # shellcheck disable=SC2086
  python -m pip install -e . $PIP_EXTRA_ARGS
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "==> Installed package info:"
python -m pip show horos || true

echo ""
echo "Setup completed successfully."
echo "  Activate with: source $VENV_DIR/bin/activate"
echo "  Test with:     bash scripts/local_test.sh"
