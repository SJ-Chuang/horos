#!/bin/bash
# horos installer — Ubuntu / macOS / Jetson.
# Creates ./.venv, installs the horos core, then runs `horos install`, which
# detects the platform (GPU, CUDA version, Jetson) and installs the matching
# ML stack. All platform logic lives in horos itself (horos/api/install.py) —
# this script only bootstraps the venv.
# Windows: use install.bat instead.
set -e

# ---------------------------------------------------------------------------
# Platform / Python detection
# ---------------------------------------------------------------------------
OS="$(uname -s)"        # Darwin | Linux
ARCH="$(uname -m)"      # x86_64 | arm64 (macOS) | aarch64 (Jetson)
PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: '$PY' not found. Install Python >= 3.10 or set PYTHON=<path>." >&2
    exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "ERROR: horos needs Python >= 3.10 (found $("$PY" -V 2>&1))." >&2
    exit 1
fi

IS_JETSON="no"
if [[ -f /etc/nv_tegra_release ]]; then
    IS_JETSON="yes"
fi

echo "Detected: OS=$OS ARCH=$ARCH Python=$("$PY" -V 2>&1 | cut -d' ' -f2) Jetson=$IS_JETSON"

# ---------------------------------------------------------------------------
# Virtual environment. Jetson keeps --system-site-packages so the JetPack
# torch stays visible; everywhere else the env is isolated.
# ---------------------------------------------------------------------------
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Using the already-activated virtualenv: $VIRTUAL_ENV"
    echo "(run 'deactivate' first if you wanted a fresh ./.venv instead)"
    VPY="$VIRTUAL_ENV/bin/python"
else
    if [[ ! -d .venv ]]; then
        echo "Creating .venv ..."
        if [[ "$IS_JETSON" == "yes" ]]; then
            "$PY" -m venv --system-site-packages .venv
        else
            "$PY" -m venv .venv
        fi
    fi
    VPY=".venv/bin/python"
fi

"$VPY" -m pip install --upgrade pip wheel >/dev/null

# ---------------------------------------------------------------------------
# horos core (torch-free by design), then the ML stack via `horos install`,
# which picks the platform-correct torch source (§4):
#   Linux + NVIDIA GPU  -> PyPI (Linux wheels bundle CUDA)
#   Linux without GPU   -> PyTorch CPU index (saves ~2 GB)
#   macOS               -> PyPI universal build (MPS)
#   Jetson              -> torch is NEVER pip-installed; JetPack wheel steps
#                          are printed, the rest installs with --no-deps
# ---------------------------------------------------------------------------
echo "Installing the horos core ..."
"$VPY" -m pip install -e .

echo "Installing the ML stack (horos install) ..."
"$VPY" -m horos.cli install

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
"$VPY" - <<'EOF'
import sys, time
t0 = time.time()
import horos  # noqa: F401
dt = time.time() - t0
assert "torch" not in sys.modules, "R1b violated: import horos pulled in torch"
print(f"import horos OK ({dt:.2f}s, lazy backends intact)")
EOF
"$VPY" -m horos.cli --version >/dev/null 2>&1 || "$VPY" -c "import horos.cli" >/dev/null
echo "horos $("$VPY" -c 'import horos; print(horos.__version__)') installed."
echo
echo "Next steps:"
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "  source .venv/bin/activate"
fi
echo "  horos doctor                   # verify the environment"
echo "  horos init ./my_project"
echo "  horos import <dataset dir or unzipped export> --project ./my_project"
echo "  horos ui ./my_project          # open http://localhost:5000"
