#!/bin/bash
# horos installer — Ubuntu / macOS / Jetson.
# Detects the platform and installs the right torch + horos into ./.venv.
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
# torch — the install path depends on the platform (§4 of the project docs).
# It is installed BEFORE horos so pip sees the requirement as satisfied and
# never swaps in the wrong build.
# ---------------------------------------------------------------------------
have_nvidia() {
    command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1
}

if [[ "$IS_JETSON" == "yes" ]]; then
    # Jetson: pip's torch has NO CUDA support here. torch must come from the
    # NVIDIA wheel matching your JetPack version — this script never installs
    # torch on Jetson, it only verifies what is already there.
    if "$VPY" -c 'import torch' >/dev/null 2>&1; then
        if "$VPY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
            echo "Jetson torch OK: CUDA is available."
        else
            echo "=============================================================="
            echo "WARNING: torch is installed but torch.cuda.is_available() is"
            echo "False. On Jetson this usually means a CPU-only PyPI torch has"
            echo "replaced the JetPack build — inference will be ~10x slower."
            echo "Reinstall the NVIDIA JetPack-matched wheel:"
            echo "https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/"
            echo "=============================================================="
        fi
    else
        echo "=============================================================="
        echo "WARNING: torch is not importable. Install the NVIDIA JetPack-"
        echo "matched torch/torchvision wheel BEFORE using training/autolabel:"
        echo "https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/"
        echo "(dataset management and manual annotation work without it)"
        echo "=============================================================="
    fi
elif [[ "$OS" == "Darwin" ]]; then
    # macOS: the default PyPI wheel is the universal CPU/MPS build.
    echo "Installing torch (PyPI CPU/MPS build) ..."
    "$VPY" -m pip install torch torchvision
elif have_nvidia; then
    # Linux + NVIDIA GPU: the default PyPI Linux wheel bundles CUDA.
    echo "NVIDIA GPU detected. Installing torch (PyPI CUDA build) ..."
    "$VPY" -m pip install torch torchvision
else
    # Linux without a GPU: the CPU index saves ~2 GB of CUDA libraries.
    echo "No NVIDIA GPU detected. Installing the CPU-only torch build ..."
    "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ---------------------------------------------------------------------------
# horos itself
# ---------------------------------------------------------------------------
if [[ "$IS_JETSON" == "yes" ]]; then
    # --no-deps so pip cannot replace the JetPack torch (§4). The runtime
    # dependencies are installed explicitly, minus torch/torchvision.
    echo "Installing horos (--no-deps Jetson path) ..."
    "$VPY" -m pip install -e . --no-deps
    "$VPY" -m pip install "pydantic>=2.6,<3" "flask>=3.0,<4" "pyyaml>=6.0" "pillow>=10.0" \
        "transformers>=5.1.0,<6"
    "$VPY" -m pip install "rfdetr==1.9.4" --no-deps
    "$VPY" -m pip install supervision pycocotools
else
    echo "Installing horos with its dependencies (rfdetr, transformers) ..."
    "$VPY" -m pip install -e .
fi

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
echo "  horos init ./my_project"
echo "  horos import <dataset dir or unzipped export> --project ./my_project"
echo "  horos ui --project ./my_project    # open http://localhost:5000"
