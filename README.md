# horos

End-to-end tooling for perception tasks: annotate → train → evaluate → deploy.

- Models are adapters behind `horos/backends/`; nothing above that layer knows which
  architecture is underneath.
- License metadata travels with every model, run, and export artifact.
- Target deployment platform is NVIDIA Jetson.

## Install

The install scripts detect your platform (OS, NVIDIA GPU, Jetson) and install the
matching torch build plus horos into `./.venv`:

```bash
# Ubuntu / macOS / Jetson
./install.sh

# Windows
install.bat
```

What they decide for you:

| Platform | torch source |
|---|---|
| Linux + NVIDIA GPU | PyPI (Linux wheels bundle CUDA) |
| Linux without GPU | PyTorch CPU index (saves ~2 GB) |
| macOS | PyPI universal build (MPS) |
| Windows + NVIDIA GPU | PyTorch index matching your CUDA (cu118/cu124/cu126) — the PyPI Windows wheel is CPU-only |
| Windows without GPU | PyPI (CPU) |
| Jetson | **never installed by the script** — see below |

Or simply:

```bash
pip install horos
horos doctor        # verifies the environment; `horos doctor --fix` installs what's missing
```

`pip install horos` does the right thing per platform: on Linux/aarch64 (Jetson)
it deliberately skips `rfdetr`/`torch` so pip can never replace the CUDA JetPack
torch — `horos doctor --fix` completes the install there with the right sources.

**Use a dedicated environment.** horos pins `rfdetr` exactly (upstream has had
silent annotation-corruption bugs; reproducibility wins) and requires
`transformers >= 5.1` — installing into a shared ML environment will upgrade
`transformers`, `supervision`, `huggingface-hub` and friends, which can break
other projects living in that environment. The install scripts create an
isolated `./.venv` for exactly this reason.

### Jetson (read this — it matters)

On Jetson, torch **must** come from NVIDIA's JetPack-matched wheel. The PyPI torch has
no CUDA support on Jetson, and a plain `pip install horos` may silently replace your
CUDA-enabled torch with a CPU-only build — everything still runs, just an order of
magnitude slower.

`./install.sh` handles this automatically on Jetson: it creates the venv with
`--system-site-packages`, verifies the existing torch has CUDA (warning loudly if
not), and installs horos with `--no-deps` so pip can never swap torch out.

Doing it by hand:

```bash
pip install horos --no-deps
pip install pydantic flask pyyaml pillow "transformers>=5.1,<6" supervision
# torch/torchvision: use the NVIDIA wheel matching your JetPack version
```

`horos` warns at backend load time if it detects a Jetson platform where
`torch.cuda.is_available()` is False.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . --no-deps
pip install pydantic flask pyyaml pillow pytest ruff
pytest tests/test_invariants.py && pytest
```
