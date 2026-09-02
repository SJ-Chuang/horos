# horos

End-to-end tooling for perception tasks: annotate → train → evaluate → deploy.

- Models are adapters behind `horos/backends/`; nothing above that layer knows which
  architecture is underneath.
- License metadata travels with every model, run, and export artifact.
- Target deployment platform is NVIDIA Jetson.

## Quickstart

```bash
horos init my-project                   # new project directory
horos import my-project path/to/data    # COCO / YOLO / VOC / Darknet / VIA, dir or zip
horos ui my-project                     # web UI: dataset, annotate, train, evaluate
```

Everything the UI does is also a Python API (`import horos`) and a CLI —
`horos train`, `horos infer`, `horos evaluate` run the same code paths.

## What works today

| Stage | Status |
|---|---|
| **Dataset** | COCO / YOLO read+write; VOC, Darknet, VIA import; validator with actionable errors; stats; split management with re-split |
| **Annotate** | Web canvas (bbox + polygon), keyboard-first, resume where you left off, optimistic concurrency for multiple annotators |
| **Auto-label** | OWLv2 open-vocabulary prompts → pending pre-labels with review (accept / fix / reject), confidence filtering, uncertainty-first ordering |
| **Train** | RF-DETR Nano–Large; hyperparameters derived from dataset stats **with recorded reasons**, every value overridable; run queue (edit/cancel queued runs); resume with full optimizer state; OOM auto-backoff; live loss/mAP curves; best-checkpoint criterion (mAP / smoothed mAP / val loss); post-run verdict with concrete suggestions |
| **Evaluate** | Upload photos for instant overlay testing; COCO metrics (pycocotools) with per-class AP and PR curves, persisted per run |
| **Deploy** | Not yet — ONNX / TensorRT / TFLite export is the next phase |

`import horos` stays fast and torch-free (backends load lazily on first use);
annotation-only installs never need a GPU.

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
pip install pydantic flask pyyaml pillow "transformers>=5.1,<6"
# torch/torchvision: use the NVIDIA wheel matching your JetPack version — FIRST,
# because the training stack below declares torch as a dependency
pip install "rfdetr==1.9.4" --no-deps
pip install supervision pycocotools scipy peft \
    "pytorch_lightning>=2.6,!=2.6.2,!=2.6.3,<3" \
    "torchmetrics[detection]>=1.2" "faster-coco-eval>=1.7.2"
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
