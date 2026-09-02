<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

<br />
<div align="center">
  <a href="https://github.com/SJ-Chuang/horos">
    <img src="horos/ui/static/favicon.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">horos</h3>

  <p align="center">
    End-to-end tooling for perception tasks: annotate → train → evaluate → deploy.
    <br />
    <a href="https://github.com/SJ-Chuang/horos/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/SJ-Chuang/horos/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#features">Features</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#jetson-read-this--it-matters">Jetson</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#development">Development</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

horos (ὅρος — *boundary, definition*) is not another model zoo. It is the
**path that takes a detection model into production**: one tool that carries a
dataset from raw images through annotation, training, and evaluation to a
deployable artifact.

Design premises the codebase is built on:

* **Models expire, workflows don't.** Every model dependency lives behind an
  adapter layer (`horos/backends/`); nothing above it knows which architecture
  is underneath. When the next architecture replaces RF-DETR, user code stays.
* **Licensing is a first-class citizen.** Every model, weight, and run carries
  queryable license metadata — the model picker shows it, the run records it,
  exports will ship it.
* **The deployment target is NVIDIA Jetson.** When "runs on a desktop" and
  "runs on a Jetson" conflict, Jetson wins.
* **Three interfaces, one capability set.** Everything the web UI does is also
  a Python API and a CLI, enforced by contract tests.

### Built With

* [Flask](https://flask.palletsprojects.com/) + vanilla JS (web API & UI)
* [pydantic](https://docs.pydantic.dev/) (every config and event is a schema)
* [RF-DETR](https://github.com/roboflow/rf-detr) (training backend, Apache 2.0)
* [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) (zero-shot auto-labeling, Apache 2.0)
* [pycocotools](https://github.com/ppwwyyxx/cocoapi) · [imageio](https://imageio.readthedocs.io/) (metrics & media decoding)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

| Stage | Status |
|---|---|
| **Dataset** | COCO / YOLO read+write; VOC, Darknet, VIA import; validator with actionable errors; stats; split management with re-split |
| **Annotate** | Web canvas (bbox + polygon), keyboard-first, resume where you left off, optimistic concurrency for multiple annotators |
| **Auto-label** | OWLv2 open-vocabulary prompts → pending pre-labels with review (accept / fix / reject), confidence filtering, uncertainty-first ordering |
| **Train** | RF-DETR Nano–Large; hyperparameters derived from dataset stats **with recorded reasons**, every value overridable; run queue with in-place editing; resume with full optimizer state; OOM auto-backoff; live loss/mAP curves; best-checkpoint criterion (mAP / smoothed mAP / val loss); post-run verdict with concrete suggestions |
| **Evaluate** | Photos, GIFs, and videos → per-frame prediction gallery with a full-screen viewer (confidence slider, frame-by-frame navigation); COCO metrics (pycocotools) with per-class AP and PR curves, persisted per run |
| **Deploy** | Not yet — ONNX / TensorRT / TFLite export is the next phase |

`import horos` stays fast and torch-free (backends load lazily on first use);
annotation-only installs never need a GPU.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Installation

The install scripts detect your platform (OS, NVIDIA GPU, Jetson) and install
the matching torch build plus horos into `./.venv`:

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
pip install pydantic flask pyyaml pillow imageio imageio-ffmpeg "transformers>=5.1,<6"
# torch/torchvision: use the NVIDIA wheel matching your JetPack version — FIRST,
# because the training stack below declares torch as a dependency
pip install "rfdetr==1.9.4" --no-deps
pip install supervision pycocotools scipy peft \
    "pytorch_lightning>=2.6,!=2.6.2,!=2.6.3,<3" \
    "torchmetrics[detection]>=1.2" "faster-coco-eval>=1.7.2"
```

`horos` warns at backend load time if it detects a Jetson platform where
`torch.cuda.is_available()` is False.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

```bash
horos init my-project                   # new project directory
horos import my-project path/to/data    # COCO / YOLO / VOC / Darknet / VIA, dir or zip
horos ui my-project                     # web UI: dataset, annotate, train, evaluate
```

The same pipeline, scripted:

```python
import horos.api as api

project = api.open_project("my-project")
record = api.start_training(project, api.TrainRunConfig(model="rfdetr-small"))
# ... poll api.training_status(project, record.run_id) ...
report = api.get_eval_report(project, record.run_id, "test")
```

And from the terminal: `horos train`, `horos infer`, `horos evaluate`,
`horos autolabel` — every UI action has a scriptable twin.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Project & dataset core (formats, validation, stats, splits)
- [x] Manual annotation (bbox + polygon, multi-annotator)
- [x] Auto-labeling (OWLv2 open-vocabulary, review workflow)
- [x] Training (derived hyperparameters, queue, resume, live monitoring)
- [x] Evaluation (media gallery, COCO metrics, per-class analysis)
- [ ] Error analysis (confusion pairs, worst-case mining)
- [ ] Experiment management (run comparison, dataset fingerprints)
- [ ] Export & deploy (ONNX / TensorRT / TFLite, model cards, parity checks)

See the [open issues](https://github.com/SJ-Chuang/horos/issues) for the full list.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . --no-deps
pip install pydantic flask pyyaml pillow imageio pytest ruff
pytest tests/test_invariants.py && pytest
```

`tests/test_invariants.py` runs first for a reason: it statically enforces the
architecture (no model imports outside `horos/backends/`, no torch after
`import horos`, no layer-skipping between UI, web API, and core).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

Model weights are downloaded at runtime and cached locally — horos never
bundles or redistributes them. Only Apache-2.0-licensed model sizes are
registered by default; non-Apache variants require an explicit acknowledgement.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

* [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow
* [OWLv2](https://arxiv.org/abs/2306.09683) by Google Research
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/SJ-Chuang/horos.svg?style=for-the-badge
[contributors-url]: https://github.com/SJ-Chuang/horos/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SJ-Chuang/horos.svg?style=for-the-badge
[forks-url]: https://github.com/SJ-Chuang/horos/network/members
[stars-shield]: https://img.shields.io/github/stars/SJ-Chuang/horos.svg?style=for-the-badge
[stars-url]: https://github.com/SJ-Chuang/horos/stargazers
[issues-shield]: https://img.shields.io/github/issues/SJ-Chuang/horos.svg?style=for-the-badge
[issues-url]: https://github.com/SJ-Chuang/horos/issues
[license-shield]: https://img.shields.io/github/license/SJ-Chuang/horos.svg?style=for-the-badge
[license-url]: https://github.com/SJ-Chuang/horos/blob/main/LICENSE
