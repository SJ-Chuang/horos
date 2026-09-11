"""ML-stack install planning: `horos install` (E4/§4 continued).

`pip install horos` ships only the lightweight core (datasets, annotation,
web UI). The ML stack — torch, rfdetr, transformers — deliberately stays out
of the base dependencies, because its correct source is platform-specific in
ways pip's static metadata cannot express:

  * Windows: the PyPI torch wheel is CPU-only; a CUDA machine must install
    from the matching download.pytorch.org index.
  * Linux without a GPU: the CPU index saves ~2 GB of CUDA libraries.
  * Jetson: torch must be NVIDIA's JetPack-matched wheel; a PyPI torch
    silently replaces it with a CPU build (§4).
  * AMD GPUs: there is no AMD build on PyPI at all. torch comes from AMD's
    own ROCm index, and the wheel is per-GPU-architecture (`--rocm gfx1201`).

`plan_install()` inspects the live environment (what is installed, whether an
NVIDIA driver is present and which CUDA version it supports, whether an AMD
GPU is present) and produces the ordered pip commands that close the gap. The
CLI (`horos install`) prints and executes them; `horos doctor` reuses the same
plan for its fix commands.

R1: this module never imports torch — it reads installed-package metadata,
runs nvidia-smi and reads the PCI device list, nothing more.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.platform_info import (
    PlatformInfo,
    detect_amd_gpu,
    detect_cuda_version,
    detect_platform,
)
from horos.errors import UnsupportedPlatformError

# R5: rfdetr is pinned exactly. The package has had silent annotation-corruption
# bugs under specific augmentation settings; a floating version makes training
# runs unreproducible. Upgrading is a standalone task with a full regression run.
# [train] pulls the Lightning training stack — rfdetr 1.9.4 cannot train
# without it, and training is a core horos feature.
# [onnx] adds the ONNX export path (onnx, onnxsim, onnx_graphsurgeon,
# onnxruntime, polygraphy — Apache-2.0 / MIT, verified in wheel metadata
# 2026-09). TensorRT and TFLite extras are NOT installed: tensorrt is an
# NVIDIA-licensed package the user installs on the target device, tensorflow
# is ~600 MB for an experimental path.
RFDETR_SPEC = "rfdetr[train,onnx]==1.9.4"
RFDETR_NO_DEPS_SPEC = "rfdetr==1.9.4"
#: rfdetr's [onnx] stack spelled out for the Jetson --no-deps path (no torch
#: in this dependency tree, so a plain install is safe there)
EXPORT_STACK_SPECS = [
    "onnx>=1.16.0,<2.0",
    "onnxsim>=0.7.0",
    "onnx_graphsurgeon",
    "onnxruntime",
    "polygraphy",
]
#: training-report export (E8): charts/PDF via matplotlib (PSF-style
#: license), Excel via openpyxl (MIT). Torch-free, installed with the ML
#: stack because reports describe training runs.
REPORT_SPECS = ["matplotlib>=3.7", "openpyxl>=3.1"]
#: TensorRT is opt-in (`horos install --tensorrt`): NVIDIA's wheels carry the
#: NVIDIA TensorRT license, so horos never adds them silently. The CUDA major
#: comes from the driver; the range tracks the polygraphy release rfdetr pins.
TENSORRT_SPEC_TEMPLATE = "tensorrt-cu{major}>=10.13,<11"
TENSORRT_LICENSE_NOTE = (
    "TensorRT wheels are distributed under the NVIDIA TensorRT license (not "
    "Apache 2.0); installing them is your choice — horos only uses them locally "
    "to build engines and never redistributes them."
)
# transformers hosts the OWLv2 backend; range matches rfdetr 1.9.4's own
# constraint (>=5.1.0,<6).
TRANSFORMERS_SPEC = "transformers>=5.1.0,<6"
# Backs rfdetr's aug_config path (the derived augmentation presets, E5) — without
# it rfdetr raises "Custom Albumentations augmentations require the optional
# augmentation extra" at the first training step. MIT (verified in wheel
# metadata 2026-09; the AGPL fork is the separate "albumentationsx" package).
# Pinned exactly for the same reason as rfdetr (R5): augmentation changes
# silently shift annotations and mAP. Installed on its own rather than via
# rfdetr[augment], which would also pull kornia — horos passes
# augmentation_backend="cpu" and never needs the GPU path. Must match the
# pin in pyproject.toml's [ml] extra.
ALBUMENTATIONS_SPEC = "albumentations==2.0.8"
# rfdetr's [train] stack spelled out, for the Jetson --no-deps path where pip
# must never get the chance to drag a PyPI torch in behind our back.
TRAIN_STACK_SPECS = [
    "supervision",
    "pycocotools",
    "pytorch_lightning>=2.6,!=2.6.2,!=2.6.3,<3",
    "torchmetrics[detection]>=1.2",
    "faster-coco-eval>=1.7.2",
    "scipy",
    "peft",
]

JETPACK_TORCH_ACTION = (
    "Install the NVIDIA JetPack-matched torch/torchvision wheel "
    "(never from PyPI): https://docs.nvidia.com/deeplearning/frameworks/"
    "install-pytorch-jetson-platform/"
)

#: import names of the ML stack the training/inference commands need; the
#: pre-flight gate for `horos train` etc. checks exactly these
ML_IMPORT_NAMES = (
    "torch",
    "torchvision",
    "rfdetr",
    "pytorch_lightning",
    "albumentations",
    "transformers",
)
#: import names of the export stack (E8): installed and doctored with the ML
#: stack, but their absence must not block training
EXPORT_IMPORT_NAMES = ("onnx", "onnxruntime", "matplotlib", "openpyxl")
ALL_IMPORT_NAMES = ML_IMPORT_NAMES + EXPORT_IMPORT_NAMES

_TORCH_INDEX_BASE = "https://download.pytorch.org/whl/"

#: the accelerator fields torch bakes into torch/version.py. A build is
#: CPU-only only when every one of them is None (see torch_is_cpu_build).
TORCH_ACCELERATOR_FIELDS = ("cuda", "hip", "xpu")

# ---------------------------------------------------------------- ROCm (AMD)
# AMD publishes its own Windows and Linux PyTorch wheels; upstream
# download.pytorch.org has no Windows ROCm build at all. The wheels carry the
# whole ROCm runtime (~1.4 GB), so no separate HIP SDK install is needed — only
# a current amdgpu driver.
#
# The architecture is explicit rather than detected: the [device-gfxNNNN] extra
# selects the compiled kernels, and the PCI-device-id-to-gfx table changes with
# every GPU generation. Guessing it wrong installs kernels the GPU cannot run,
# so `horos install --rocm <arch>` asks for it instead. AMD's compatibility
# matrix lists the arch per card.
ROCM_INDEX_URL = "https://stable.repo.amd.com/rocm/whl-next/"
ROCM_TORCH_VERSION = "2.13.0+rocm10.0.0"
ROCM_TORCHVISION_VERSION = "0.28.0+rocm10.0.0"
#: gfx target names only — this string is interpolated into a pip requirement
_ROCM_ARCH_RE = re.compile(r"^gfx[0-9a-f]{3,}$")
ROCM_DRIVER_NOTE = (
    "The ROCm wheels bundle the ROCm runtime but not the driver: a current "
    "AMD adrenalin / amdgpu driver is required. torch reports a ROCm GPU "
    "through torch.cuda (HIP maps onto the CUDA API), so horos selects it as "
    "device 'cuda' — TensorRT export stays NVIDIA-only."
)

# CUDA wheel indexes PyTorch publishes for Windows/Linux, newest first. The
# driver's supported CUDA version (nvidia-smi) must be >= the index's version;
# we pick the newest index the driver can run. Update when pytorch.org
# rotates its published indexes.
_CUDA_WHEEL_INDEXES: tuple[tuple[tuple[int, int], str], ...] = (
    ((13, 2), "cu132"),
    ((13, 0), "cu130"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((11, 8), "cu118"),
)


def cuda_index_url(cuda_version: tuple[int, int]) -> str | None:
    """The newest PyTorch CUDA wheel index this driver can run, or None."""
    for minimum, label in _CUDA_WHEEL_INDEXES:
        if cuda_version >= minimum:
            return _TORCH_INDEX_BASE + label
    return None


class InstallPlan(BaseModel):
    platform: PlatformInfo
    cuda_version: str | None  # driver-supported CUDA, e.g. "13.1"; None = no GPU
    #: pip install argument lists, in execution order (order matters: torch
    #: must land before rfdetr so pip sees its requirement satisfied)
    pip_commands: list[list[str]]
    #: steps that must never be automated (Jetson torch), spelled out
    manual_actions: list[str]
    notes: list[str] = Field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.pip_commands and not self.manual_actions


class MLReadiness(BaseModel):
    """Cheap pre-flight check for ML-dependent commands (no torch import)."""

    missing: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing


def _find_spec(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def probe_missing(names: Collection[str] = ALL_IMPORT_NAMES) -> list[str]:
    """Import names (ML + export stack by default) not importable here."""
    return [name for name in names if not _find_spec(name)]


def _version_py_field(text: str, field: str) -> str | None:
    """The literal assigned to `field` in torch/version.py, None if absent.

    Handles both the bare `cuda = None` of older builds and the annotated
    `cuda: Optional[str] = None` of current ones.
    """
    match = re.search(
        rf"^{field}\s*(?::[^=\n]+)?=\s*(.+?)\s*$", text, re.MULTILINE
    )
    return match.group(1) if match else None


def version_py_is_cpu_build(text: str) -> bool:
    """Decide CPU-vs-GPU from the text of torch/version.py.

    Split out from torch_is_cpu_build so the three wheel flavours (CUDA, ROCm,
    CPU) can be asserted without a torch install on the test machine.
    """
    values = [_version_py_field(text, f) for f in TORCH_ACCELERATOR_FIELDS]
    if all(value is None for value in values):
        return "+cpu" in text  # very old torch: fall back to the version tag
    # absent field == that accelerator not built in (older torch had no hip/xpu)
    return all(value in (None, "None") for value in values)


def torch_is_cpu_build() -> bool | None:
    """True if the installed torch has no GPU support, None if not installed.

    Reads torch/version.py off disk instead of importing torch — an import
    costs seconds, a file read is free. The dist metadata version is NOT
    enough: PyPI wheels are versioned plain "2.14.0" (PEP 440 bans local tags
    on PyPI), so the Windows CPU wheel is only identifiable by the baked-in
    `cuda = None` / `__version__ = '...+cpu'` in version.py.

    All three accelerator fields are checked, not just `cuda`: an AMD ROCm
    build reports `cuda = None` alongside `hip = '7.15.26333'`, so reading
    `cuda` alone reports a perfectly good GPU build as CPU-only and makes
    doctor tell the user to "fix" a working install.
    """
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        version_file = Path(location) / "version.py"
        try:
            text = version_file.read_text(encoding="utf-8")
        except OSError:
            continue
        values = [_version_py_field(text, f) for f in TORCH_ACCELERATOR_FIELDS]
        if all(value is None for value in values):
            return "+cpu" in text  # very old torch: fall back to the version tag
        # absent field == that accelerator not built in (older torch had no hip/xpu)
        return all(value in (None, "None") for value in values)
    return None


def _plan_torch(
    plan_commands: list[list[str]],
    notes: list[str],
    platform: PlatformInfo,
    cuda: tuple[int, int] | None,
    *,
    cpu: bool,
    reinstall: bool,
    rocm: str | None = None,
    amd_gpu: str | None = None,
) -> None:
    """Append the torch install command for a non-Jetson platform."""
    if rocm:
        # AMD's own index; the [device-gfxNNNN] extra picks the compiled kernels
        command = [
            f"torch[device-{rocm}]=={ROCM_TORCH_VERSION}",
            f"torchvision[device-{rocm}]=={ROCM_TORCHVISION_VERSION}",
            "--index-url",
            ROCM_INDEX_URL,
        ]
        notes.append(f"Installing the ROCm torch build for {rocm}.")
        notes.append(ROCM_DRIVER_NOTE)
        if reinstall:
            command.append("--force-reinstall")
        plan_commands.append(command)
        return
    command = ["torch", "torchvision"]
    if cpu or cuda is None:
        if platform.os_family == "linux":
            # the CPU index saves ~2 GB of CUDA libraries the machine can't use
            command += ["--index-url", _TORCH_INDEX_BASE + "cpu"]
        notes.append("Installing the CPU-only torch build.")
        if amd_gpu and not cpu:
            # §4: never let a GPU machine end up on CPU torch without saying so
            notes.append(
                f"{amd_gpu} detected, but torch has no AMD build on PyPI. "
                f"Run 'horos install --rocm <arch>' (e.g. --rocm gfx1201) to "
                "install AMD's ROCm wheels instead; see AMD's ROCm "
                "compatibility matrix for your card's gfx architecture."
            )
    elif platform.os_family == "windows":
        # the default PyPI Windows wheel is CPU-only — a CUDA machine must
        # install from the matching PyTorch index
        url = cuda_index_url(cuda)
        if url is None:
            notes.append(
                f"Driver supports CUDA {cuda[0]}.{cuda[1]}, older than any "
                "published torch CUDA index — falling back to the CPU build. "
                "Update the NVIDIA driver to enable GPU support."
            )
        else:
            command += ["--index-url", url]
    # linux + GPU and macOS: the default PyPI wheel is already right
    # (Linux wheels bundle CUDA; macOS wheels are the universal CPU/MPS build)
    if reinstall:
        command.append("--force-reinstall")
    plan_commands.append(command)


@capability(
    "system.install",
    summary="Plan the platform-matched ML-stack install (torch, rfdetr, transformers)",
    web_route=None,
    not_web_because="Diagnoses and mutates the local Python environment, not a project.",
    cli="install",
)
def plan_install(
    platform: PlatformInfo | None = None,
    *,
    cpu: bool = False,
    missing: Collection[str] | None = None,
    cuda_version: tuple[int, int] | None | Literal["auto"] = "auto",
    torch_cpu_build: bool | None | Literal["auto"] = "auto",
    tensorrt: bool = False,
    tensorrt_installed: bool | Literal["auto"] = "auto",
    rocm: str | None = None,
    amd_gpu: str | None | Literal["auto"] = "auto",
) -> InstallPlan:
    """Plan the pip commands that complete this environment's ML stack.

    Every parameter defaults to probing the live environment; tests (and
    doctor, which has already probed) inject explicit values instead.
    `tensorrt=True` additionally plans NVIDIA's TensorRT wheels for the
    driver's CUDA major (E8-T2) — opt-in because of their license.
    `rocm="gfx1201"` plans AMD's ROCm torch wheels for that architecture
    instead of the PyPI build — opt-in because the architecture cannot be
    detected reliably (see ROCM_INDEX_URL).
    """
    plat = platform or detect_platform()
    if missing is None:
        missing = probe_missing()
    missing = set(missing)
    if rocm is not None:
        rocm = rocm.strip().lower()
        if not _ROCM_ARCH_RE.match(rocm):
            raise ValueError(
                f"Invalid ROCm architecture {rocm!r}; expected a gfx target "
                "such as 'gfx1201' (see AMD's ROCm compatibility matrix)."
            )
        if plat.os_family == "macos":
            raise UnsupportedPlatformError(
                "ROCm has no macOS build; macOS uses the MPS backend of the "
                "standard PyPI torch build."
            )
        if plat.is_jetson:
            raise UnsupportedPlatformError(
                "Jetson is an NVIDIA platform: torch must come from NVIDIA's "
                "JetPack-matched wheel, never from AMD's ROCm index."
            )
        # ROCm replaces CUDA entirely; never plan both
        cuda_version, cpu = None, False
    if cuda_version == "auto":
        cuda_version = None if cpu else detect_cuda_version()
    if torch_cpu_build == "auto":
        torch_cpu_build = torch_is_cpu_build()
    if amd_gpu == "auto":
        # only worth probing when no NVIDIA GPU took the decision already
        amd_gpu = None if (cuda_version or plat.is_jetson) else detect_amd_gpu()

    commands: list[list[str]] = []
    manual: list[str] = []
    notes: list[str] = []
    torch_missing = bool({"torch", "torchvision"} & missing)

    if plat.is_jetson:
        # torch on Jetson is never automated: only the JetPack-matched NVIDIA
        # wheel has CUDA support there, and pip cannot install it (§4).
        if torch_missing:
            manual.append(JETPACK_TORCH_ACTION)
        if {"rfdetr", "pytorch_lightning"} & missing:
            if "rfdetr" in missing:
                # --no-deps so rfdetr cannot drag a PyPI torch in behind our back
                commands.append([RFDETR_NO_DEPS_SPEC, "--no-deps"])
            if torch_missing:
                manual.append(
                    "After installing the JetPack torch, re-run 'horos install' "
                    "to add the training stack (pytorch_lightning and friends "
                    "declare torch as a dependency and would pull the PyPI "
                    "build in if installed first)."
                )
            else:
                commands.append(list(TRAIN_STACK_SPECS))
        if {"onnx", "onnxruntime"} & missing:
            commands.append(list(EXPORT_STACK_SPECS))
    else:
        if torch_missing:
            _plan_torch(
                commands, notes, plat, cuda_version,
                cpu=cpu, reinstall=False, rocm=rocm, amd_gpu=amd_gpu,
            )
        elif rocm:
            # explicit request: replace whatever is installed with the ROCm build
            notes.append("Reinstalling torch from AMD's ROCm index as requested.")
            _plan_torch(
                commands, notes, plat, None, cpu=False, reinstall=True, rocm=rocm
            )
        elif torch_cpu_build and cuda_version is not None and not cpu:
            # torch is installed but it is the CPU build on a machine with a
            # working NVIDIA driver — the classic Windows `pip install` trap
            notes.append(
                "torch is installed but it is a CPU-only build while an NVIDIA "
                "GPU is present — reinstalling the matching CUDA build."
            )
            _plan_torch(commands, notes, plat, cuda_version, cpu=False, reinstall=True)
        elif torch_cpu_build and amd_gpu and not cpu:
            # same trap, AMD flavour: torch works, it just ignores the GPU
            notes.append(
                f"torch is installed but it is a CPU-only build while {amd_gpu} "
                "is present — run 'horos install --rocm <arch>' (e.g. "
                "--rocm gfx1201) to replace it with AMD's ROCm build."
            )
        if {"rfdetr", "pytorch_lightning", "onnx", "onnxruntime"} & missing:
            # one spec carries the training and the ONNX export stack
            commands.append([RFDETR_SPEC])

    if {"matplotlib", "openpyxl"} & missing:
        commands.append(list(REPORT_SPECS))

    if tensorrt:
        if tensorrt_installed == "auto":
            tensorrt_installed = _find_spec("tensorrt")
        if tensorrt_installed:
            notes.append("tensorrt is already installed — nothing to add for TensorRT.")
        elif plat.os_family == "macos":
            notes.append("TensorRT is not available on macOS; export engines on the target device.")
        elif plat.is_jetson:
            manual.append(
                "TensorRT on Jetson comes with JetPack (python3-libnvinfer). Use a venv "
                "created with --system-site-packages so the system tensorrt module is "
                "visible; never pip-install a tensorrt wheel on Jetson."
            )
        elif cuda_version is None or cpu:
            notes.append(
                "No NVIDIA GPU detected (or --cpu given): TensorRT engines can only be "
                "built on the GPU they run on, so nothing was planned."
            )
        else:
            commands.append([TENSORRT_SPEC_TEMPLATE.format(major=cuda_version[0])])
            notes.append(TENSORRT_LICENSE_NOTE)
    if "albumentations" in missing:
        # safe with deps on every platform, Jetson included: albumentations
        # depends on numpy/scipy/opencv-python-headless/albucore, never torch
        commands.append([ALBUMENTATIONS_SPEC])
    if "transformers" in missing:
        commands.append([TRANSFORMERS_SPEC])

    return InstallPlan(
        platform=plat,
        cuda_version=(f"{cuda_version[0]}.{cuda_version[1]}" if cuda_version else None),
        pip_commands=commands,
        manual_actions=manual,
        notes=notes,
    )


def check_ml_ready() -> MLReadiness:
    """Pre-flight for ML-dependent CLI commands. Fast: find_spec + metadata;
    nvidia-smi runs only when torch is already known to be a CPU build."""
    missing = probe_missing(ML_IMPORT_NAMES)  # the export stack never blocks training
    warnings: list[str] = []
    plat = detect_platform()
    if not plat.is_jetson and "torch" not in missing and torch_is_cpu_build():
        if detect_cuda_version() is not None:
            warnings.append(
                "An NVIDIA GPU is present but the installed torch is a CPU-only "
                "build — training and inference will not use the GPU. "
                "Run 'horos install' to replace it with the matching CUDA build."
            )
        elif (amd := detect_amd_gpu()) is not None:
            warnings.append(
                f"{amd} is present but the installed torch is a CPU-only build: "
                "training and inference will not use the GPU. Run "
                "'horos install --rocm <arch>' (e.g. --rocm gfx1201) to "
                "replace it with AMD's ROCm build."
            )
    return MLReadiness(missing=missing, warnings=warnings)
