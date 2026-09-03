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

`plan_install()` inspects the live environment (what is installed, whether an
NVIDIA driver is present and which CUDA version it supports) and produces the
ordered pip commands that close the gap. The CLI (`horos install`) prints and
executes them; `horos doctor` reuses the same plan for its fix commands.

R1: this module never imports torch — it reads installed-package metadata and
runs nvidia-smi, nothing more.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.platform_info import PlatformInfo, detect_cuda_version, detect_platform

# R5: rfdetr is pinned exactly. The package has had silent annotation-corruption
# bugs under specific augmentation settings; a floating version makes training
# runs unreproducible. Upgrading is a standalone task with a full regression run.
# [train] pulls the Lightning training stack — rfdetr 1.9.4 cannot train
# without it, and training is a core horos feature.
RFDETR_SPEC = "rfdetr[train]==1.9.4"
RFDETR_NO_DEPS_SPEC = "rfdetr==1.9.4"
# transformers hosts the OWLv2 backend; range matches rfdetr 1.9.4's own
# constraint (>=5.1.0,<6).
TRANSFORMERS_SPEC = "transformers>=5.1.0,<6"
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

#: import names of the ML stack, as probed by the readiness check and doctor
ML_IMPORT_NAMES = ("torch", "torchvision", "rfdetr", "pytorch_lightning", "transformers")

_TORCH_INDEX_BASE = "https://download.pytorch.org/whl/"

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


def probe_missing() -> list[str]:
    """ML-stack import names not importable in this environment."""
    return [name for name in ML_IMPORT_NAMES if not _find_spec(name)]


def torch_is_cpu_build() -> bool | None:
    """True if the installed torch has no CUDA support, None if not installed.

    Reads torch/version.py off disk instead of importing torch — an import
    costs seconds, a file read is free. The dist metadata version is NOT
    enough: PyPI wheels are versioned plain "2.14.0" (PEP 440 bans local tags
    on PyPI), so the Windows CPU wheel is only identifiable by the baked-in
    `cuda = None` / `__version__ = '...+cpu'` in version.py.
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
        match = re.search(r"^cuda\s*(?::[^=\n]+)?=\s*(.+?)\s*$", text, re.MULTILINE)
        if match:
            return match.group(1) == "None"
        return "+cpu" in text  # very old torch: fall back to the version tag
    return None


def _plan_torch(
    plan_commands: list[list[str]],
    notes: list[str],
    platform: PlatformInfo,
    cuda: tuple[int, int] | None,
    *,
    cpu: bool,
    reinstall: bool,
) -> None:
    """Append the torch install command for a non-Jetson platform."""
    command = ["torch", "torchvision"]
    if cpu or cuda is None:
        if platform.os_family == "linux":
            # the CPU index saves ~2 GB of CUDA libraries the machine can't use
            command += ["--index-url", _TORCH_INDEX_BASE + "cpu"]
        notes.append("Installing the CPU-only torch build.")
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
) -> InstallPlan:
    """Plan the pip commands that complete this environment's ML stack.

    Every parameter defaults to probing the live environment; tests (and
    doctor, which has already probed) inject explicit values instead.
    """
    plat = platform or detect_platform()
    if missing is None:
        missing = probe_missing()
    missing = set(missing)
    if cuda_version == "auto":
        cuda_version = None if cpu else detect_cuda_version()
    if torch_cpu_build == "auto":
        torch_cpu_build = torch_is_cpu_build()

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
    else:
        if torch_missing:
            _plan_torch(commands, notes, plat, cuda_version, cpu=cpu, reinstall=False)
        elif torch_cpu_build and cuda_version is not None and not cpu:
            # torch is installed but it is the CPU build on a machine with a
            # working NVIDIA driver — the classic Windows `pip install` trap
            notes.append(
                "torch is installed but it is a CPU-only build while an NVIDIA "
                "GPU is present — reinstalling the matching CUDA build."
            )
            _plan_torch(commands, notes, plat, cuda_version, cpu=False, reinstall=True)
        if {"rfdetr", "pytorch_lightning"} & missing:
            commands.append([RFDETR_SPEC])

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
    missing = probe_missing()
    warnings: list[str] = []
    plat = detect_platform()
    if (
        not plat.is_jetson
        and "torch" not in missing
        and torch_is_cpu_build()
        and detect_cuda_version() is not None
    ):
        warnings.append(
            "An NVIDIA GPU is present but the installed torch is a CPU-only "
            "build — training and inference will not use the GPU. "
            "Run 'horos install' to replace it with the matching CUDA build."
        )
    return MLReadiness(missing=missing, warnings=warnings)
