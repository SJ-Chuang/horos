"""Platform detection without touching any ML dependency.

Used by the capability matrix (E4-T13) and the environment check (E4-T6).
Jetson detection must work before torch is ever imported, because the whole
point of the Jetson warning is to catch a broken torch install.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

OsFamily = Literal["linux", "macos", "windows"]

_JETSON_RELEASE_FILE = Path("/etc/nv_tegra_release")
_DEVICE_TREE_MODEL = Path("/proc/device-tree/model")


class PlatformInfo(BaseModel):
    os_family: OsFamily
    arch: str
    is_jetson: bool
    python_version: str


def _detect_jetson() -> bool:
    if _JETSON_RELEASE_FILE.exists():
        return True
    try:
        model = _DEVICE_TREE_MODEL.read_text(errors="ignore").lower()
    except OSError:
        return False
    return "jetson" in model or "nvidia" in model


# nvidia-smi reports the highest CUDA version the installed driver supports —
# the real constraint on which torch CUDA wheel can run. nvcc is only a
# fallback signal: it names the local toolkit, not the driver's ceiling.
_SMI_CUDA_RE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
_NVCC_CUDA_RE = re.compile(r"release\s+(\d+)\.(\d+)")


def detect_cuda_version(timeout: float = 10.0) -> tuple[int, int] | None:
    """(major, minor) CUDA version the NVIDIA driver supports, else None.

    None means "no usable NVIDIA GPU detected" (no driver, or the tools are
    not on PATH). Never imports torch.
    """
    for command, pattern in (
        (["nvidia-smi"], _SMI_CUDA_RE),
        (["nvcc", "--version"], _NVCC_CUDA_RE),
    ):
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = pattern.search(proc.stdout) if proc.returncode == 0 else None
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


# PCI-SIG vendor id for AMD/ATI. This answers "is there an AMD GPU", which is
# all the install/doctor notices need. The gfx architecture is a separate
# question with a proper answer (detect_rocm_arch below) — it is never derived
# from the device id here, because that table changes every GPU generation.
_AMD_PCI_VENDOR = "1002"
# the Windows "Display adapters" setup class
_WINDOWS_DISPLAY_CLASS = (
    r"SYSTEM\CurrentControlSet\Control\Class"
    r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
)
_PCI_DEVICES = Path("/sys/bus/pci/devices")


def _detect_amd_gpu_windows() -> str | None:
    import winreg  # noqa: PLC0415 — Windows-only stdlib

    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_DISPLAY_CLASS)
    except OSError:
        return None
    with root:
        for index in range(64):  # one subkey per installed display adapter
            try:
                name = winreg.EnumKey(root, index)
            except OSError:
                break
            if not name.isdigit():
                continue
            try:
                with winreg.OpenKey(root, name) as key:
                    matching = winreg.QueryValueEx(key, "MatchingDeviceId")[0]
                    if f"ven_{_AMD_PCI_VENDOR}" not in str(matching).lower():
                        continue
                    return str(winreg.QueryValueEx(key, "DriverDesc")[0])
            except OSError:
                continue
    return None


def _detect_amd_gpu_linux(pci_root: Path | None = None) -> str | None:
    root = pci_root or _PCI_DEVICES
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return None
    for entry in entries:
        try:
            vendor = (entry / "vendor").read_text().strip().lower()
            pci_class = (entry / "class").read_text().strip().lower()
        except OSError:
            continue
        # class 0x03xxxx = display controller; skip audio/USB functions
        if vendor == f"0x{_AMD_PCI_VENDOR}" and pci_class.startswith("0x03"):
            try:
                device = (entry / "device").read_text().strip()
            except OSError:
                device = "unknown"
            return f"AMD GPU (PCI {_AMD_PCI_VENDOR}:{device.removeprefix('0x')})"
    return None


def detect_amd_gpu() -> str | None:
    """Name of an installed AMD GPU, else None. Never imports torch.

    Used to tell a user on an AMD machine that the CPU torch they are about to
    get is not their only option (§4 forbids a silent CPU fallback). The gfx
    architecture that comes with it is answered by detect_rocm_arch, not here.
    macOS returns None: ROCm has no macOS build.
    """
    system = platform.system()
    if system == "Windows":
        return _detect_amd_gpu_windows()
    if system == "Darwin":
        return None
    return _detect_amd_gpu_linux()


# gfx target names as every AMD tool spells them: gfx1201, gfx90a, gfx942.
# Unambiguous enough to grep out of a tool's whole stdout, which is far more
# robust than depending on any one tool's line layout.
_GFX_RE = re.compile(r"\bgfx[0-9a-f]{3,}\b", re.IGNORECASE)
#: escape hatch for machines where no probe works (containers, CI, a GPU
#: newer than the installed driver's tooling)
ROCM_ARCH_ENV = "HOROS_ROCM_ARCH"


def _arch_from_rocm_bootstrap() -> list[str]:
    """AMD's own detector (the rocm-bootstrap package, MIT).

    Authoritative where it works: it reads the KFD topology and ip_discovery
    sysfs nodes. Linux only in practice — its Windows helper exists but is not
    wired into its detection chain — and it is only installed once ROCm is,
    so this is the confirmation path rather than the bootstrap one.
    """
    try:
        from rocm_bootstrap.detect import detect_gfx_targets  # noqa: PLC0415
    except ImportError:
        return []
    try:
        return [str(target) for target in detect_gfx_targets()]
    except Exception:  # noqa: BLE001 — a probe must never break `horos doctor`
        logger.debug("rocm_bootstrap detection failed", exc_info=True)
        return []


def _arch_from_command(command: list[str], timeout: float) -> list[str]:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [match.group(0).lower() for match in _GFX_RE.finditer(proc.stdout)]


def detect_rocm_arch(timeout: float = 20.0) -> str | None:
    """The gfx architecture of the installed AMD GPU, e.g. "gfx1201".

    This is what AMD's ROCm wheels are selected by, and it is needed *before*
    ROCm exists, so the chain is ordered by what is present on a bare machine:

      1. HOROS_ROCM_ARCH, for machines where no probe can work
      2. rocm-bootstrap, AMD's own detector, when already installed
      3. clinfo — the one that carries a fresh Windows box: the AMD display
         driver installs it into System32 and it prints the gfx name directly
      4. rocminfo / hipInfo, present once ROCm itself is

    None means "could not tell" — never a guess. Callers must then ask for the
    architecture explicitly rather than install kernels the GPU cannot run.
    """
    forced = os.environ.get(ROCM_ARCH_ENV, "").strip().lower()
    if forced:
        return forced
    found = _arch_from_rocm_bootstrap()
    for command in (["clinfo"], ["rocminfo"], ["hipInfo"]):
        if found:
            break
        found = _arch_from_command(command, timeout)
    if not found:
        return None
    distinct = sorted(set(found))
    if len(distinct) > 1:
        # mixed-GPU box: one wheel set cannot serve both, so say which one won
        logger.warning(
            "Several AMD GPU architectures detected (%s); using %s. "
            "Set %s to choose a different one.",
            ", ".join(distinct),
            distinct[0],
            ROCM_ARCH_ENV,
        )
    return distinct[0]


def detect_platform() -> PlatformInfo:
    system = platform.system()
    if system == "Darwin":
        os_family: OsFamily = "macos"
    elif system == "Windows":
        os_family = "windows"
    else:
        os_family = "linux"
    return PlatformInfo(
        os_family=os_family,
        arch=platform.machine(),
        is_jetson=os_family == "linux" and _detect_jetson(),
        python_version="{}.{}.{}".format(*sys.version_info[:3]),
    )
