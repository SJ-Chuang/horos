"""Platform detection without touching any ML dependency.

Used by the capability matrix (E4-T13) and the environment check (E4-T6).
Jetson detection must work before torch is ever imported, because the whole
point of the Jetson warning is to catch a broken torch install.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

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


# PCI-SIG vendor id for AMD/ATI. Detecting the *vendor* is reliable and cheap;
# deriving the gfx architecture from the device id is not (the table changes
# every GPU generation), so `horos install --rocm <arch>` takes it explicitly.
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
    architecture is deliberately not derived from the device id — see
    `_AMD_PCI_VENDOR`. macOS returns None: ROCm has no macOS build.
    """
    system = platform.system()
    if system == "Windows":
        return _detect_amd_gpu_windows()
    if system == "Darwin":
        return None
    return _detect_amd_gpu_linux()


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
