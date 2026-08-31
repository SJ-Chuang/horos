"""Platform detection without touching any ML dependency.

Used by the capability matrix (E4-T13) and the environment check (E4-T6).
Jetson detection must work before torch is ever imported, because the whole
point of the Jetson warning is to catch a broken torch install.
"""

from __future__ import annotations

import platform
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
