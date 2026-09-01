"""Cross-platform memory probe (E5-T6c, R7).

Feeds the batch-size derivation (E5-T1): CUDA reports free/total VRAM, MPS
reports the unified-memory budget, CPU reports system RAM with no availability
claim — the deriver treats unknown availability conservatively.

torch is imported lazily inside the probes (R1b); the CPU/system paths are
stdlib-only so the probe works in an annotation-only install too.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_GB = 1024**3


class MemoryInfo(BaseModel):
    kind: Literal["cuda", "mps", "cpu"]
    total_gb: float | None
    #: memory a training run may reasonably claim; None = unknown (be conservative)
    available_gb: float | None
    source: str  # where the numbers came from, for the derivation reason


def _system_ram_bytes() -> int | None:
    if os.name == "nt":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def probe_memory(kind: str | None = None) -> MemoryInfo:
    """Measure the memory the selected (or given) device can offer training."""
    from horos.backends.device import select_device

    device_kind = kind or select_device().kind

    if device_kind == "cuda":
        import torch

        free, total = torch.cuda.mem_get_info()
        return MemoryInfo(
            kind="cuda",
            total_gb=total / _GB,
            available_gb=free / _GB,
            source="torch.cuda.mem_get_info",
        )

    if device_kind == "mps":
        import torch

        # Unified memory: Metal caps GPU working sets below physical RAM.
        # recommended_max_memory is that cap; what torch already holds is gone.
        recommended = torch.mps.recommended_max_memory()
        allocated = torch.mps.driver_allocated_memory()
        return MemoryInfo(
            kind="mps",
            total_gb=recommended / _GB,
            available_gb=max(recommended - allocated, 0) / _GB,
            source="torch.mps.recommended_max_memory",
        )

    ram = _system_ram_bytes()
    return MemoryInfo(
        kind="cpu",
        total_gb=ram / _GB if ram else None,
        # System RAM is shared with everything else; claiming a fixed fraction
        # would be a guess dressed as a measurement. Unknown → derive conservatively.
        available_gb=None,
        source="system RAM (availability unknown — conservative defaults apply)",
    )
