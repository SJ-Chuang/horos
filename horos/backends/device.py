"""Device selection abstraction (E4-T12, R7).

Priority: CUDA → MPS → CPU. Nothing outside `horos/backends/` may write
`.cuda()` or `device="cuda"`; they ask this module instead. torch is imported
lazily inside the probe functions so this module obeys R1b.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from horos.errors import UnsupportedPlatformError

logger = logging.getLogger(__name__)

DeviceKind = Literal["cuda", "mps", "cpu"]


class DeviceInfo(BaseModel):
    kind: DeviceKind
    name: str
    torch_device: str  # the string to hand to torch, e.g. "cuda:0"


def _torch():
    try:
        import torch  # noqa: PLC0415 — lazy by design (R1b)
    except ImportError:
        return None
    return torch


def cuda_available() -> bool:
    torch = _torch()
    return bool(torch and torch.cuda.is_available())


def mps_available() -> bool:
    torch = _torch()
    return bool(
        torch
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )


def select_device(prefer: DeviceKind | None = None) -> DeviceInfo:
    """Pick the best available device, or validate a forced choice.

    Forcing an unavailable device raises UnsupportedPlatformError — no silent
    CPU fallback (§4).
    """
    if prefer is not None:
        if prefer == "cuda" and not cuda_available():
            raise UnsupportedPlatformError(
                "Device 'cuda' was requested but CUDA is not available. "
                "horos does not fall back silently; pass prefer=None to auto-select."
            )
        if prefer == "mps" and not mps_available():
            raise UnsupportedPlatformError(
                "Device 'mps' was requested but MPS is not available. "
                "horos does not fall back silently; pass prefer=None to auto-select."
            )
        kind: DeviceKind = prefer
    elif cuda_available():
        kind = "cuda"
    elif mps_available():
        kind = "mps"
    else:
        kind = "cpu"

    if kind == "cuda":
        torch = _torch()
        name = torch.cuda.get_device_name(0) if torch else "CUDA device"
        info = DeviceInfo(kind="cuda", name=name, torch_device="cuda:0")
    elif kind == "mps":
        info = DeviceInfo(kind="mps", name="Apple Silicon (MPS)", torch_device="mps")
    else:
        info = DeviceInfo(kind="cpu", name="CPU", torch_device="cpu")
    logger.debug("selected device: %s", info)
    return info
