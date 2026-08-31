"""torch / CUDA environment check (E4-T6).

The one check that matters most: on Jetson, a pip-installed PyPI torch has no
CUDA support, and the failure is silent — everything runs, just 10x slower.
`check_environment()` makes that failure loud.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from horos.backends import device as device_mod
from horos.core.platform_info import PlatformInfo, detect_platform

logger = logging.getLogger(__name__)

JETSON_NO_CUDA_WARNING = (
    "Jetson platform detected but torch.cuda.is_available() is False. "
    "You are almost certainly running a CPU-only torch from PyPI, which on Jetson "
    "means inference an order of magnitude slower than it should be. "
    "Install the NVIDIA torch wheel matching your JetPack version "
    "(pip install horos --no-deps, then the NVIDIA wheel). "
    "See the Jetson section of the horos install docs."
)


class EnvReport(BaseModel):
    platform: PlatformInfo
    torch_version: str | None
    cuda_available: bool
    mps_available: bool
    warnings: list[str]


def _torch_version() -> str | None:
    try:
        import torch  # noqa: PLC0415 — lazy by design (R1b)
    except ImportError:
        return None
    return str(torch.__version__)


def check_environment(*, emit_warnings: bool = True) -> EnvReport:
    """Inspect the ML runtime environment. Called by backends on first use.

    Importing this module is cheap; calling this function loads torch (if
    installed) — callers must already be past the lazy-loading boundary.
    """
    plat = detect_platform()
    torch_version = _torch_version()
    cuda = device_mod.cuda_available()
    mps = device_mod.mps_available()

    warnings: list[str] = []
    if plat.is_jetson and not cuda:
        warnings.append(JETSON_NO_CUDA_WARNING)
    if torch_version is None:
        warnings.append(
            "torch is not installed — training, inference and autolabeling are "
            "unavailable. Dataset management and annotation still work."
        )

    if emit_warnings:
        for message in warnings:
            logger.warning(message)

    return EnvReport(
        platform=plat,
        torch_version=torch_version,
        cuda_available=cuda,
        mps_available=mps,
        warnings=warnings,
    )
