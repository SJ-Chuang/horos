"""Platform capabilities (E4-T13/T14) and the model catalog (E4-S1/S2).

The capability list is the single source of truth for what works where (§4).
The Web API serves it verbatim and the WebUI drives button-disabled states
from it — platform checks are never hard-coded in the frontend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from horos.api.manifest import capability
from horos.core.platform_info import PlatformInfo, detect_platform
from horos.core.registry import ModelInfo
from horos.core.registry import list_models as _registry_list_models
from horos.errors import UnsupportedPlatformError

SupportLevel = Literal["full", "limited", "unavailable"]

FEATURES = (
    "dataset_management",
    "manual_annotation",
    "autolabel",
    "training",
    "export_onnx",
    "export_tflite",
    "export_tensorrt",
    "inference_service",
)


class FeatureSupport(BaseModel):
    feature: str
    level: SupportLevel
    note: str = ""

    @property
    def available(self) -> bool:
        return self.level != "unavailable"


class PlatformCapabilities(BaseModel):
    platform: PlatformInfo
    features: list[FeatureSupport]

    def get(self, feature: str) -> FeatureSupport:
        match = next((f for f in self.features if f.feature == feature), None)
        if match is None:
            raise KeyError(f"Unknown feature '{feature}' (known: {FEATURES})")
        return match


def _support_matrix(platform: PlatformInfo) -> list[FeatureSupport]:
    full = {f: FeatureSupport(feature=f, level="full") for f in FEATURES}
    if platform.os_family == "macos":
        full["autolabel"] = FeatureSupport(
            feature="autolabel",
            level="limited",
            note="Works, but MPS/CPU inference is slower than CUDA.",
        )
        full["training"] = FeatureSupport(
            feature="training",
            level="limited",
            note="Suitable for small-dataset pipeline validation only.",
        )
        full["export_tensorrt"] = FeatureSupport(
            feature="export_tensorrt",
            level="unavailable",
            note=(
                "TensorRT is not supported on macOS. TensorRT engines are not "
                "portable — export on the target (Jetson/CUDA) device instead."
            ),
        )
    elif platform.is_jetson:
        full["training"] = FeatureSupport(
            feature="training",
            level="limited",
            note="Not recommended on Jetson (compute-constrained), but not blocked.",
        )
    return [full[f] for f in FEATURES]


@capability(
    "system.capabilities",
    summary="Structured list of which horos features work on this platform",
    web_route="/api/v1/capabilities",
    web_methods=("GET",),
    cli="capabilities",
)
def platform_capabilities() -> PlatformCapabilities:
    """What can this machine do? (E4-S10) Drives UI disabled states."""
    platform = detect_platform()
    return PlatformCapabilities(platform=platform, features=_support_matrix(platform))


def ensure_supported(feature: str) -> FeatureSupport:
    """Guard for feature entry points (E4-T14): raise a clear error for
    unsupported platform/feature combinations — never fall back silently."""
    support = platform_capabilities().get(feature)
    if not support.available:
        raise UnsupportedPlatformError(
            f"'{feature}' is not supported on this platform. {support.note}"
        )
    return support


@capability(
    "models.list",
    summary="List available models with size, latency hint, and license",
    web_route="/api/v1/models",
    web_methods=("GET",),
    cli="models",
)
def list_models(task: str | None = None) -> list[ModelInfo]:
    """Registry passthrough (static metadata only — nothing is imported)."""
    return _registry_list_models(task)  # type: ignore[arg-type]
