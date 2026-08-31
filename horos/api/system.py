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


# ------------------------------------------------------------------ doctor


class DependencyStatus(BaseModel):
    name: str
    required: str
    installed: str | None
    ok: bool
    note: str = ""


class DoctorReport(BaseModel):
    platform: PlatformInfo
    dependencies: list[DependencyStatus]
    torch_cuda_available: bool | None = None  # None when torch is missing
    torch_mps_available: bool | None = None
    #: pip install argument lists `doctor --fix` would run, in order
    fix_commands: list[list[str]]
    #: steps that must never be automated (Jetson torch), spelled out
    manual_actions: list[str]

    @property
    def ok(self) -> bool:
        return all(d.ok for d in self.dependencies) and not self.manual_actions


_RUNTIME_DEPS: list[tuple[str, str]] = [
    ("pydantic", "pydantic>=2.6,<3"),
    ("flask", "flask>=3.0,<4"),
    ("yaml", "pyyaml>=6.0"),
    ("PIL", "pillow>=10.0"),
    ("transformers", "transformers>=5.1,<6"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("rfdetr", "rfdetr[train]==1.9.4"),
    # the [train] extra's marker package — missing means rfdetr was installed
    # without its training stack and horos cannot train
    ("pytorch_lightning", "rfdetr[train]==1.9.4"),
]

_IMPORT_TO_DIST = {
    "yaml": "PyYAML",
    "PIL": "pillow",
    "pytorch_lightning": "pytorch-lightning",
}


def _installed_version(import_name: str) -> str | None:
    import importlib.metadata
    import importlib.util

    if importlib.util.find_spec(import_name) is None:
        return None
    dist = _IMPORT_TO_DIST.get(import_name, import_name)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _plan_fixes(
    missing: list[str], platform: PlatformInfo
) -> tuple[list[list[str]], list[str]]:
    """Map missing deps to pip commands per platform. torch on Jetson is never
    automated — a PyPI torch would silently replace the CUDA JetPack build (§4)."""
    commands: list[list[str]] = []
    manual: list[str] = []
    needs_torch = "torch" in missing or "torchvision" in missing
    if needs_torch:
        if platform.is_jetson:
            manual.append(
                "Install the NVIDIA JetPack-matched torch/torchvision wheel "
                "(never from PyPI): https://docs.nvidia.com/deeplearning/frameworks/"
                "install-pytorch-jetson-platform/"
            )
        elif platform.os_family == "windows":
            manual.append(
                "On Windows with an NVIDIA GPU, install torch from the matching "
                "CUDA index first (install.bat does this); the plain PyPI wheel "
                "is CPU-only. CPU-only is fine? run: pip install torch torchvision"
            )
        else:
            commands.append(["torch", "torchvision"])
    if "rfdetr" in missing or "pytorch_lightning" in missing:
        if platform.is_jetson:
            # --no-deps so rfdetr cannot drag a PyPI torch in behind our back;
            # the training stack is then spelled out explicitly. Several of
            # those packages declare torch as a dependency, so they are only
            # safe to install once the JetPack torch is in place.
            if "rfdetr" in missing:
                commands.append(["rfdetr==1.9.4", "--no-deps"])
            if needs_torch:
                manual.append(
                    "After installing the JetPack torch, re-run 'horos doctor "
                    "--fix' to install the training stack (pytorch_lightning "
                    "and friends declare torch as a dependency and would pull "
                    "the PyPI build in if installed first)."
                )
            else:
                commands.append(
                    [
                        "supervision",
                        "pycocotools",
                        "pytorch_lightning>=2.6,!=2.6.2,!=2.6.3,<3",
                        "torchmetrics[detection]>=1.2",
                        "faster-coco-eval>=1.7.2",
                        "scipy",
                        "peft",
                    ]
                )
        else:
            commands.append(["rfdetr[train]==1.9.4"])
    if "transformers" in missing:
        commands.append(["transformers>=5.1,<6"])
    for name in missing:
        if name in ("torch", "torchvision", "rfdetr", "pytorch_lightning",
                    "transformers"):
            continue
        spec = dict(_RUNTIME_DEPS)[name]
        commands.append([spec])
    return commands, manual


@capability(
    "system.doctor",
    summary="Check installed dependencies against the platform; plan the fixes",
    web_route=None,
    not_web_because="Diagnoses and mutates the local Python environment, not a project.",
    cli="doctor",
)
def doctor_report() -> DoctorReport:
    """`pip install horos` intentionally skips rfdetr/torch on Linux/aarch64
    (the Jetson trap) and cannot pick CUDA builds — this closes the gap:
    it reports what is missing and plans the platform-correct installs that
    `horos doctor --fix` executes."""
    platform = detect_platform()
    deps: list[DependencyStatus] = []
    missing: list[str] = []
    for import_name, spec in _RUNTIME_DEPS:
        version = _installed_version(import_name)
        ok = version is not None
        if not ok:
            missing.append(import_name)
        deps.append(
            DependencyStatus(
                name=import_name, required=spec, installed=version, ok=ok
            )
        )

    cuda = mps = None
    if _installed_version("torch") is not None:
        from horos.backends.env import check_environment

        env = check_environment(emit_warnings=False)
        cuda, mps = env.cuda_available, env.mps_available
        if platform.is_jetson and not cuda:
            for dep in deps:
                if dep.name == "torch":
                    dep.ok = False
                    dep.note = "installed, but CUDA is unavailable on Jetson (PyPI build?)"
            missing.append("torch")

    commands, manual = _plan_fixes(missing, platform)
    return DoctorReport(
        platform=platform,
        dependencies=deps,
        torch_cuda_available=cuda,
        torch_mps_available=mps,
        fix_commands=commands,
        manual_actions=manual,
    )
