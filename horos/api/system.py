"""Platform capabilities (E4-T13/T14) and the model catalog (E4-S1/S2).

The capability list is the single source of truth for what works where (§4).
The Web API serves it verbatim and the WebUI drives button-disabled states
from it — platform checks are never hard-coded in the frontend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from horos.api.install import ML_IMPORT_NAMES, plan_install, torch_is_cpu_build
from horos.api.manifest import capability
from horos.core.platform_info import PlatformInfo, detect_cuda_version, detect_platform
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
    """Light deps get their spec verbatim; the ML stack (torch, rfdetr,
    transformers) is delegated to the `horos install` planner, so doctor and
    install can never disagree about the platform-correct sources."""
    plan = plan_install(
        platform, missing=[name for name in missing if name in ML_IMPORT_NAMES]
    )
    commands = list(plan.pip_commands)
    for name in missing:
        if name in ML_IMPORT_NAMES:
            continue
        spec = dict(_RUNTIME_DEPS)[name]
        commands.append([spec])
    return commands, list(plan.manual_actions)


@capability(
    "system.doctor",
    summary="Check installed dependencies against the platform; plan the fixes",
    web_route=None,
    not_web_because="Diagnoses and mutates the local Python environment, not a project.",
    cli="doctor",
)
def doctor_report() -> DoctorReport:
    """`pip install horos` intentionally ships without the ML stack (torch,
    rfdetr, transformers) — its correct source is platform-specific and
    `horos install` picks it. This report closes the loop: it says what is
    missing or mis-built and plans the same platform-correct installs that
    `horos install` (or `horos doctor --fix`) executes."""
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
        elif not cuda and torch_is_cpu_build() and detect_cuda_version() is not None:
            # the classic Windows trap: pip's default wheel is CPU-only, so a
            # perfectly healthy GPU machine ends up training on CPU. torch
            # stays out of `missing` (it IS installed) — the planner's
            # mismatch path emits the --force-reinstall command instead.
            for dep in deps:
                if dep.name == "torch":
                    dep.ok = False
                    dep.note = (
                        "CPU-only build, but an NVIDIA GPU is present — "
                        "run 'horos install' to switch to the CUDA build"
                    )

    commands, manual = _plan_fixes(missing, platform)
    return DoctorReport(
        platform=platform,
        dependencies=deps,
        torch_cuda_available=cuda,
        torch_mps_available=mps,
        fix_commands=commands,
        manual_actions=manual,
    )
