"""system.install: environment-matched ML-stack planning (`horos install`).

Every test injects the probed values (missing packages, CUDA version, torch
build) — the planner must be a pure function of them, so plans are assertable
regardless of the machine the tests run on.
"""

import pytest

from horos.api.install import (
    ALBUMENTATIONS_SPEC,
    RFDETR_NO_DEPS_SPEC,
    RFDETR_SPEC,
    ROCM_INDEX_URL,
    ROCM_TORCH_VERSION,
    ROCM_TORCHVISION_VERSION,
    TRAIN_STACK_SPECS,
    TRANSFORMERS_SPEC,
    cuda_index_url,
    plan_install,
    version_py_is_cpu_build,
)
from horos.core.platform_info import PlatformInfo
from horos.errors import UnsupportedPlatformError

ALL_ML = [
    "torch",
    "torchvision",
    "rfdetr",
    "pytorch_lightning",
    "albumentations",
    "transformers",
]


def _plat(os_family="linux", arch="x86_64", is_jetson=False):
    return PlatformInfo(
        os_family=os_family, arch=arch, is_jetson=is_jetson, python_version="3.10.6"
    )


def _plan(
    platform=None,
    *,
    missing=ALL_ML,
    cuda=None,
    cpu_build=None,
    cpu=False,
    rocm_arch=None,
    amd=None,
):
    return plan_install(
        platform or _plat(),
        cpu=cpu,
        missing=missing,
        cuda_version=cuda,
        torch_cpu_build=cpu_build,
        rocm_arch=rocm_arch,
        amd_gpu=amd,
    )


def test_cuda_index_picks_newest_the_driver_can_run():
    assert cuda_index_url((13, 2)) == "https://download.pytorch.org/whl/cu132"
    assert cuda_index_url((13, 1)) == "https://download.pytorch.org/whl/cu130"
    assert cuda_index_url((12, 9)) == "https://download.pytorch.org/whl/cu126"
    assert cuda_index_url((12, 4)) == "https://download.pytorch.org/whl/cu124"
    assert cuda_index_url((11, 8)) == "https://download.pytorch.org/whl/cu118"
    assert cuda_index_url((11, 0)) is None  # driver too old for any index


def test_windows_gpu_installs_torch_from_the_matching_index():
    plan = _plan(_plat(os_family="windows"), cuda=(13, 1))
    assert plan.pip_commands[0] == [
        "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu130",
    ]
    # torch lands first so pip sees rfdetr's requirement satisfied
    assert [RFDETR_SPEC] in plan.pip_commands
    assert [TRANSFORMERS_SPEC] in plan.pip_commands
    assert plan.manual_actions == []


def test_windows_without_gpu_installs_the_plain_cpu_wheel():
    plan = _plan(_plat(os_family="windows"), cuda=None)
    assert plan.pip_commands[0] == ["torch", "torchvision"]


def test_linux_gpu_uses_pypi_wheels_that_bundle_cuda():
    plan = _plan(cuda=(12, 8))
    assert plan.pip_commands[0] == ["torch", "torchvision"]


def test_linux_without_gpu_uses_the_cpu_index():
    plan = _plan(cuda=None)
    assert plan.pip_commands[0] == [
        "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu",
    ]


def test_macos_uses_the_universal_pypi_build():
    plan = _plan(_plat(os_family="macos", arch="arm64"), cuda=None)
    assert plan.pip_commands[0] == ["torch", "torchvision"]


def test_cpu_flag_overrides_a_present_gpu():
    plan = _plan(_plat(os_family="windows"), cuda=(13, 1), cpu=True)
    assert plan.pip_commands[0] == ["torch", "torchvision"]
    assert not any("--index-url" in command for command in plan.pip_commands)


def test_driver_older_than_any_index_falls_back_to_cpu_with_a_note():
    plan = _plan(_plat(os_family="windows"), cuda=(11, 0))
    assert plan.pip_commands[0] == ["torch", "torchvision"]
    assert any("driver" in note.lower() for note in plan.notes)


def test_jetson_never_pip_installs_torch():
    plan = _plan(_plat(arch="aarch64", is_jetson=True), cuda=(12, 2))
    flat = [arg for command in plan.pip_commands for arg in command]
    assert "torch" not in flat  # §4: only the JetPack wheel has CUDA there
    assert [RFDETR_NO_DEPS_SPEC, "--no-deps"] in plan.pip_commands
    assert any("JetPack" in action for action in plan.manual_actions)
    # the train stack must wait until the JetPack torch is in place
    assert list(TRAIN_STACK_SPECS) not in plan.pip_commands
    assert any("re-run" in action for action in plan.manual_actions)


def test_jetson_with_torch_present_installs_the_train_stack():
    plan = _plan(
        _plat(arch="aarch64", is_jetson=True),
        missing=["rfdetr", "pytorch_lightning", "transformers"],
        cuda=(12, 2),
    )
    assert [RFDETR_NO_DEPS_SPEC, "--no-deps"] in plan.pip_commands
    assert list(TRAIN_STACK_SPECS) in plan.pip_commands
    assert plan.manual_actions == []


def test_cpu_build_with_gpu_plans_a_force_reinstall():
    # the classic Windows trap: `pip install` gave a CPU torch on a GPU machine
    plan = _plan(
        _plat(os_family="windows"), missing=[], cuda=(13, 1), cpu_build=True
    )
    assert plan.pip_commands == [[
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu130",
        "--force-reinstall",
    ]]
    assert any("CPU-only" in note for note in plan.notes)


def test_cpu_build_without_gpu_is_healthy():
    plan = _plan(missing=[], cuda=None, cpu_build=True)
    assert plan.empty


def test_healthy_environment_plans_nothing():
    plan = _plan(missing=[], cuda=(13, 1), cpu_build=False)
    assert plan.empty


def test_forced_cpu_never_reinstalls_over_a_cpu_build():
    plan = _plan(
        _plat(os_family="windows"), missing=[], cuda=(13, 1), cpu_build=True, cpu=True
    )
    assert plan.empty


def test_albumentations_is_planned_pinned_on_every_platform():
    # rfdetr[train] does not ship it, yet horos's derived aug_config presets
    # need it — without this command training dies at the first step with
    # "Custom Albumentations augmentations require the optional augmentation extra"
    assert ALBUMENTATIONS_SPEC == "albumentations==2.0.8"  # R5 pin, matches pyproject
    for platform in (_plat(), _plat("windows"), _plat("macos", "arm64")):
        assert [ALBUMENTATIONS_SPEC] in _plan(platform).pip_commands
    jetson = _plan(_plat(arch="aarch64", is_jetson=True), missing=["albumentations"])
    # its dependency tree has no torch, so the with-deps install is safe on Jetson
    assert jetson.pip_commands == [[ALBUMENTATIONS_SPEC]] and jetson.manual_actions == []


def test_albumentations_is_never_installed_via_the_rfdetr_augment_extra():
    # rfdetr[augment] would also pull kornia; horos's CPU augmentation
    # backend never needs it
    flat = [arg for command in _plan().pip_commands for arg in command]
    assert not any("augment" in arg for arg in flat)


def test_present_albumentations_is_left_alone():
    plan = _plan(missing=["transformers"])
    assert plan.pip_commands == [[TRANSFORMERS_SPEC]]


def _trt_plan(platform=None, *, cuda=(13, 0), installed=False, cpu=False):
    return plan_install(
        platform or _plat(), cpu=cpu, missing=[], cuda_version=cuda,
        torch_cpu_build=False, tensorrt=True, tensorrt_installed=installed,
    )


def test_tensorrt_is_opt_in_and_follows_the_cuda_major():
    assert not any("tensorrt" in arg for c in _plan(missing=[]).pip_commands for arg in c)
    plan = _trt_plan(cuda=(13, 0))
    assert ["tensorrt-cu13>=10.13,<11"] in plan.pip_commands
    assert any("NVIDIA TensorRT license" in n for n in plan.notes)
    assert ["tensorrt-cu12>=10.13,<11"] in _trt_plan(cuda=(12, 6)).pip_commands
    assert ["tensorrt-cu13>=10.13,<11"] in _trt_plan(_plat("windows")).pip_commands


def test_tensorrt_is_never_pip_installed_where_it_cannot_run():
    already = _trt_plan(installed=True)
    assert already.pip_commands == [] and any("already installed" in n for n in already.notes)
    mac = _trt_plan(_plat("macos", "arm64"), cuda=None)
    assert mac.pip_commands == [] and any("not available on macOS" in n for n in mac.notes)
    nogpu = _trt_plan(cuda=None)
    assert nogpu.pip_commands == [] and any("No NVIDIA GPU" in n for n in nogpu.notes)
    jetson = _trt_plan(_plat(arch="aarch64", is_jetson=True), cuda=(12, 6))
    assert jetson.pip_commands == [] and any("JetPack" in m for m in jetson.manual_actions)


# --------------------------------------------------------------- ROCm (AMD)
# The three wheel flavours as torch actually writes version.py. The ROCm case
# is the regression: it reports cuda = None next to a real hip version, so
# reading `cuda` alone calls a working GPU build "CPU-only" and makes doctor
# tell the user to fix an install that is already correct.
_VERSION_PY_ROCM = (
    "__version__ = '2.13.0+rocm10.0.0'\n"
    "cuda: Optional[str] = None\n"
    "hip: Optional[str] = '7.15.26333'\n"
    "rocm: Optional[str] = '10.0.0'\n"
    "xpu: Optional[str] = None\n"
)
_VERSION_PY_CUDA = (
    "__version__ = '2.14.0+cu130'\n"
    "cuda: Optional[str] = '13.0'\n"
    "hip: Optional[str] = None\n"
    "xpu: Optional[str] = None\n"
)
_VERSION_PY_CPU = (
    "__version__ = '2.14.0+cpu'\n"
    "cuda: Optional[str] = None\n"
    "hip: Optional[str] = None\n"
    "xpu: Optional[str] = None\n"
)
_VERSION_PY_OLD_CPU = "__version__ = '1.13.1+cpu'\ncuda = None\n"
_VERSION_PY_OLD_CUDA = "__version__ = '1.13.1+cu117'\ncuda = '11.7'\n"


def test_rocm_build_is_not_mistaken_for_a_cpu_build():
    assert version_py_is_cpu_build(_VERSION_PY_ROCM) is False
    assert version_py_is_cpu_build(_VERSION_PY_CUDA) is False
    assert version_py_is_cpu_build(_VERSION_PY_CPU) is True
    # torch old enough to have no hip/xpu fields at all
    assert version_py_is_cpu_build(_VERSION_PY_OLD_CPU) is True
    assert version_py_is_cpu_build(_VERSION_PY_OLD_CUDA) is False


def test_an_amd_gpu_is_planned_for_rocm_without_being_asked():
    # Symmetry with NVIDIA: `horos install` detects the GPU and picks the
    # matching wheels. There is no flag to remember.
    plan = _plan(
        _plat(os_family="windows"),
        amd="AMD Radeon RX 9070 XT",
        rocm_arch="gfx1201",
    )
    assert plan.pip_commands[0] == [
        f"torch[device-gfx1201]=={ROCM_TORCH_VERSION}",
        f"torchvision[device-gfx1201]=={ROCM_TORCHVISION_VERSION}",
        "--index-url",
        ROCM_INDEX_URL,
    ]
    # the rest of the stack still comes from PyPI, after torch
    assert [RFDETR_SPEC] in plan.pip_commands
    assert any("gfx1201" in note for note in plan.notes)
    # never both accelerators
    assert plan.cuda_version is None
    assert not any("download.pytorch.org" in arg
                   for command in plan.pip_commands for arg in command)


def test_a_cpu_torch_on_an_amd_box_is_replaced_like_the_cuda_one():
    plan = _plan(
        _plat(os_family="windows"),
        missing=[],
        cpu_build=True,
        amd="AMD Radeon RX 9070 XT",
        rocm_arch="gfx1100",
    )
    assert plan.pip_commands[0][-1] == "--force-reinstall"
    assert "torch[device-gfx1100]" in plan.pip_commands[0][0]
    assert any("reinstalling AMD's ROCm build" in n for n in plan.notes)


def test_an_undetectable_architecture_falls_back_to_cpu_and_says_why():
    # The one case horos cannot fix by itself. It must not guess an
    # architecture (wrong kernels won't run) and must not go quiet either.
    plan = _plan(
        _plat(os_family="windows"), amd="AMD Radeon RX 9070 XT", rocm_arch=None
    )
    assert plan.pip_commands[0] == ["torch", "torchvision"]
    note = next(n for n in plan.notes if "RX 9070 XT" in n)
    assert "HOROS_ROCM_ARCH" in note
    assert "will not guess" in note
    # ... and the same when torch is already installed as the CPU build
    installed = _plan(
        _plat(os_family="windows"), missing=[], cpu_build=True,
        amd="AMD Radeon RX 9070 XT", rocm_arch=None,
    )
    assert installed.pip_commands == []
    assert any("HOROS_ROCM_ARCH" in n for n in installed.notes)


def test_an_architecture_that_is_not_a_gfx_target_is_refused():
    # the value reaches a pip requirement string, so it is validated even
    # though it now arrives from a probe or HOROS_ROCM_ARCH
    for bad in ("", "1201", "gfx", "cuda", "gfx1201; rm -rf /", "--index-url"):
        with pytest.raises(ValueError, match="Invalid ROCm architecture"):
            _plan(_plat(os_family="windows"), rocm_arch=bad)
    # case and surrounding blanks are tolerated
    assert "gfx1201" in _plan(
        _plat("windows"), rocm_arch="  GFX1201 "
    ).pip_commands[0][0]


def test_rocm_is_refused_where_it_cannot_run():
    with pytest.raises(UnsupportedPlatformError, match="no macOS build"):
        _plan(_plat(os_family="macos", arch="arm64"), rocm_arch="gfx1201")
    with pytest.raises(UnsupportedPlatformError, match="JetPack"):
        _plan(_plat(arch="aarch64", is_jetson=True), rocm_arch="gfx1201")


def test_explicit_cpu_choice_is_not_second_guessed_on_an_amd_box():
    plan = _plan(
        _plat(os_family="windows"), cpu=True,
        amd="AMD Radeon RX 9070 XT", rocm_arch="gfx1201",
    )
    assert plan.pip_commands[0] == ["torch", "torchvision"]
    assert not any("ROCm" in note for note in plan.notes)


def test_an_nvidia_driver_keeps_priority_over_the_amd_path():
    plan = _plan(_plat(os_family="windows"), cuda=(13, 0), missing=ALL_ML)
    assert plan.pip_commands[0] == [
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu130",
    ]
    assert not any("amd.com" in arg
                   for command in plan.pip_commands for arg in command)


def test_the_architecture_is_probed_only_when_an_amd_gpu_is_present(monkeypatch):
    from horos.api import install as install_mod

    calls = []
    monkeypatch.setattr(
        install_mod, "detect_rocm_arch", lambda: calls.append(1) or "gfx1201"
    )
    # no AMD GPU: asking for the architecture would be wasted work
    _plan(_plat(os_family="windows"), amd=None, rocm_arch="auto")
    assert calls == []
    _plan(_plat(os_family="windows"), amd="AMD Radeon RX 9070 XT", rocm_arch="auto")
    assert calls == [1]


# --------------------------------------------------------------- TFLite (E8-T3)


def test_tflite_toolchain_is_opt_in():
    assert not any("onnx2tf" in arg for c in _plan(missing=[]).pip_commands for arg in c)
    plan = plan_install(_plat("linux"), missing=[], tflite=True, tflite_installed=False)
    assert ["onnx2tf", "tensorflow>=2.16,<3", "tf-keras", "ai-edge-litert"] in plan.pip_commands
    assert any("Apache 2.0 / MIT" in n for n in plan.notes)
    already = plan_install(_plat("linux"), missing=[], tflite=True, tflite_installed=True)
    assert not any("onnx2tf" in arg for c in already.pip_commands for arg in c)
    assert any("already installed" in n for n in already.notes)
    jetson = plan_install(
        _plat(arch="aarch64", is_jetson=True), missing=[], tflite=True, tflite_installed=False
    )
    assert any("installs no torch" in n for n in jetson.notes)
