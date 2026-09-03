"""system.install: environment-matched ML-stack planning (`horos install`).

Every test injects the probed values (missing packages, CUDA version, torch
build) — the planner must be a pure function of them, so plans are assertable
regardless of the machine the tests run on.
"""

from horos.api.install import (
    RFDETR_NO_DEPS_SPEC,
    RFDETR_SPEC,
    TRAIN_STACK_SPECS,
    TRANSFORMERS_SPEC,
    cuda_index_url,
    plan_install,
)
from horos.core.platform_info import PlatformInfo

ALL_ML = ["torch", "torchvision", "rfdetr", "pytorch_lightning", "transformers"]


def _plat(os_family="linux", arch="x86_64", is_jetson=False):
    return PlatformInfo(
        os_family=os_family, arch=arch, is_jetson=is_jetson, python_version="3.10.6"
    )


def _plan(platform=None, *, missing=ALL_ML, cuda=None, cpu_build=None, cpu=False):
    return plan_install(
        platform or _plat(),
        cpu=cpu,
        missing=missing,
        cuda_version=cuda,
        torch_cpu_build=cpu_build,
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
