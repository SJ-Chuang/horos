"""system.doctor: dependency check + platform-correct fix planning."""

from horos.api.system import _plan_fixes, doctor_report
from horos.core.platform_info import PlatformInfo


def _plat(os_family="linux", arch="x86_64", is_jetson=False):
    return PlatformInfo(
        os_family=os_family, arch=arch, is_jetson=is_jetson, python_version="3.10.6"
    )


def test_report_on_current_env():
    report = doctor_report()
    names = [d.name for d in report.dependencies]
    assert {"pydantic", "flask", "torch", "rfdetr", "transformers"} <= set(names)
    # this dev env has everything installed
    assert report.ok
    assert report.fix_commands == [] and report.manual_actions == []


def test_missing_rfdetr_plans_pinned_install():
    commands, manual = _plan_fixes(["rfdetr"], _plat())
    assert ["rfdetr==1.9.4"] in commands and manual == []


def test_jetson_never_automates_torch():
    commands, manual = _plan_fixes(["torch", "rfdetr"], _plat(arch="aarch64", is_jetson=True))
    flat = [arg for command in commands for arg in command]
    assert "torch" not in flat  # never pip-install torch on Jetson (§4)
    assert ["rfdetr==1.9.4", "--no-deps"] in commands  # cannot drag torch in
    assert any("JetPack" in m for m in manual)


def test_linux_without_jetson_installs_torch():
    commands, manual = _plan_fixes(["torch"], _plat())
    assert ["torch", "torchvision"] in commands and manual == []


def test_windows_torch_is_a_manual_note():
    commands, manual = _plan_fixes(["torch"], _plat(os_family="windows"))
    flat = [arg for command in commands for arg in command]
    assert "torch" not in flat
    assert any("CUDA index" in m for m in manual)


def test_light_deps_use_their_spec():
    commands, _ = _plan_fixes(["flask"], _plat())
    assert ["flask>=3.0,<4"] in commands
