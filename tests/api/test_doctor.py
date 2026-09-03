"""system.doctor: dependency check + platform-correct fix planning.

Platform-specific torch/rfdetr planning lives in horos.api.install and is
covered by test_install_plan.py; here we test doctor's own report and the
delegation. `_plan_fixes` probes the live environment for GPU/torch-build
state, so assertions are membership-based (extra mismatch-repair commands may
legitimately appear on some machines).
"""

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
    # the lightweight core (`pip install horos`) is always complete in dev/CI
    light = {"pydantic", "flask", "yaml", "PIL"}
    assert all(d.ok for d in report.dependencies if d.name in light)
    # a broken or incomplete ML stack must always come with a plan
    if not report.ok:
        assert report.fix_commands or report.manual_actions
    else:
        assert report.fix_commands == [] and report.manual_actions == []


def test_missing_rfdetr_plans_pinned_install_with_training_stack():
    commands, manual = _plan_fixes(["rfdetr"], _plat())
    assert ["rfdetr[train]==1.9.4"] in commands and manual == []


def test_missing_training_stack_alone_reinstalls_the_extra():
    # rfdetr installed without [train] (e.g. an old horos env): fix via the extra
    commands, manual = _plan_fixes(["pytorch_lightning"], _plat())
    assert ["rfdetr[train]==1.9.4"] in commands and manual == []


def test_missing_transformers_plans_the_owlv2_range():
    commands, _ = _plan_fixes(["transformers"], _plat())
    assert ["transformers>=5.1.0,<6"] in commands


def test_jetson_never_automates_torch():
    commands, manual = _plan_fixes(["torch", "rfdetr"], _plat(arch="aarch64", is_jetson=True))
    flat = [arg for command in commands for arg in command]
    assert "torch" not in flat  # never pip-install torch on Jetson (§4)
    assert ["rfdetr==1.9.4", "--no-deps"] in commands  # cannot drag torch in
    assert any("JetPack" in m for m in manual)


def test_light_deps_use_their_spec():
    commands, _ = _plan_fixes(["flask"], _plat())
    assert ["flask>=3.0,<4"] in commands
