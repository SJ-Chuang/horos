"""E4-T6: environment check — the Jetson-without-CUDA warning must be loud."""

import logging

import pytest

from horos.backends import device as device_mod
from horos.backends import env as env_mod
from horos.core.platform_info import PlatformInfo


def _platform(is_jetson: bool) -> PlatformInfo:
    return PlatformInfo(
        os_family="linux",
        arch="aarch64" if is_jetson else "x86_64",
        is_jetson=is_jetson,
        python_version="3.10.0",
    )


@pytest.fixture
def fake_env(monkeypatch):
    def _set(*, jetson: bool, cuda: bool, torch_version: str | None = "2.2.0"):
        monkeypatch.setattr(env_mod, "detect_platform", lambda: _platform(jetson))
        monkeypatch.setattr(env_mod, "_torch_version", lambda: torch_version)
        monkeypatch.setattr(device_mod, "cuda_available", lambda: cuda)
        monkeypatch.setattr(device_mod, "mps_available", lambda: False)

    return _set


def test_jetson_without_cuda_warns(fake_env, caplog):
    fake_env(jetson=True, cuda=False)
    with caplog.at_level(logging.WARNING, logger="horos.backends.env"):
        report = env_mod.check_environment()
    assert env_mod.JETSON_NO_CUDA_WARNING in report.warnings
    assert any("Jetson" in r.message for r in caplog.records)
    assert any("--no-deps" in r.message for r in caplog.records)


def test_jetson_with_cuda_does_not_warn_about_cuda(fake_env):
    fake_env(jetson=True, cuda=True)
    report = env_mod.check_environment()
    assert env_mod.JETSON_NO_CUDA_WARNING not in report.warnings


def test_non_jetson_without_cuda_does_not_get_jetson_warning(fake_env):
    fake_env(jetson=False, cuda=False)
    report = env_mod.check_environment()
    assert env_mod.JETSON_NO_CUDA_WARNING not in report.warnings


def test_missing_torch_reported_but_not_fatal(fake_env):
    fake_env(jetson=False, cuda=False, torch_version=None)
    report = env_mod.check_environment()
    assert report.torch_version is None
    assert any("torch is not installed" in w for w in report.warnings)


def test_report_is_structured(fake_env):
    fake_env(jetson=True, cuda=False)
    report = env_mod.check_environment(emit_warnings=False)
    assert report.platform.is_jetson is True
    assert report.cuda_available is False
    assert report.model_dump()  # serializable for run metadata
