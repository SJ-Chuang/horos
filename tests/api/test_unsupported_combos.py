"""E4-T14: unsupported platform/feature combos raise clearly — no silent fallback."""

import pytest

from horos.api import system as system_mod
from horos.api.system import ensure_supported
from horos.core.platform_info import PlatformInfo
from horos.errors import HorosError, UnsupportedPlatformError


@pytest.fixture
def on(monkeypatch):
    def _set(os_family: str, is_jetson: bool = False):
        monkeypatch.setattr(
            system_mod,
            "detect_platform",
            lambda: PlatformInfo(
                os_family=os_family,  # type: ignore[arg-type]
                arch="arm64",
                is_jetson=is_jetson,
                python_version="3.10.0",
            ),
        )

    return _set


def test_macos_tensorrt_raises_unsupported(on):
    on("macos")
    with pytest.raises(UnsupportedPlatformError) as exc_info:
        ensure_supported("export_tensorrt")
    message = str(exc_info.value)
    assert "not supported" in message
    assert "target" in message  # points the user at the right workflow


def test_unsupported_error_is_a_horos_error(on):
    # E4-S8: a clear typed error, not a weird low-level exception
    on("macos")
    with pytest.raises(HorosError):
        ensure_supported("export_tensorrt")


def test_supported_features_pass_and_return_support(on):
    on("linux")
    support = ensure_supported("export_tensorrt")
    assert support.level == "full"


def test_limited_features_pass_the_guard(on):
    on("macos")
    support = ensure_supported("training")
    assert support.level == "limited"
