"""E4-T13: structured platform capability list (drives Web API + UI states)."""

import pytest

from horos.api import system as system_mod
from horos.api.system import FEATURES, platform_capabilities
from horos.core.platform_info import PlatformInfo


def _platform(os_family: str, is_jetson: bool = False) -> PlatformInfo:
    return PlatformInfo(
        os_family=os_family,  # type: ignore[arg-type]
        arch="x86_64",
        is_jetson=is_jetson,
        python_version="3.10.0",
    )


@pytest.fixture
def on(monkeypatch):
    def _set(os_family: str, is_jetson: bool = False):
        monkeypatch.setattr(
            system_mod, "detect_platform", lambda: _platform(os_family, is_jetson)
        )

    return _set


def test_every_feature_is_reported(on):
    on("linux")
    caps = platform_capabilities()
    assert [f.feature for f in caps.features] == list(FEATURES)


def test_linux_is_fully_supported(on):
    on("linux")
    caps = platform_capabilities()
    assert all(f.level == "full" for f in caps.features)


def test_macos_tensorrt_unavailable_with_explanation(on):
    on("macos")
    support = platform_capabilities().get("export_tensorrt")
    assert support.level == "unavailable"
    assert not support.available
    assert "macOS" in support.note and "portable" in support.note


def test_macos_training_limited_not_blocked(on):
    on("macos")
    support = platform_capabilities().get("training")
    assert support.level == "limited"
    assert support.available


def test_jetson_training_limited_not_blocked(on):
    on("linux", is_jetson=True)
    support = platform_capabilities().get("training")
    assert support.level == "limited"
    assert support.available
    assert platform_capabilities().get("export_tensorrt").level == "full"


def test_capabilities_serialize_for_web(on):
    on("macos")
    payload = platform_capabilities().model_dump()
    assert payload["platform"]["os_family"] == "macos"


def test_unknown_feature_lookup_is_explicit(on):
    on("linux")
    with pytest.raises(KeyError):
        platform_capabilities().get("time_travel")
