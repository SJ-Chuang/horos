"""E9-T1: the capability manifest enumerates the public Python API."""

import inspect

import pytest

import horos.api
from horos.api.manifest import capability, get_capability, list_capabilities

# Names in horos.api.__all__ that are types/helpers, not capabilities.
_NON_CAPABILITY_EXPORTS = {
    "Capability",
    "ImportSummary",
    "PlatformCapabilities",
    "get_capability",
    "list_capabilities",
    "ensure_supported",  # internal guard used by feature entry points
}


def test_manifest_is_not_empty():
    caps = list_capabilities()
    assert len(caps) >= 8
    names = {c.name for c in caps}
    assert {"dataset.import", "dataset.stats", "models.list", "system.capabilities"} <= names


def test_every_public_api_function_is_registered():
    unregistered = []
    for name in horos.api.__all__:
        if name in _NON_CAPABILITY_EXPORTS:
            continue
        obj = getattr(horos.api, name)
        if inspect.isfunction(obj) and not hasattr(obj, "__capability__"):
            unregistered.append(name)
    assert unregistered == [], (
        f"Public API functions missing @capability registration: {unregistered}"
    )


def test_parity_exceptions_are_declared_not_implied():
    for cap in list_capabilities():
        assert cap.web_route or cap.not_web_because, cap.name
        assert cap.cli or cap.not_cli_because, cap.name


def test_web_capabilities_declare_methods():
    for cap in list_capabilities():
        if cap.web_route:
            assert cap.web_methods, cap.name
            assert all(m in {"GET", "POST", "PUT", "DELETE", "PATCH"} for m in cap.web_methods)


def test_capability_lookup():
    cap = get_capability("dataset.stats")
    assert cap is not None
    assert cap.web_route == "/api/v1/dataset/stats"
    assert get_capability("nope") is None


def test_undeclared_exception_is_rejected_at_registration():
    with pytest.raises(ValueError, match="not_web_because"):
        capability("bad.cap", summary="no web story given", cli="x")


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="registered twice"):
        capability(
            "dataset.stats", summary="dup", web_route="/x", cli=None,
            not_cli_because="test",
        )(lambda: None)
