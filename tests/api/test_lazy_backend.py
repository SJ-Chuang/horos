"""E4-T11: backends resolve lazily from registry entrypoint strings."""

import sys

import pytest

from horos.backends import get_backend
from horos.core.registry import APACHE_2_0, ModelInfo
from horos.errors import BackendError, UnknownModelError


def _fake_info(entrypoint: str) -> ModelInfo:
    return ModelInfo(
        key="fake-model",
        family="fake",
        display_name="Fake",
        task="detection",
        code_license=APACHE_2_0,
        weights_license=APACHE_2_0,
        license_url="https://example.com/license",
        input_resolution=64,
        params_millions=0.1,
        latency_hint="test",
        entrypoint=entrypoint,
    )


@pytest.fixture
def register_fake(monkeypatch):
    def _register(entrypoint: str):
        info = _fake_info(entrypoint)
        monkeypatch.setattr(
            "horos.core.registry._MODELS",
            {**sys.modules["horos.core.registry"]._MODELS, "fake-model": info},
        )
        return info

    return _register


def test_backend_module_not_imported_until_get_backend(register_fake):
    register_fake("helpers.fake_backend:FakeBackend")
    sys.modules.pop("helpers.fake_backend", None)
    assert "helpers.fake_backend" not in sys.modules
    backend = get_backend("fake-model")
    assert "helpers.fake_backend" in sys.modules
    assert backend.family == "fake"


def test_get_backend_passes_device_through(register_fake):
    register_fake("helpers.fake_backend:FakeBackend")
    backend = get_backend("fake-model", device="cpu")
    assert backend.device == "cpu"


def test_unknown_model_raises(register_fake):
    with pytest.raises(UnknownModelError):
        get_backend("no-such-model")


def test_missing_backend_module_gives_clear_error(register_fake):
    register_fake("horos.backends.nonexistent:Nope")
    with pytest.raises(BackendError) as exc_info:
        get_backend("fake-model")
    assert "could not be imported" in str(exc_info.value)


def test_missing_class_gives_clear_error(register_fake):
    register_fake("helpers.fake_backend:NoSuchClass")
    with pytest.raises(BackendError) as exc_info:
        get_backend("fake-model")
    assert "NoSuchClass" in str(exc_info.value)


def test_malformed_entrypoint_gives_clear_error(register_fake):
    register_fake("helpers.fake_backend.FakeBackend")  # missing ':'
    with pytest.raises(BackendError) as exc_info:
        get_backend("fake-model")
    assert "malformed" in str(exc_info.value)


def test_importing_horos_backends_does_not_import_ml_deps():
    # Cheap in-process guard; the authoritative subprocess check is E4-T10 in
    # tests/test_invariants.py.
    import horos.backends  # noqa: F401

    for mod in ("torch", "rfdetr", "transformers"):
        assert mod not in sys.modules, f"{mod} leaked in via horos.backends"
