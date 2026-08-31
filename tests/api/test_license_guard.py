"""E4-T9: non-Apache models are blocked unless explicitly acknowledged (§9)."""

import pytest

from horos.backends import get_backend
from horos.errors import BackendError, LicenseError


@pytest.mark.parametrize("key", ["rfdetr-xl", "rfdetr-2xl"])
def test_gated_model_raises_license_error_by_default(key):
    with pytest.raises(LicenseError) as exc_info:
        get_backend(key)
    message = str(exc_info.value)
    assert "PML-1.0" in message  # explains the license difference
    assert "Apache" in message
    assert "acknowledge_non_apache=True" in message  # tells the user the way through


@pytest.mark.parametrize("key", ["rfdetr-xl", "rfdetr-2xl"])
def test_gated_model_passes_guard_with_acknowledgement(key):
    # With acknowledgement, the guard steps aside; resolution then proceeds to
    # the rfdetr backend (which may fail later for other reasons, e.g. the
    # dependency not being installed — that error must NOT be a LicenseError).
    try:
        get_backend(key, acknowledge_non_apache=True)
    except LicenseError:
        pytest.fail("acknowledge_non_apache=True must bypass the license guard")
    except BackendError:
        pass  # acceptable: rfdetr not installed in the test environment


def test_apache_models_need_no_acknowledgement(monkeypatch):
    # Redirect the entrypoint so this test does not import the real rfdetr.
    from horos.core import registry

    info = registry.get_model_info("rfdetr-nano").model_copy(
        update={"entrypoint": "helpers.fake_backend:FakeBackend"}
    )
    monkeypatch.setitem(registry._MODELS, "rfdetr-nano", info)
    backend = get_backend("rfdetr-nano")
    assert backend.info.license == "Apache-2.0"
