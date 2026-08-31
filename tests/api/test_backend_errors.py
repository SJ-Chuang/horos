"""E4-T5: backend exceptions are translated into horos error types (E4-S6)."""

from pathlib import Path

import pytest
from helpers.fake_backend import ExplodingBackend

from horos.backends.base import TrainSpec, translate_backend_errors
from horos.errors import (
    BackendError,
    BackendOutOfMemoryError,
    HorosError,
    LicenseError,
)


def test_foreign_exception_becomes_backend_error():
    with pytest.raises(BackendError) as exc_info:
        with translate_backend_errors("fake"):
            raise RuntimeError("simulated library failure")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "simulated library failure" in str(exc_info.value)
    assert exc_info.value.backend == "fake"


def test_oom_message_is_recognized_as_oom_error():
    with pytest.raises(BackendOutOfMemoryError):
        with translate_backend_errors("fake"):
            raise RuntimeError("CUDA error: Out Of Memory")


def test_memory_error_is_recognized_as_oom_error():
    with pytest.raises(BackendOutOfMemoryError):
        with translate_backend_errors("fake"):
            raise MemoryError()


def test_horos_errors_pass_through_untranslated():
    with pytest.raises(LicenseError):
        with translate_backend_errors("fake"):
            raise LicenseError("already a horos error")


def test_translated_errors_are_horos_errors_via_backend_methods(tmp_path):
    from horos.core.registry import get_model_info

    backend = ExplodingBackend(get_model_info("rfdetr-nano"))
    with pytest.raises(HorosError):
        backend.infer_one(Path("x.jpg"))
    spec = TrainSpec(dataset_dir=tmp_path, output_dir=tmp_path, epochs=1, batch_size=1)
    with pytest.raises(BackendOutOfMemoryError):
        list(backend.train(spec))
