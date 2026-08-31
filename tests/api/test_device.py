"""E4-T12: device selection — CUDA → MPS → CPU, forced choice validated."""

import pytest

from horos.backends import device as device_mod
from horos.errors import UnsupportedPlatformError


@pytest.fixture
def fake_availability(monkeypatch):
    def _set(cuda: bool, mps: bool):
        monkeypatch.setattr(device_mod, "cuda_available", lambda: cuda)
        monkeypatch.setattr(device_mod, "mps_available", lambda: mps)
        if cuda:
            # avoid touching torch for the device name
            monkeypatch.setattr(device_mod, "_torch", lambda: None)

    return _set


def test_priority_cuda_first(fake_availability):
    fake_availability(cuda=True, mps=True)
    assert device_mod.select_device().kind == "cuda"


def test_priority_mps_second(fake_availability):
    fake_availability(cuda=False, mps=True)
    info = device_mod.select_device()
    assert info.kind == "mps"
    assert info.torch_device == "mps"


def test_priority_cpu_last(fake_availability):
    fake_availability(cuda=False, mps=False)
    assert device_mod.select_device().kind == "cpu"


def test_forcing_available_device_works(fake_availability):
    fake_availability(cuda=False, mps=True)
    assert device_mod.select_device(prefer="mps").kind == "mps"


def test_forcing_unavailable_cuda_raises_not_falls_back(fake_availability):
    fake_availability(cuda=False, mps=True)
    with pytest.raises(UnsupportedPlatformError):
        device_mod.select_device(prefer="cuda")


def test_forcing_cpu_always_works(fake_availability):
    fake_availability(cuda=True, mps=True)
    assert device_mod.select_device(prefer="cpu").kind == "cpu"


def test_probes_survive_missing_torch(monkeypatch):
    monkeypatch.setattr(device_mod, "_torch", lambda: None)
    assert device_mod.cuda_available() is False
    assert device_mod.mps_available() is False
    assert device_mod.select_device().kind == "cpu"
