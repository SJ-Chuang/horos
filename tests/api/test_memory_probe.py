"""E5-T6c: cross-platform memory probe feeding the batch-size derivation."""

from __future__ import annotations

import importlib.util

import pytest

from horos.backends.memory import MemoryInfo, _system_ram_bytes, probe_memory

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_system_ram_is_measurable_on_this_platform():
    ram = _system_ram_bytes()
    assert ram is None or ram > 1024**3  # any real machine has >1 GB


def test_cpu_probe_never_claims_availability():
    info = probe_memory("cpu")
    assert info.kind == "cpu"
    assert info.available_gb is None  # unknown by design → deriver is conservative
    assert "RAM" in info.source


def test_auto_probe_returns_a_valid_shape():
    info = probe_memory()
    assert isinstance(info, MemoryInfo)
    assert info.kind in ("cuda", "mps", "cpu")
    if info.kind in ("cuda", "mps"):
        assert info.total_gb and info.total_gb > 0
        assert info.available_gb is not None and info.available_gb >= 0
        assert info.available_gb <= info.total_gb + 1e-6


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_accelerator_probe_matches_selected_device():
    from horos.backends.device import select_device

    assert probe_memory().kind == select_device().kind
