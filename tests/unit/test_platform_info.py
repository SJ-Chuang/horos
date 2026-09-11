"""Platform probing that must work before any ML dependency exists.

detect_amd_gpu backs the "your GPU is idle" notice in `horos install` /
`horos doctor` (§4 forbids a silent CPU fallback). The Linux branch reads the
sysfs PCI tree, so it is assertable on any machine with a fake root; the
Windows branch reads the registry and is exercised on Windows CI only.
"""

from __future__ import annotations

import sys

import pytest

from horos.core.platform_info import (
    _detect_amd_gpu_linux,
    detect_amd_gpu,
    detect_rocm_arch,
)

AMD_VENDOR = "0x1002"
NVIDIA_VENDOR = "0x10de"
DISPLAY_CLASS = "0x030000"
AUDIO_CLASS = "0x040300"


def _pci_device(root, slot, vendor, device, pci_class):
    # slot names use underscores, not the real 0000:03:00.0 spelling:
    # Windows forbids ':' in a path and the probe only iterates entries
    entry = root / slot
    entry.mkdir(parents=True)
    (entry / "vendor").write_text(vendor + "\n")
    (entry / "device").write_text(device + "\n")
    (entry / "class").write_text(pci_class + "\n")
    return entry


def test_linux_finds_an_amd_display_controller(tmp_path):
    _pci_device(tmp_path, "0000_03_00_0", AMD_VENDOR, "0x7550", DISPLAY_CLASS)
    found = _detect_amd_gpu_linux(tmp_path)
    assert found is not None
    assert "1002:7550" in found


def test_linux_ignores_the_gpus_audio_function(tmp_path):
    # every AMD card also exposes an HDMI audio device under vendor 0x1002;
    # reporting that as the GPU would be wrong
    _pci_device(tmp_path, "0000_03_00_1", AMD_VENDOR, "0xab30", AUDIO_CLASS)
    assert _detect_amd_gpu_linux(tmp_path) is None


def test_linux_ignores_other_vendors(tmp_path):
    _pci_device(tmp_path, "0000_01_00_0", NVIDIA_VENDOR, "0x2684", DISPLAY_CLASS)
    assert _detect_amd_gpu_linux(tmp_path) is None


def test_linux_without_a_pci_tree_is_not_an_error(tmp_path):
    assert _detect_amd_gpu_linux(tmp_path / "missing") is None


def test_partial_sysfs_entries_are_skipped(tmp_path):
    # a device directory with no class file must not raise
    incomplete = tmp_path / "0000_00_00_0"
    incomplete.mkdir(parents=True)
    (incomplete / "vendor").write_text(AMD_VENDOR)
    _pci_device(tmp_path, "0000_03_00_0", AMD_VENDOR, "0x7550", DISPLAY_CLASS)
    assert "1002:7550" in (_detect_amd_gpu_linux(tmp_path) or "")


def test_detect_amd_gpu_never_raises_on_this_machine():
    # whatever this machine is, the probe must answer rather than explode:
    # it runs inside `horos doctor` on all four platforms
    result = detect_amd_gpu()
    assert result is None or isinstance(result, str)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only branch")
def test_macos_reports_no_amd_gpu_because_rocm_has_no_macos_build():
    assert detect_amd_gpu() is None




# ------------------------------------------------- gfx architecture detection
# Real clinfo output on Windows (AMD's display driver ships clinfo in
# System32), trimmed to the lines that matter. This is the path that carries a
# bare machine: it needs no ROCm and no sysfs.
CLINFO_WINDOWS = """\
Number of platforms:				 1
  Platform Name:				 AMD Accelerated Parallel Processing
  Number of devices:				 1
    Board name:					 AMD Radeon RX 9070 XT
    Name:						 gfx1201
    Max compute units:				 32
"""
ROCMINFO_LINUX = """\
Agent 1
  Name:                    AMD Ryzen 9 7950X
  Device Type:             CPU
Agent 2
  Name:                    gfx90a
  Marketing Name:          AMD Instinct MI210
  Device Type:             GPU
"""


def _fake_tools(monkeypatch, outputs):
    """Stand in for the probe subprocesses. `outputs` maps argv[0] -> stdout;
    anything absent behaves like a tool that is not installed."""
    import subprocess

    from horos.core import platform_info as pi

    monkeypatch.setattr(pi, "_arch_from_rocm_bootstrap", lambda: [])
    monkeypatch.delenv(pi.ROCM_ARCH_ENV, raising=False)

    def fake_run(command, **kwargs):
        if command[0] not in outputs:
            raise FileNotFoundError(command[0])
        return subprocess.CompletedProcess(command, 0, outputs[command[0]], "")

    monkeypatch.setattr(pi.subprocess, "run", fake_run)


def test_arch_comes_from_clinfo_on_a_machine_without_rocm(monkeypatch):
    _fake_tools(monkeypatch, {"clinfo": CLINFO_WINDOWS})
    assert detect_rocm_arch() == "gfx1201"


def test_arch_falls_through_to_rocminfo(monkeypatch):
    _fake_tools(monkeypatch, {"rocminfo": ROCMINFO_LINUX})
    # the CPU agent's name must not be mistaken for a gfx target
    assert detect_rocm_arch() == "gfx90a"


def test_no_probe_available_returns_none_rather_than_a_guess(monkeypatch):
    _fake_tools(monkeypatch, {})
    assert detect_rocm_arch() is None


def test_amds_own_detector_wins_when_installed(monkeypatch):
    from horos.core import platform_info as pi

    _fake_tools(monkeypatch, {"clinfo": CLINFO_WINDOWS})
    monkeypatch.setattr(pi, "_arch_from_rocm_bootstrap", lambda: ["gfx942"])
    assert detect_rocm_arch() == "gfx942"


def test_env_override_beats_every_probe(monkeypatch):
    from horos.core import platform_info as pi

    _fake_tools(monkeypatch, {"clinfo": CLINFO_WINDOWS})
    monkeypatch.setenv(pi.ROCM_ARCH_ENV, "  GFX1100 ")
    assert detect_rocm_arch() == "gfx1100"


def test_mixed_gpu_machine_picks_one_and_says_so(monkeypatch, caplog):
    from horos.core import platform_info as pi

    _fake_tools(monkeypatch, {})
    monkeypatch.setattr(pi, "_arch_from_rocm_bootstrap", lambda: ["gfx1201", "gfx90a"])
    with caplog.at_level("WARNING"):
        arch = detect_rocm_arch()
    assert arch in ("gfx1201", "gfx90a")
    assert "gfx1201" in caplog.text and "gfx90a" in caplog.text


def test_a_failing_probe_is_treated_as_absent(monkeypatch):
    import subprocess

    from horos.core import platform_info as pi

    monkeypatch.setattr(pi, "_arch_from_rocm_bootstrap", lambda: [])
    monkeypatch.delenv(pi.ROCM_ARCH_ENV, raising=False)
    monkeypatch.setattr(
        pi.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 1, "gfx9999", ""),
    )
    # a non-zero exit means the output is not trustworthy
    assert detect_rocm_arch() is None
