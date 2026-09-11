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


