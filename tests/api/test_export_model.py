"""E8 model export: the job, the model card (R3), the parity record, the bundle,
and the platform / dependency gates — on the fake backend."""

from __future__ import annotations

import json
import time
import zipfile

import pytest
from helpers.runs import completed_fake_run

import horos.api.export as export_mod
from horos.api.export import (
    export_file_path,
    list_exports,
    model_export_events,
    start_model_export,
)
from horos.api.jobs import job_status
from horos.core.platform_info import PlatformInfo
from horos.errors import ProjectError, UnsupportedPlatformError


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return completed_fake_run(tmp_path_factory.mktemp("run"), epochs=2)


def _wait(project, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = job_status(project, job_id)
        if status.state != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("export job still running")


def test_export_job_writes_artifact_card_and_bundle(run):
    project, record = run
    status = _wait(project, start_model_export(project, record.run_id, format="onnx"))
    assert status.state == "completed" and status.kind == "export"
    phases = [e["phase"] for e in status.events if e["type"] == "progress"]
    assert "exporting" in phases and "writing model card" in phases and "bundling" in phases
    result = status.events[-1]["result"]
    assert result["format"] == "onnx" and result["bundle"] == f"{record.run_id}_onnx.zip"

    exports = project.root / "runs" / record.run_id / "exports"
    with zipfile.ZipFile(exports / result["bundle"]) as zf:
        names = sorted(zf.namelist())
    assert names == ["class_names.txt", "model.onnx", "model_card.json"]

    card = json.loads((exports / "onnx" / "model_card.json").read_text("utf-8"))
    # R3: the license is a first-class field, never assumed
    assert card["weights_license"] == "Apache-2.0" and card["code_license"] == "Apache-2.0"
    assert card["license_url"].startswith("https://")
    assert card["classes"] == ["forklift", "pallet"]
    assert card["dataset"]["fingerprint"].startswith("sha256:")
    assert card["dataset"]["splits"]["train"] == 2
    assert card["metrics"]["loss"] == pytest.approx(0.5)
    assert card["hyperparameters"]["epochs"] == 2
    assert card["input"]["shape"][1:] == [3, card["input"]["shape"][2], card["input"]["shape"][2]]
    # the fake backend has no parity check: recorded honestly, not invented
    assert card["parity"]["status"] == "not_available"
    assert result["parity"]["status"] == "not_available"
    names = (exports / "onnx" / "class_names.txt").read_text("utf-8").split()
    assert names == ["forklift", "pallet"]


def test_pytorch_bundle_and_listing(run):
    project, record = run
    status = _wait(project, start_model_export(project, record.run_id, format="pytorch"))
    assert status.state == "completed"
    listed = {a.name: a for a in list_exports(project, record.run_id)}
    bundle = listed[f"{record.run_id}_pytorch.zip"]
    assert bundle.kind == "model" and bundle.format == "pytorch"
    assert export_file_path(project, record.run_id, bundle.name).is_file()
    with pytest.raises(ProjectError, match="No export named"):
        export_file_path(project, record.run_id, "../run.json")


def test_dataset_fingerprint_is_stable_and_content_bound(run, tmp_path):
    project, record = run
    run_dir = project.root / "runs" / record.run_id
    first = export_mod._dataset_fingerprint(run_dir)
    assert first == export_mod._dataset_fingerprint(run_dir)
    other_project, other = completed_fake_run(tmp_path / "other", epochs=1)
    # the same sample dataset trained elsewhere: identical data, identical fingerprint
    assert export_mod._dataset_fingerprint(
        other_project.root / "runs" / other.run_id
    )["fingerprint"] == first["fingerprint"]


def test_unknown_format_is_refused_synchronously(run):
    project, record = run
    with pytest.raises(ProjectError, match="Unsupported model format"):
        start_model_export(project, record.run_id, format="coreml")


def test_tflite_is_refused_without_the_toolchain_and_exports_with_it(run, monkeypatch):
    project, record = run
    monkeypatch.setattr(export_mod, "_tflite_available", lambda: False)
    with pytest.raises(ProjectError, match="horos install --tflite"):
        start_model_export(project, record.run_id, format="tflite")

    # with the toolchain present the pipeline runs end to end (fake backend
    # here; the real conversion is tests/api/test_export_tflite.py)
    monkeypatch.setattr(export_mod, "_tflite_available", lambda: True)
    status = _wait(project, start_model_export(project, record.run_id, format="tflite"))
    assert status.state == "completed", status.events[-1]
    card = json.loads(
        (project.root / "runs" / record.run_id / "exports" / "tflite" / "model_card.json")
        .read_text("utf-8")
    )
    assert card["format"] == "tflite" and card["artifact"] == "model.tflite"
    assert card["weights_license"] == "Apache-2.0"  # R3 on every format
    assert "Portable" in card["portability"]


def test_tensorrt_is_refused_on_macos_and_without_the_package(run, monkeypatch):
    project, record = run
    import horos.api.system as system

    monkeypatch.setattr(
        system, "detect_platform",
        lambda: PlatformInfo(os_family="macos", arch="arm64", is_jetson=False,
                             python_version="3.12.0"),
    )
    with pytest.raises(UnsupportedPlatformError, match="TensorRT is not supported on macOS"):
        start_model_export(project, record.run_id, format="tensorrt")

    monkeypatch.setattr(
        system, "detect_platform",
        lambda: PlatformInfo(os_family="linux", arch="x86_64", is_jetson=False,
                             python_version="3.12.0"),
    )
    monkeypatch.setattr(export_mod, "_tensorrt_available", lambda: False)
    with pytest.raises(ProjectError, match="horos install --tensorrt"):
        start_model_export(project, record.run_id, format="tensorrt")


def test_tensorrt_stream_warns_about_portability(run, monkeypatch):
    project, record = run
    monkeypatch.setattr(export_mod, "_tensorrt_available", lambda: True)
    import horos.api.system as system

    monkeypatch.setattr(
        system, "detect_platform",
        lambda: PlatformInfo(os_family="linux", arch="x86_64", is_jetson=False,
                             python_version="3.12.0"),
    )
    events = list(model_export_events(project, record.run_id, format="tensorrt"))
    assert events[0].type == "started"
    assert any(e.type == "warning" and "not load on another GPU" in e.message for e in events)
    assert events[-1].type == "completed"
    assert events[-1].result["model_card"]["portability"].startswith("A TensorRT engine")


def test_incomplete_run_is_refused(tmp_path):
    from helpers.data import write_sample_coco_dir
    from helpers.runs import FAKE, ensure_worker_can_import_helpers

    from horos.api.dataset import import_dataset
    from horos.api.project import create_project
    from horos.api.train import TrainRunConfig, start_training, training_status

    ensure_worker_can_import_helpers()
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=1, extra={"fail": True})
    )
    deadline = time.monotonic() + 60
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    with pytest.raises(ProjectError, match="only completed runs"):
        start_model_export(project, record.run_id, format="onnx")
