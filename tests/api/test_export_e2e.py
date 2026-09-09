"""E8 end to end on the real RF-DETR backend: train one epoch on a tiny fixture,
export the PyTorch bundle and ONNX, and check the ONNX graph reproduces the
original weights' outputs (E8-T5, the acceptance gate for export)."""

from __future__ import annotations

import importlib.util
import json
import random
import time
import zipfile
from pathlib import Path

import pytest
from helpers.data import make_image

from horos.api.dataset import import_dataset
from horos.api.export import start_model_export
from horos.api.jobs import job_status
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status

_MISSING = [
    name
    for name in ("torch", "rfdetr", "pytorch_lightning", "albumentations", "onnx", "onnxruntime")
    if importlib.util.find_spec(name) is None
]
pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"export stack not installed: {', '.join(_MISSING)}"
)

IMAGES = 16
SIZE = 128


def _fixture_coco(root: Path) -> Path:
    rng = random.Random(7)
    for split, count in (("train", IMAGES - 4), ("valid", 4)):
        images, annotations = [], []
        for i in range(count):
            name = f"{split}_{i:02d}.png"
            make_image(root / split / name, SIZE, SIZE, color=(rng.randrange(256), 90, 40))
            images.append({"id": i + 1, "file_name": name, "width": SIZE, "height": SIZE})
            x, y = rng.randrange(0, SIZE - 40), rng.randrange(0, SIZE - 40)
            annotations.append({"id": i + 1, "image_id": i + 1, "category_id": 1,
                                "bbox": [x, y, 32, 32], "area": 1024, "iscrowd": 0,
                                "segmentation": []})
        (root / split / "_annotations.coco.json").write_text(json.dumps({
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "thing", "supercategory": "none"}],
        }), "utf-8")
    return root


def _wait_job(project, job_id, timeout=600.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = job_status(project, job_id)
        if status.state != "running":
            return status
        time.sleep(0.5)
    raise AssertionError("export job still running")


def test_real_pytorch_and_onnx_export_with_parity(tmp_path):
    project = create_project(tmp_path / "proj")
    import_dataset(project, _fixture_coco(tmp_path / "coco"))
    record = start_training(
        project,
        TrainRunConfig(model="rfdetr-nano", epochs=1, batch_size=4, resolution=SIZE * 3),
    )
    deadline = time.monotonic() + 900
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(1.0)
    status = training_status(project, record.run_id)
    assert status.run.state == "completed", status.run.error

    exports = project.root / "runs" / record.run_id / "exports"

    pt = _wait_job(project, start_model_export(project, record.run_id, format="pytorch"))
    assert pt.state == "completed", pt.events[-1]
    with zipfile.ZipFile(exports / pt.events[-1]["result"]["bundle"]) as zf:
        assert {"weights.pt", "class_names.txt", "model_card.json"} <= set(zf.namelist())
    assert (exports / "pytorch" / "class_names.txt").read_text("utf-8").split() == ["thing"]

    onnx = _wait_job(project, start_model_export(project, record.run_id, format="onnx"))
    assert onnx.state == "completed", onnx.events[-1]
    result = onnx.events[-1]["result"]
    assert Path(result["artifact"]).suffix == ".onnx"
    card = json.loads((exports / "onnx" / "model_card.json").read_text("utf-8"))
    assert card["weights_license"] == "Apache-2.0"
    assert card["input"]["mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert [o["name"] for o in card["outputs"]] == ["dets", "labels"]
    # E8-T5: same inputs, same detections (class, IoU >= 0.9, score within tolerance)
    parity = card["parity"]
    assert parity["status"] == "passed", parity
    assert parity["images"] >= 1 and parity["unmatched_detections"] == 0
    assert "raw_max_abs_diff" in parity  # recorded for the record, not the gate
