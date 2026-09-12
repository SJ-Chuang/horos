"""E8-T3 on the real RF-DETR backend: train one epoch, export TFLite through
onnx2tf, and check the .tflite reproduces the original weights (E8-T5)."""

from __future__ import annotations

import importlib.util
import json
import time

import pytest
from test_export_e2e import SIZE, _fixture_coco, _wait_job

from horos.api.dataset import import_dataset
from horos.api.export import start_model_export
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status

_MISSING = [
    name
    for name in ("torch", "rfdetr", "pytorch_lightning", "albumentations", "onnx",
                 "onnxruntime", "onnx2tf", "tensorflow")
    if importlib.util.find_spec(name) is None
]
pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"TFLite export stack not installed: {', '.join(_MISSING)}"
)


def test_real_tflite_export_with_parity(tmp_path):
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
    assert training_status(project, record.run_id).run.state == "completed"

    status = _wait_job(project, start_model_export(
        project, record.run_id, format="tflite", options={"int8": True},  # E8-T3b rides along
    ), timeout=1200)
    assert status.state == "completed", status.events[-1]
    phases = [e["phase"] for e in status.events if e["type"] == "progress"]
    assert any("onnx2tf" in p for p in phases)
    assert any("quantising weights to int8" in p for p in phases)
    assert any("int8 model against" in p for p in phases)

    bundle = project.root / "runs" / record.run_id / "exports" / "tflite"
    names = sorted(p.name for p in bundle.iterdir())
    assert "rfdetr-nano_float32.tflite" in names and "rfdetr-nano_float16.tflite" in names
    assert "rfdetr-nano_int8.tflite" in names
    assert not any(n.endswith(".onnx") or n.endswith(".npy") for n in names)  # TFLite only
    card = json.loads((bundle / "model_card.json").read_text("utf-8"))
    assert card["format"] == "tflite" and card["artifact"] == "rfdetr-nano_float32.tflite"
    assert card["input"]["shape"] == [1, 3, SIZE * 3, SIZE * 3]  # NCHW kept
    assert card["parity"]["status"] == "passed", card["parity"]
    assert "TFLite interpreter" in card["parity"]["method"]
    assert card["parity"]["unmatched_detections"] == 0
    assert card["variants"]["float16"]["artifact"] == "rfdetr-nano_float16.tflite"
    int8 = card["variants"]["int8"]
    assert int8["artifact"] == "rfdetr-nano_int8.tflite" and int8["method"] == "dynamic_range"
    assert int8["weights"] == "int8" and int8["input_layout"] == "NHWC"
    # int8 weights drift more than a float16 cast; what must hold is that the
    # check ran on the variant and was recorded — a failed verdict is a warning
    assert int8["parity"]["status"] in ("passed", "failed"), int8["parity"]
    assert "TFLite interpreter" in int8["parity"]["method"]
    size32 = (bundle / "rfdetr-nano_float32.tflite").stat().st_size
    assert (bundle / "rfdetr-nano_int8.tflite").stat().st_size < size32 / 3  # ~4x smaller

    # the int8 file serves through the framework-free executor like any .tflite
    from horos.api.serve import create_inference_server, resolve_source

    source = resolve_source(path=bundle / "rfdetr-nano_int8.tflite")
    assert source.kind == "tflite" and source.classes == card["classes"]
    server = create_inference_server(source, threshold=0.05)
    assert server.health()["runtime"].startswith(("LiteRT", "tensorflow.lite"))
