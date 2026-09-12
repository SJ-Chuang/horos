"""Serve-T1 on the real RF-DETR backend: train one epoch, export ONNX,
TensorRT and TFLite, and check that `horos serve`'s framework-free executor
gives the same detections from every artifact (E8-S6 across formats)."""

from __future__ import annotations

import importlib.util
import time

import pytest
from test_export_e2e import SIZE, _fixture_coco, _wait_job

from horos.api.dataset import import_dataset
from horos.api.export import start_model_export
from horos.api.project import create_project
from horos.api.serve import create_inference_server, resolve_source
from horos.api.train import TrainRunConfig, start_training, training_status

_MISSING = [
    name
    for name in ("torch", "rfdetr", "pytorch_lightning", "albumentations", "onnx",
                 "onnxruntime", "tensorrt", "polygraphy", "onnx2tf", "tensorflow")
    if importlib.util.find_spec(name) is None
]
pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"engine serving stack not installed: {', '.join(_MISSING)}"
)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _top_matches(reference, other, *, top=3, within=10, score_tolerance=0.05, iou_min=0.8):
    """The reference's `top` strongest detections each have a same-class
    partner among `other`'s `within` strongest — what a user sees agrees,
    whichever runtime produced it (raw query order is not comparable, E8-T5)."""
    ref = sorted(reference, key=lambda i: -i.score)[:top]
    pool = sorted(other, key=lambda i: -i.score)[:within]
    assert ref and pool, (reference, other)
    for inst in ref:
        partner = max(pool, key=lambda o: _iou(inst.bbox, o.bbox))
        assert partner.category_id == inst.category_id, (inst, partner)
        assert _iou(inst.bbox, partner.bbox) >= iou_min, (inst, partner)
        assert abs(partner.score - inst.score) <= score_tolerance, (inst, partner)


def test_every_exported_artifact_serves_the_same_detections(tmp_path):
    from horos.backends.device import cuda_available

    if not cuda_available():
        pytest.skip("a TensorRT engine needs a CUDA device")
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

    for fmt in ("onnx", "tensorrt", "tflite"):
        status = _wait_job(project, start_model_export(project, record.run_id, format=fmt),
                           timeout=900)
        assert status.state == "completed", (fmt, status.events[-1])

    image = next((tmp_path / "coco" / "train").glob("*.png"))  # a training image
    servers = {}
    for fmt in ("onnx", "tensorrt", "tflite"):
        source = resolve_source(project, run_id=record.run_id, format=fmt)
        assert source.kind == fmt and source.card["format"] == fmt
        servers[fmt] = create_inference_server(source, threshold=0.05)
    assert servers["onnx"].device in ("cuda", "cpu")
    assert servers["tensorrt"].device == "cuda"
    assert servers["tflite"].device == "cpu"
    assert servers["tensorrt"].health()["runtime"].startswith("TensorRT")

    # threshold 0 keeps every query: a one-epoch model scores low everywhere,
    # so compare the strongest detections rather than a thresholded set
    predictions = {fmt: s.predict(image, threshold=0.0) for fmt, s in servers.items()}
    reference = predictions["onnx"].instances
    assert len(reference) > 0
    _top_matches(reference, predictions["tflite"].instances)
    # fp16 engines drift more than float32 graphs
    _top_matches(reference, predictions["tensorrt"].instances, score_tolerance=0.1)
