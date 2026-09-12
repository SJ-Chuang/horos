"""A tiny ONNX detector with the exported RF-DETR contract, for executor and
converter tests that must not depend on a trained model."""

from __future__ import annotations

from pathlib import Path


def synthetic_detector(
    path: Path, *, resolution: int = 32, classes: int = 3, coupling: float = 0.0
) -> Path:
    """input [1,3,H,W] → dets [1,Q,4] (normalised cxcywh) and labels [1,Q,C]
    logits. Two fixed queries: a confident centre box of class 1 and a 50/50
    small box of class 0. By default the input only contributes a zero so
    the graph keeps a real data path and the outputs are exact; a non-zero
    `coupling` adds `coupling * mean(input)` instead, so the activation has
    a real range — integer quantisation cannot calibrate an always-zero
    tensor (its scale would be 0)."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    dets = np.array([[[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.1, 0.1]]], dtype=np.float32)
    logits = np.full((1, 2, classes), -5.0, dtype=np.float32)
    logits[0, 0, 1] = 3.0   # sigmoid ≈ 0.953
    logits[0, 1, 0] = 0.0   # sigmoid = 0.5
    reduce = "ReduceMean" if coupling else "ReduceSum"
    zero = helper.make_node(reduce, ["input"], ["summed"], keepdims=0)
    scale = helper.make_node("Mul", ["summed", "zero_c"], ["zero"])
    graph = helper.make_graph(
        [
            zero, scale,
            helper.make_node("Add", ["dets_c", "zero"], ["dets"]),
            helper.make_node("Add", ["labels_c", "zero"], ["labels"]),
        ],
        "synthetic",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, resolution, resolution])],
        [
            helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 2, classes]),
        ],
        initializer=[
            numpy_helper.from_array(dets, "dets_c"),
            numpy_helper.from_array(logits, "labels_c"),
            numpy_helper.from_array(np.array(float(coupling), dtype=np.float32), "zero_c"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return Path(path)
