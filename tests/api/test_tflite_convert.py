"""E8-T3b: int8 post-training quantisation through onnx2tf, calibrated on
preprocessed tensors, yields a model the framework-free executor runs with
the same decoded detections as the float32 graph."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.synthetic import synthetic_detector

from horos.errors import BackendError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def toolchain():
    from horos.backends.convert.tflite import toolchain_available

    if not toolchain_available():
        pytest.skip("TFLite toolchain not installed (horos install --tflite)")


def test_int8_needs_calibration_and_rejects_unknown_precisions(tmp_path, toolchain):
    from horos.backends.convert.tflite import convert_onnx_to_tflite

    graph = synthetic_detector(tmp_path / "m.onnx")
    with pytest.raises(BackendError, match="needs calibration data"):
        convert_onnx_to_tflite(graph, tmp_path / "out", precisions=("float32", "int8"))
    with pytest.raises(BackendError, match="unknown TFLite precision"):
        convert_onnx_to_tflite(graph, tmp_path / "out", precisions=("float32", "bf16"))
    with pytest.raises(BackendError, match="unknown TFLite backend"):
        convert_onnx_to_tflite(graph, tmp_path / "out", backend="trtexec")


def test_dynamic_range_variant_through_the_legacy_converter(tmp_path, toolchain):
    """What RF-DETR ships as "int8": weights-only quantisation from the
    TensorFlow converter backend (NHWC input), decoded by the same executor."""
    from horos.backends.convert.tflite import convert_onnx_to_tflite
    from horos.backends.runtime import ArtifactModel

    graph = synthetic_detector(tmp_path / "m.onnx", coupling=1e-3)
    produced = convert_onnx_to_tflite(
        graph, tmp_path / "out8", stem="det", precisions=("int8_dynamic",),
        backend="tf_converter", pseudo_operators=["Erf"],
    )
    assert list(produced) == ["int8_dynamic"]
    assert produced["int8_dynamic"].name == "det_int8_dynamic.tflite"
    model = ArtifactModel(produced["int8_dynamic"],
                          card={"classes": ["a", "b", "c"], "input": {"shape": [1, 3, 32, 32]}})
    image = make_image(tmp_path / "img.png", 64, 48)
    prediction = model.infer_one(image, threshold=0.3)
    assert model.load().input_shape in ([1, 32, 32, 3], [1, 3, 32, 32])  # NHWC accepted
    assert [i.category_name for i in prediction.instances] == ["b", "a"]
    assert prediction.instances[0].score == pytest.approx(0.9526, abs=0.05)
    assert prediction.instances[0].bbox == pytest.approx((16.0, 12.0, 32.0, 24.0), abs=1.0)


def test_int8_variant_decodes_like_float32(tmp_path, toolchain):
    from horos.backends.convert.tflite import convert_onnx_to_tflite
    from horos.backends.runtime import ArtifactModel

    # a real (tiny) activation range, or the quantiser has nothing to calibrate
    graph = synthetic_detector(tmp_path / "m.onnx", coupling=1e-3)
    rng = np.random.default_rng(0)
    calibration = tmp_path / "calib.npy"
    np.save(calibration, rng.standard_normal((6, 3, 32, 32)).astype(np.float32))
    produced = convert_onnx_to_tflite(
        graph, tmp_path / "out", input_names=["input"], stem="det",
        precisions=("float32", "int8"), calibration=calibration,
    )
    assert sorted(produced) == ["float32", "int8"]
    assert produced["int8"].name == "det_int8.tflite" and produced["int8"].is_file()
    assert not (tmp_path / "out" / "_onnx2tf").exists()  # work dir cleaned up

    card = {"classes": ["a", "b", "c"], "input": {"shape": [1, 3, 32, 32]}}
    image = make_image(tmp_path / "img.png", 64, 48)
    reference = ArtifactModel(produced["float32"], card=card).infer_one(image, threshold=0.3)
    quantised = ArtifactModel(produced["int8"], card=card).infer_one(image, threshold=0.3)
    names = [i.category_name for i in quantised.instances]
    assert names == [i.category_name for i in reference.instances] == ["b", "a"]
    for ref, q in zip(reference.instances, quantised.instances, strict=True):
        assert q.score == pytest.approx(ref.score, abs=0.05)
        assert q.bbox == pytest.approx(ref.bbox, abs=1.0)
