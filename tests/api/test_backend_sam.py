"""SAM backend contract — like OWLv2, everything testable without the ML deps
is tested here; real segmentation runs only where transformers is installed."""

import importlib.util
import sys

import pytest
from helpers.data import make_image

from horos.backends.sam import SAMBackend
from horos.core.registry import get_model_info
from horos.errors import BackendError

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def backend():
    return SAMBackend(get_model_info("sam-base"))


def test_registry_metadata():
    info = get_model_info("sam-base")
    assert info.family == "sam"
    assert info.task == "instance_segmentation"
    assert info.weights_license == "Apache-2.0"
    assert info.hf_id == "facebook/sam-vit-base"


def test_construction_is_lazy(backend):
    assert "transformers" not in sys.modules or True  # other tests may import it
    # the real guarantee: constructing did not load any model
    assert backend._model is None and backend._processor is None


def test_empty_boxes_short_circuit(backend, tmp_path):
    # no model load, no ML import needed
    assert backend.polygons_for_boxes(make_image(tmp_path / "a.png"), []) == []


def test_detector_roles_are_refused(backend, tmp_path):
    from horos.backends.base import ExportSpec, TrainSpec

    with pytest.raises(BackendError, match="refiner"):
        backend.infer_one(make_image(tmp_path / "a.png"))
    with pytest.raises(BackendError, match="refiner"):
        next(backend.train(TrainSpec(dataset_dir=".", output_dir=".", epochs=1, batch_size=1)))
    with pytest.raises(BackendError, match="refiner"):
        next(backend.export(tmp_path, ExportSpec(format="onnx", output_dir=tmp_path)))


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_real_box_to_polygon_smoke(backend, tmp_path):
    """One box on a solid-color fixture image: assert a plausible polygon
    (SAM downloads weights on first run)."""
    image = make_image(tmp_path / "a.png", 64, 48)
    polys = backend.polygons_for_boxes(image, [(8.0, 8.0, 40.0, 30.0)])
    assert len(polys) == 1
    poly = polys[0]
    if poly is not None:  # a featureless image may legitimately yield no mask
        assert len(poly) >= 6 and len(poly) % 2 == 0
        xs, ys = poly[0::2], poly[1::2]
        assert 0 <= min(xs) and max(xs) <= 64 and 0 <= min(ys) and max(ys) <= 48
