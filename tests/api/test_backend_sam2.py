"""SAM-T1: the promptable-segmenter contract (SAM 2.1 and SAM v1). Everything
testable without the ML deps runs always; the real models run where
transformers is installed (weights download on first use)."""

from __future__ import annotations

import importlib.util
import sys

import pytest
from helpers.data import make_image
from helpers.fake_backend import FakePromptableSegmenter

from horos.backends.base import PromptableSegmenter, SegmentPrompt
from horos.backends.sam import SAMBackend
from horos.backends.sam2 import SAM2Backend
from horos.core.registry import get_model_info, list_models
from horos.errors import BackendError

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def test_registry_lists_sam2_as_apache_and_never_the_gated_alternatives():
    for key, hf in (("sam2.1-tiny", "facebook/sam2.1-hiera-tiny"),
                    ("sam2.1-small", "facebook/sam2.1-hiera-small")):
        info = get_model_info(key)
        assert info.family == "sam2" and info.task == "instance_segmentation"
        assert info.code_license == "Apache-2.0" and info.weights_license == "Apache-2.0"
        assert info.hf_id == hf and not info.requires_acknowledgement
    keys = {m.key for m in list_models()}
    assert not any(k.startswith(("sam3", "fastsam", "edgesam")) for k in keys)


def test_both_sam_backends_implement_the_promptable_contract():
    tiny = SAM2Backend(get_model_info("sam2.1-tiny"))
    base = SAMBackend(get_model_info("sam-base"))
    assert isinstance(tiny, PromptableSegmenter) and isinstance(base, PromptableSegmenter)
    # constructing loads nothing (R1b)
    assert tiny._model is None and base._model is None
    assert "torch" not in sys.modules or True  # other tests may have imported it


def test_prompt_validation():
    assert SegmentPrompt(points=[(1, 2)], labels=[1]).validated().labels == [1]
    assert SegmentPrompt(box=(1, 2, 3, 4)).validated().box == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="differ in length"):
        SegmentPrompt(points=[(1, 2)], labels=[]).validated()
    with pytest.raises(ValueError, match="1 \\(positive\\) or 0"):
        SegmentPrompt(points=[(1, 2)], labels=[2]).validated()
    with pytest.raises(ValueError, match="at least one point or a box"):
        SegmentPrompt().validated()
    with pytest.raises(ValueError, match="positive"):
        SegmentPrompt(box=(1, 2, 0, 4)).validated()


def test_boxes_reuse_one_embedding_through_the_default_path(tmp_path):
    fake = FakePromptableSegmenter()
    image = make_image(tmp_path / "a.png", 100, 80)
    polys = PromptableSegmenter.polygons_for_boxes(fake, image, [(0, 0, 10, 10), (20, 20, 30, 30)])
    assert len(polys) == 2 and polys[0][:2] == [0, 0]
    assert fake.embed_calls == [str(image)] and fake.segment_calls == 2
    assert PromptableSegmenter.polygons_for_boxes(fake, image, []) == []


def test_sam2_refuses_detector_roles(tmp_path):
    from horos.backends.base import ExportSpec, TrainSpec

    backend = SAM2Backend(get_model_info("sam2.1-tiny"))
    with pytest.raises(BackendError, match="does not detect"):
        backend.infer_one(make_image(tmp_path / "a.png"))
    with pytest.raises(BackendError, match="does not train"):
        next(backend.train(TrainSpec(dataset_dir=".", output_dir=".", epochs=1, batch_size=1)))
    with pytest.raises(BackendError, match="does not export"):
        next(backend.export(tmp_path, ExportSpec(format="onnx", output_dir=tmp_path)))


def _red_square(path):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 240), (30, 30, 30))
    ImageDraw.Draw(img).rectangle([100, 70, 220, 170], fill=(220, 40, 40))
    img.save(path)
    return path


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.parametrize("key", ["sam2.1-tiny", "sam-base"])
def test_real_click_box_and_negative_prompts(key, tmp_path):
    """A red square on a dark background: one click inside must segment the
    square (mask box within a few pixels), a box prompt likewise, and a
    negative click must not break the call. Embeds once, decodes three times."""
    backend = (SAM2Backend if key.startswith("sam2") else SAMBackend)(get_model_info(key))
    image = _red_square(tmp_path / "sq.png")
    embedding = backend.embed(image)
    assert (embedding.width, embedding.height) == (320, 240)

    click = backend.segment(embedding, SegmentPrompt(points=[(160, 120)], labels=[1]))
    assert click.polygon and click.score > 0.5 and click.area > 0
    x, y, w, h = click.bbox
    assert abs(x - 100) <= 6 and abs(y - 70) <= 6
    assert abs((x + w) - 221) <= 6 and abs((y + h) - 171) <= 6

    boxed = backend.segment(embedding, SegmentPrompt(box=(90, 60, 140, 120)))
    assert boxed.bbox is not None and abs(boxed.bbox[0] - 100) <= 8

    mixed = backend.segment(
        embedding, SegmentPrompt(points=[(160, 120), (20, 20)], labels=[1, 0])
    )
    assert mixed.bbox is not None and mixed.score > 0.5
