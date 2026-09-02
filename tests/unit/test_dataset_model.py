"""E1-T2: Dataset / Split / Annotation data models, bbox + polygon."""

import pytest
from helpers.data import sample_dataset
from pydantic import ValidationError

from horos.core.dataset import Annotation, Category, Dataset, ImageRecord


def test_bbox_annotation():
    ann = Annotation(id=1, image_id=1, category_id=1, bbox=(1, 2, 3, 4))
    assert ann.area == 12
    assert ann.segmentation == []
    assert ann.source == "manual"
    assert ann.status == "confirmed"


def test_polygon_annotation():
    ann = Annotation(
        id=1, image_id=1, category_id=1,
        bbox=(0, 0, 10, 10),
        segmentation=[[0, 0, 10, 0, 10, 10]],
    )
    assert len(ann.segmentation) == 1
    assert len(ann.segmentation[0]) == 6


def test_auto_annotations_are_distinguishable():
    # E3-T5 foundation: autolabel output must be separable from human work
    ann = Annotation(
        id=1, image_id=1, category_id=1, bbox=(0, 0, 1, 1),
        source="auto", status="pending", score=0.42,
    )
    assert ann.source == "auto" and ann.status == "pending"
    with pytest.raises(ValidationError):
        Annotation(id=1, image_id=1, category_id=1, bbox=(0, 0, 1, 1), source="alien")


def test_category_name_must_not_be_blank():
    with pytest.raises(ValidationError):
        Category(id=1, name="   ")


def test_image_record_rejects_nonpositive_dimensions():
    with pytest.raises(ValidationError):
        ImageRecord(id=1, file_name="x.png", width=0, height=10)


def test_split_values_are_constrained():
    with pytest.raises(ValidationError):
        ImageRecord(id=1, file_name="x.png", width=1, height=1, split="dev")


def test_dataset_lookups():
    ds = sample_dataset()
    assert ds.category_by_name("pallet").id == 2
    assert ds.category_by_id(99) is None
    assert ds.image_by_id(3).split == "valid"
    assert [a.id for a in ds.annotations_for(1)] == [1, 2]
    assert len(ds.images_in_split("train")) == 2


def test_dataset_id_allocation():
    ds = sample_dataset()
    assert ds.next_image_id() == 4
    assert ds.next_annotation_id() == 5
    assert ds.next_category_id() == 3
    assert Dataset().next_image_id() == 1


def test_dataset_serializes_roundtrip():
    ds = sample_dataset()
    clone = Dataset.model_validate_json(ds.model_dump_json())
    assert clone == ds


def test_clamp_to_image_clips_bbox_and_polygons():
    from horos.core.dataset import Annotation, clamp_to_image

    ann = Annotation(
        id=1, image_id=1, category_id=0,
        bbox=(-2.2, 448.2, 901.8, 1133.2),           # the balloon-det case
        segmentation=[[-3.0, 10.0, 50.0, -1.5, 50.0, 2000.0]],
    )
    clamped = clamp_to_image(ann, 2048, 1625)
    x, y, w, h = clamped.bbox
    assert (x, y) == (0.0, 448.2)
    assert x + w <= 2048 and y + h <= 1625
    assert w == pytest.approx(901.8 - 2.2)  # only the x<0 overshoot is cut
    assert h == pytest.approx(1133.2)       # bottom was already inside
    xs = clamped.segmentation[0][0::2]
    ys = clamped.segmentation[0][1::2]
    assert min(xs) >= 0 and max(xs) <= 2048
    assert min(ys) >= 0 and max(ys) <= 1625


def test_clamp_to_image_marks_fully_outside_as_degenerate():
    from horos.core.dataset import Annotation, clamp_to_image

    ann = Annotation(id=1, image_id=1, category_id=0, bbox=(300.0, 50.0, 40.0, 20.0))
    clamped = clamp_to_image(ann, 200, 100)  # box entirely right of the frame
    assert clamped.bbox[2] == 0.0  # zero width — caller drops or errors


def test_in_bounds_annotation_is_unchanged():
    from horos.core.dataset import Annotation, clamp_to_image

    ann = Annotation(id=1, image_id=1, category_id=0, bbox=(10.0, 10.0, 30.0, 20.0))
    assert clamp_to_image(ann, 100, 100).bbox == ann.bbox
