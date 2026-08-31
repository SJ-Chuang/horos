"""E1-T5: COCO → YOLO → COCO round trip must not lose or distort annotations.

Identity is asserted per (category name, image file, geometry): raw numeric ids
may legitimately be renumbered by a format that has no id concept (YOLO), but
no annotation may be lost, gain/lose its polygon, change class, or move by more
than float-precision epsilon (coordinates are written at full repr precision).
"""

import pytest
from helpers.data import write_sample_coco_dir

from horos.core.formats.coco import read_coco, write_coco
from horos.core.formats.yolo import read_yolo, write_yolo

EPS = 1e-6


def _canonical(dataset):
    """Set of comparable annotation tuples, independent of raw ids."""
    cats = {c.id: c.name for c in dataset.categories}
    images = {i.id: i for i in dataset.images}
    rows = []
    for a in dataset.annotations:
        image = images[a.image_id]
        rows.append(
            (
                image.file_name,
                image.split,
                cats[a.category_id],
                tuple(round(v, 6) for v in a.bbox),
                tuple(tuple(round(v, 6) for v in p) for p in a.segmentation),
            )
        )
    return sorted(rows)


def test_coco_yolo_coco_roundtrip_is_lossless(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco0")
    original, paths0 = read_coco(coco_dir)

    write_yolo(original, tmp_path / "yolo", image_paths=paths0)
    via_yolo, paths1 = read_yolo(tmp_path / "yolo")

    write_coco(via_yolo, tmp_path / "coco1", image_paths=paths1, copy_images=True)
    final, _ = read_coco(tmp_path / "coco1")

    assert _canonical(final) == _canonical(original)
    assert {c.name for c in final.categories} == {c.name for c in original.categories}
    assert len(final.annotations) == len(original.annotations)
    assert len(final.images) == len(original.images)


def test_roundtrip_preserves_splits(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco0")
    original, paths0 = read_coco(coco_dir)
    write_yolo(original, tmp_path / "yolo", image_paths=paths0)
    via_yolo, _ = read_yolo(tmp_path / "yolo")
    for split in ("train", "valid", "test"):
        assert len(via_yolo.images_in_split(split)) == len(
            original.images_in_split(split)
        )


def test_roundtrip_box_coordinates_within_float_epsilon(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco0")
    original, paths0 = read_coco(coco_dir)
    write_yolo(original, tmp_path / "yolo", image_paths=paths0)
    via_yolo, _ = read_yolo(tmp_path / "yolo")

    def by_key(ds):
        cats = {c.id: c.name for c in ds.categories}
        images = {i.id: i.file_name for i in ds.images}
        return {
            (images[a.image_id], cats[a.category_id], round(a.bbox[0])): a.bbox
            for a in ds.annotations
        }

    orig, back = by_key(original), by_key(via_yolo)
    assert orig.keys() == back.keys()
    for key, bbox in orig.items():
        assert bbox == pytest.approx(back[key], abs=EPS)
