"""E1-T3: COCO JSON read/write, including the _annotations.coco.json convention."""

import json

import pytest
from helpers.data import read_json, sample_dataset, write_sample_coco_dir

from horos.core.formats.coco import (
    COCO_CONVENTION_NAME,
    find_annotation_files,
    read_coco,
    write_coco,
)
from horos.errors import DatasetFormatError


def test_write_split_layout_uses_convention_name(tmp_path):
    write_sample_coco_dir(tmp_path)
    assert (tmp_path / "train" / COCO_CONVENTION_NAME).exists()
    assert (tmp_path / "valid" / COCO_CONVENTION_NAME).exists()
    assert not (tmp_path / "test").exists()  # empty split not materialized


def test_read_split_layout_preserves_splits(tmp_path):
    write_sample_coco_dir(tmp_path)
    dataset, image_paths = read_coco(tmp_path)
    assert len(dataset.images) == 3
    assert len(dataset.images_in_split("train")) == 2
    assert len(dataset.images_in_split("valid")) == 1
    assert all(p.exists() for p in image_paths.values())


def test_read_flat_single_json(tmp_path):
    write_sample_coco_dir(tmp_path, split_layout=False)
    dataset, _ = read_coco(tmp_path)
    assert len(dataset.images) == 3
    assert {c.name for c in dataset.categories} == {"forklift", "pallet"}


def test_read_preserves_boxes_and_polygons(tmp_path):
    write_sample_coco_dir(tmp_path)
    dataset, _ = read_coco(tmp_path)
    by_image_count = sorted(len(dataset.annotations_for(i.id)) for i in dataset.images)
    assert by_image_count == [1, 1, 2]
    polygons = [a for a in dataset.annotations if a.segmentation]
    assert len(polygons) == 1
    assert polygons[0].segmentation[0] == [2.0, 2.0, 14.0, 2.0, 14.0, 12.0, 2.0, 12.0]


def test_written_json_is_valid_coco(tmp_path):
    write_sample_coco_dir(tmp_path)
    data = read_json(tmp_path / "train" / COCO_CONVENTION_NAME)
    assert set(data) >= {"images", "annotations", "categories"}
    ann = data["annotations"][0]
    assert set(ann) >= {"id", "image_id", "category_id", "bbox", "area", "iscrowd"}


def test_val_directory_alias_maps_to_valid_split(tmp_path):
    write_sample_coco_dir(tmp_path)
    (tmp_path / "valid").rename(tmp_path / "val")
    dataset, _ = read_coco(tmp_path)
    assert len(dataset.images_in_split("valid")) == 1


def test_missing_annotation_file_is_explicit(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetFormatError, match="No COCO annotation file"):
        find_annotation_files(tmp_path / "empty")


def test_broken_json_is_explicit(tmp_path):
    bad = tmp_path / COCO_CONVENTION_NAME
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="Cannot parse"):
        read_coco(tmp_path)


def test_missing_required_lists_is_explicit(tmp_path):
    bad = tmp_path / COCO_CONVENTION_NAME
    bad.write_text(json.dumps({"images": []}), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="'annotations'"):
        read_coco(tmp_path)


def test_categories_merge_by_name_across_splits(tmp_path):
    write_sample_coco_dir(tmp_path)
    dataset, _ = read_coco(tmp_path)
    assert len(dataset.categories) == 2  # not duplicated per split


def test_rle_segmentation_is_dropped_not_crashed(tmp_path):
    ds = sample_dataset()
    write_coco(ds, tmp_path, split_layout=False)
    data = read_json(tmp_path / COCO_CONVENTION_NAME)
    data["annotations"][0]["segmentation"] = {"counts": "abc", "size": [48, 64]}
    (tmp_path / COCO_CONVENTION_NAME).write_text(json.dumps(data), encoding="utf-8")
    dataset, _ = read_coco(tmp_path)
    assert dataset.annotations[0].segmentation == []


def test_annotation_ids_are_unique_within_each_written_file(tmp_path):
    """Regression: horos numbers annotations per image (every image's first
    box is id 1), but a COCO file needs ids unique across the whole file.

    pycocotools indexes annotations by id in a dict, so duplicates silently
    overwrite each other: metrics and torchvision-style CocoDetection loaders
    (which fetch targets via loadAnns(getAnnIds(...))) then read OTHER images'
    boxes as this image's ground truth. The file looks fine; training just
    quietly produces bad models."""
    from horos.core.dataset import Annotation, Category, Dataset, ImageRecord

    # two images, each with per-image annotation ids restarting at 1
    dataset = Dataset(
        categories=[Category(id=1, name="balloon", color="#e6194b")],
        images=[
            ImageRecord(id=1, file_name="a.png", width=64, height=48, split="train"),
            ImageRecord(id=2, file_name="b.png", width=64, height=48, split="train"),
            ImageRecord(id=3, file_name="c.png", width=64, height=48, split="valid"),
        ],
        annotations=[
            Annotation(id=1, image_id=1, category_id=1, bbox=(1.0, 1.0, 8.0, 8.0)),
            Annotation(id=2, image_id=1, category_id=1, bbox=(2.0, 2.0, 8.0, 8.0)),
            Annotation(id=1, image_id=2, category_id=1, bbox=(3.0, 3.0, 8.0, 8.0)),
            Annotation(id=1, image_id=3, category_id=1, bbox=(4.0, 4.0, 8.0, 8.0)),
        ],
    )
    for path in write_coco(dataset, tmp_path):
        payload = read_json(path)
        ids = [a["id"] for a in payload["annotations"]]
        assert len(ids) == len(set(ids)), f"duplicate annotation ids in {path}"
        # renumbering must not move any box to another image
        by_image: dict[int, list] = {}
        for ann in payload["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann["bbox"])
        for image_id, boxes in by_image.items():
            expected = [
                list(a.bbox) for a in dataset.annotations if a.image_id == image_id
            ]
            assert sorted(boxes) == sorted(expected)


def test_written_coco_survives_a_pycocotools_round_trip(tmp_path):
    """End-to-end guard on the same bug: scoring the ground truth against
    itself must give a perfect score. Any id collision drops mAP well below
    1.0, which is what makes this failure mode so hard to spot by eye."""
    pytest.importorskip("pycocotools")
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ds = sample_dataset()
    # give every image several boxes with per-image ids, as Project does
    from horos.core.dataset import Annotation

    ds.annotations = [
        Annotation(
            id=n + 1,
            image_id=image.id,
            category_id=ds.categories[0].id,
            bbox=(2.0 + n, 2.0 + n, 10.0, 10.0),
        )
        for image in ds.images
        for n in range(3)
    ]
    (written,) = [p for p in write_coco(ds, tmp_path) if p.parent.name == "train"]
    payload = read_json(written)
    detections = [
        {
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": list(a["bbox"]),
            "score": 1.0,
        }
        for a in payload["annotations"]
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(written))
        ev = COCOeval(coco, coco.loadRes(detections), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    assert ev.stats[1] == pytest.approx(1.0), "ground truth must score 1.0 mAP50"
