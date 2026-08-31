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
