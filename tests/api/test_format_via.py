"""VIA (VGG Image Annotator) import — the balloon-dataset format."""

from __future__ import annotations

import json

import pytest
from helpers.data import make_image

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.core.formats import detect_format
from horos.core.formats.via import read_via
from horos.errors import ClassNamesRequiredError, DatasetFormatError


def _region(shape, attrs=None):
    return {"shape_attributes": shape, "region_attributes": attrs or {}}


def _polygon(xs, ys):
    return {"name": "polygon", "all_points_x": xs, "all_points_y": ys}


def _write_via(root, split, entries):
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, regions in entries.items():
        make_image(split_dir / name, 64, 48)
        payload[f"{name}{1234}"] = {
            "filename": name,
            "size": 1234,
            "regions": regions,
            "file_attributes": {},
        }
    (split_dir / "via_region_data.json").write_text(json.dumps(payload), "utf-8")
    return root


@pytest.fixture
def labeled_via(tmp_path):
    """Two splits; regions labeled through region_attributes (class key)."""
    root = tmp_path / "via_ds"
    _write_via(root, "train", {
        "a.png": [
            _region(_polygon([10, 30, 20], [10, 10, 30]), {"class": "balloon"}),
            _region({"name": "rect", "x": 5, "y": 6, "width": 10, "height": 8},
                    {"class": "kite"}),
        ],
        "b.png": [],  # a negative image: imported, no annotations
    })
    _write_via(root, "val", {
        "c.png": [_region(_polygon([1, 9, 5], [1, 1, 9]), {"class": "balloon"})],
    })
    return root


def test_detection_recognizes_via(labeled_via):
    assert detect_format(labeled_via) == "via"


def test_macosx_zip_junk_is_ignored(labeled_via):
    junk = labeled_via / "__MACOSX" / "train"
    junk.mkdir(parents=True)
    (junk / "via_region_data.json").write_text("not even json", "utf-8")
    dataset, _ = read_via(labeled_via)  # must not try to parse the junk copy
    assert len(dataset.images) == 3


def test_read_labeled_dataset(labeled_via):
    dataset, image_paths = read_via(labeled_via)
    assert {c.name for c in dataset.categories} == {"balloon", "kite"}
    assert len(dataset.images) == 3 and len(dataset.annotations) == 3
    assert len(image_paths) == 3

    by_split = {i.file_name: i.split for i in dataset.images}
    assert by_split == {"a.png": "train", "b.png": "train", "c.png": "valid"}

    polygon = next(a for a in dataset.annotations if a.segmentation)
    assert polygon.segmentation == [[10.0, 10.0, 30.0, 10.0, 20.0, 30.0]]
    assert polygon.bbox == (10.0, 10.0, 20.0, 20.0)  # derived from the points

    rect = next(a for a in dataset.annotations if not a.segmentation)
    assert rect.bbox == (5.0, 6.0, 10.0, 8.0)


def test_unlabeled_regions_collapse_into_one_class(tmp_path):
    root = _write_via(tmp_path / "balloon", "train", {
        "a.png": [_region(_polygon([1, 9, 5], [1, 1, 9]))],
        "b.png": [_region(_polygon([2, 8, 4], [2, 2, 8]))],
    })
    dataset, _ = read_via(root)
    assert [c.name for c in dataset.categories] == ["object"]
    assert any("no class attribute" in w for w in dataset.reader_warnings)

    named, _ = read_via(root, class_names=["balloon"])
    assert [c.name for c in named.categories] == ["balloon"]
    assert named.reader_warnings == []


def test_webui_upload_path_prompts_for_the_class_name(tmp_path):
    root = _write_via(tmp_path / "balloon", "train", {
        "a.png": [_region(_polygon([1, 9, 5], [1, 1, 9]))],
    })
    project = create_project(tmp_path / "proj")
    with pytest.raises(ClassNamesRequiredError) as excinfo:
        import_dataset(project, root, require_class_names=True)
    assert excinfo.value.default_names == ["object"]

    summary = import_dataset(project, root, class_names=["balloon"])
    assert summary.format == "via"
    assert summary.instances_per_category == {"balloon": 1}


def test_labeled_dataset_never_prompts(labeled_via, tmp_path):
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, labeled_via, require_class_names=True)
    assert summary.num_images == 3 and summary.num_categories == 2


def test_circle_becomes_bbox_with_warning(tmp_path):
    root = _write_via(tmp_path / "ds", "train", {
        "a.png": [_region({"name": "circle", "cx": 20, "cy": 20, "r": 5},
                          {"class": "ball"})],
    })
    dataset, _ = read_via(root)
    assert dataset.annotations[0].bbox == (15.0, 15.0, 10.0, 10.0)
    assert any("enclosing bounding box" in w for w in dataset.reader_warnings)


def test_unsupported_shape_is_a_clear_error(tmp_path):
    root = _write_via(tmp_path / "ds", "train", {
        "a.png": [_region({"name": "point", "cx": 3, "cy": 4}, {"class": "x"})],
    })
    with pytest.raises(DatasetFormatError, match="unsupported VIA region shape"):
        read_via(root)


def test_missing_image_is_a_clear_error(tmp_path):
    root = tmp_path / "ds"
    (root / "train").mkdir(parents=True)
    (root / "train" / "via_region_data.json").write_text(json.dumps({
        "ghost.png123": {"filename": "ghost.png", "regions": [], "size": 123},
    }), "utf-8")
    with pytest.raises(DatasetFormatError, match="not found next to"):
        read_via(root)


def test_via1_dict_regions_are_supported(tmp_path):
    """The original balloon dataset keys regions '0', '1', ... in a dict."""
    root = tmp_path / "ds"
    (root / "train").mkdir(parents=True)
    make_image(root / "train" / "a.png", 64, 48)
    (root / "train" / "via_region_data.json").write_text(json.dumps({
        "a.png999": {
            "filename": "a.png", "size": 999,
            "regions": {"0": _region(_polygon([1, 9, 5], [1, 1, 9]),
                                     {"class": "balloon"})},
        },
    }), "utf-8")
    dataset, _ = read_via(root)
    assert len(dataset.annotations) == 1
    assert dataset.categories[0].name == "balloon"
