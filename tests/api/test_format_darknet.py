"""Darknet import (read-only by design — horos exports COCO/YOLO)."""

import pytest
from helpers.data import make_image

from horos.core.formats import detect_format
from horos.core.formats.darknet import find_labels_file, read_darknet
from horos.errors import DatasetFormatError


@pytest.fixture
def darknet_dir(tmp_path):
    for split, stem, line in [
        ("train", "a", "0 0.5 0.5 0.25 0.5\n"),
        ("train", "b", "1 0.25 0.25 0.5 0.5\n"),
        ("test", "c", ""),
    ]:
        make_image(tmp_path / split / f"{stem}.png", 64, 48)
        (tmp_path / split / f"{stem}.txt").write_text(line, encoding="utf-8")
        (tmp_path / split / "_darknet.labels").write_text(
            "helmet\nvest\n", encoding="utf-8"
        )
    return tmp_path


def test_detects_darknet(darknet_dir):
    assert detect_format(darknet_dir) == "darknet"


def test_detects_darknet_without_labels_file(darknet_dir):
    for labels in darknet_dir.rglob("_darknet.labels"):
        labels.unlink()
    assert detect_format(darknet_dir) == "darknet"


def test_read_with_labels_file(darknet_dir):
    dataset, image_paths = read_darknet(darknet_dir)
    assert [c.name for c in dataset.categories] == ["helmet", "vest"]
    assert len(dataset.images) == 3
    assert len(dataset.annotations) == 2
    a = next(i for i in dataset.images if i.file_name == "a.png")
    assert a.split == "train"
    (ann,) = dataset.annotations_for(a.id)
    # cx=.5 cy=.5 w=.25 h=.5 on 64x48 -> x=24 y=12 w=16 h=24
    assert ann.bbox == (24.0, 12.0, 16.0, 24.0)
    c = next(i for i in dataset.images if i.file_name == "c.png")
    assert c.split == "test"
    assert dataset.annotations_for(c.id) == []


def test_placeholder_names_without_labels_file(darknet_dir):
    for labels in darknet_dir.rglob("_darknet.labels"):
        labels.unlink()
    assert find_labels_file(darknet_dir) is None
    dataset, _ = read_darknet(darknet_dir)
    assert [c.name for c in dataset.categories] == ["0", "1"]


def test_explicit_class_names_override(darknet_dir):
    dataset, _ = read_darknet(darknet_dir, class_names=["casque", "gilet"])
    assert [c.name for c in dataset.categories] == ["casque", "gilet"]


def test_class_index_outside_names_is_explicit(darknet_dir):
    with pytest.raises(DatasetFormatError, match="outside the class-name list"):
        read_darknet(darknet_dir, class_names=["only_one"])


def test_disagreeing_labels_files_are_explicit(darknet_dir):
    (darknet_dir / "test" / "_darknet.labels").write_text(
        "different\n", encoding="utf-8"
    )
    with pytest.raises(DatasetFormatError, match="disagree"):
        read_darknet(darknet_dir)


def test_no_label_files_is_explicit(tmp_path):
    make_image(tmp_path / "a.png")
    with pytest.raises(DatasetFormatError, match="No Darknet label"):
        read_darknet(tmp_path)
