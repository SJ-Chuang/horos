"""Pascal VOC import (read-only by design — horos exports COCO/YOLO)."""

import pytest
from helpers.data import make_image

from horos.core.formats import detect_format
from horos.core.formats.voc import read_voc
from horos.errors import DatasetFormatError


def _write_xml(path, filename, size, objects):
    body = "".join(
        f"<object><name>{name}</name><bndbox>"
        f"<xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>"
        f"</bndbox></object>"
        for name, (x0, y0, x1, y1) in objects
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<annotation><filename>{filename}</filename>"
        f"<size><width>{size[0]}</width><height>{size[1]}</height></size>"
        f"{body}</annotation>",
        encoding="utf-8",
    )


@pytest.fixture
def voc_dir(tmp_path):
    make_image(tmp_path / "train" / "a.png", 64, 48)
    _write_xml(
        tmp_path / "train" / "a.xml",
        "a.png",
        (64, 48),
        [("helmet", (10, 10, 30, 25)), ("vest", (5, 5, 20, 20))],
    )
    make_image(tmp_path / "valid" / "b.png", 64, 48)
    _write_xml(tmp_path / "valid" / "b.xml", "b.png", (64, 48), [("vest", (1, 2, 11, 22))])
    return tmp_path


def test_detects_voc(voc_dir):
    assert detect_format(voc_dir) == "voc"


def test_read_boxes_and_splits(voc_dir):
    dataset, image_paths = read_voc(voc_dir)
    assert [c.name for c in dataset.categories] == ["helmet", "vest"]
    assert len(dataset.images) == 2
    assert len(dataset.annotations) == 3
    a = next(i for i in dataset.images if i.file_name == "a.png")
    assert a.split == "train"
    helmet = next(
        ann for ann in dataset.annotations_for(a.id)
        if dataset.categories[ann.category_id].name == "helmet"
    )
    assert helmet.bbox == (10.0, 10.0, 20.0, 15.0)  # xyxy -> xywh
    b = next(i for i in dataset.images if i.file_name == "b.png")
    assert b.split == "valid"
    assert all(p.is_file() for p in image_paths.values())


def test_missing_image_is_explicit(tmp_path):
    _write_xml(tmp_path / "a.xml", "gone.png", (64, 48), [("x", (1, 1, 2, 2))])
    with pytest.raises(DatasetFormatError, match="no image file"):
        read_voc(tmp_path)


def test_degenerate_box_is_explicit(tmp_path):
    make_image(tmp_path / "a.png", 64, 48)
    _write_xml(tmp_path / "a.xml", "a.png", (64, 48), [("x", (30, 10, 10, 25))])
    with pytest.raises(DatasetFormatError, match="degenerate box"):
        read_voc(tmp_path)


def test_no_annotations_is_explicit(tmp_path):
    (tmp_path / "notes.xml").write_text("<notes/>", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="No VOC"):
        read_voc(tmp_path)
