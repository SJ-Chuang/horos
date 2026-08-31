"""E1-T4: YOLO format read/write including data.yaml."""

import pytest
import yaml
from helpers.data import make_image, write_sample_yolo_dir

from horos.core.formats.yolo import read_yolo, write_yolo
from horos.errors import DatasetFormatError


def test_write_creates_expected_layout(tmp_path):
    write_sample_yolo_dir(tmp_path)
    assert (tmp_path / "data.yaml").exists()
    assert (tmp_path / "train" / "images" / "a.png").exists()
    assert (tmp_path / "train" / "labels" / "a.txt").exists()
    assert (tmp_path / "valid" / "labels" / "c.txt").exists()


def test_data_yaml_contents(tmp_path):
    write_sample_yolo_dir(tmp_path)
    config = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert config["names"] == ["forklift", "pallet"]
    assert config["nc"] == 2
    assert config["train"] == "train/images"
    assert config["val"] == "valid/images"


def test_read_back_boxes(tmp_path):
    write_sample_yolo_dir(tmp_path)
    dataset, image_paths = read_yolo(tmp_path)
    assert len(dataset.images) == 3
    assert len(dataset.annotations) == 4
    a_image = next(i for i in dataset.images if i.file_name == "a.png")
    anns = dataset.annotations_for(a_image.id)
    boxes = sorted(tuple(round(v, 6) for v in a.bbox) for a in anns)
    assert boxes == [(4.0, 4.0, 16.0, 12.0), (20.0, 8.0, 8.0, 8.0)]


def test_read_back_polygon(tmp_path):
    write_sample_yolo_dir(tmp_path)
    dataset, _ = read_yolo(tmp_path)
    polys = [a for a in dataset.annotations if a.segmentation]
    assert len(polys) == 1
    assert [round(v, 6) for v in polys[0].segmentation[0]] == [
        2.0, 2.0, 14.0, 2.0, 14.0, 12.0, 2.0, 12.0,
    ]
    # bbox derived from polygon extent
    assert [round(v, 6) for v in polys[0].bbox] == [2.0, 2.0, 12.0, 10.0]


def test_names_as_dict_is_supported(tmp_path):
    write_sample_yolo_dir(tmp_path)
    config = yaml.safe_load((tmp_path / "data.yaml").read_text())
    config["names"] = {0: "forklift", 1: "pallet"}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(config))
    dataset, _ = read_yolo(tmp_path)
    assert [c.name for c in dataset.categories] == ["forklift", "pallet"]


def test_missing_data_yaml_is_explicit(tmp_path):
    with pytest.raises(DatasetFormatError, match="No data.yaml"):
        read_yolo(tmp_path)


def test_bad_label_line_is_explicit(tmp_path):
    write_sample_yolo_dir(tmp_path)
    label = tmp_path / "train" / "labels" / "a.txt"
    label.write_text("0 0.5 0.5 not-a-number 0.2\n")
    with pytest.raises(DatasetFormatError, match="unparsable label line"):
        read_yolo(tmp_path)


def test_class_index_out_of_range_is_explicit(tmp_path):
    write_sample_yolo_dir(tmp_path)
    label = tmp_path / "train" / "labels" / "a.txt"
    label.write_text("7 0.5 0.5 0.2 0.2\n")
    with pytest.raises(DatasetFormatError, match="class index 7"):
        read_yolo(tmp_path)


def test_odd_polygon_coordinate_count_is_explicit(tmp_path):
    write_sample_yolo_dir(tmp_path)
    label = tmp_path / "train" / "labels" / "a.txt"
    label.write_text("0 0.1 0.1 0.5 0.1 0.5\n")
    with pytest.raises(DatasetFormatError, match="even number"):
        read_yolo(tmp_path)


def test_image_without_label_file_reads_as_unannotated(tmp_path):
    write_sample_yolo_dir(tmp_path)
    (tmp_path / "train" / "labels" / "a.txt").unlink()
    dataset, _ = read_yolo(tmp_path)
    a_image = next(i for i in dataset.images if i.file_name == "a.png")
    assert dataset.annotations_for(a_image.id) == []


def test_write_skips_missing_source_images(tmp_path):
    from helpers.data import sample_dataset

    dataset = sample_dataset()
    # only provide one real file
    paths = {1: make_image(tmp_path / "src" / "a.png")}
    write_yolo(dataset, tmp_path / "out", image_paths=paths)
    assert (tmp_path / "out" / "train" / "images" / "a.png").exists()
    assert not (tmp_path / "out" / "train" / "images" / "b.png").exists()
    # labels are still written for every record
    assert (tmp_path / "out" / "train" / "labels" / "b.txt").exists()


def _rewrite_yaml_roboflow_style(root):
    """Rewrite data.yaml paths the way Roboflow exports them: ../<split>/images
    relative to an assumed enclosing datasets dir, with the yaml actually at
    the dataset root."""
    yaml_path = root / "data.yaml"
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    for key in ("train", "val", "test"):
        if key in config:
            config[key] = "../" + str(config[key]).removeprefix("./")
    yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_roboflow_parent_relative_paths(tmp_path):
    # Regression: Roboflow zips fail with "missing directory: ../train/images"
    write_sample_yolo_dir(tmp_path)
    expected, _ = read_yolo(tmp_path)
    _rewrite_yaml_roboflow_style(tmp_path)
    dataset, image_paths = read_yolo(tmp_path)
    assert len(dataset.images) == len(expected.images)
    assert len(dataset.annotations) == len(expected.annotations)
    assert all(p.is_file() for p in image_paths.values())


def test_missing_split_dir_is_still_explicit(tmp_path):
    write_sample_yolo_dir(tmp_path)
    yaml_path = tmp_path / "data.yaml"
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    config["train"] = "../nowhere/images"
    yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="missing directory"):
        read_yolo(tmp_path)
