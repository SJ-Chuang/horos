"""E1-S1/S2/S3 API paths: import (dir/zip, copy/reference), convert."""

import zipfile

import pytest
from helpers.data import write_sample_coco_dir, write_sample_yolo_dir

from horos.api.dataset import convert_dataset, import_dataset, import_zip
from horos.api.project import create_project, open_project
from horos.errors import DatasetFormatError


def test_three_line_workflow(tmp_path):
    # E1-S1: existing COCO dir -> horos project in three lines
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, coco_dir)
    assert summary.format == "coco"
    assert summary.num_images == 3
    assert summary.num_annotations == 4
    assert summary.instances_per_category == {"forklift": 2, "pallet": 2}
    assert summary.split_counts == {"train": 2, "valid": 1}


def test_import_copies_images_by_default(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    for record in project.list_images():
        assert (project.images_dir / record.file_name).exists()
        assert record.external_path is None


def test_import_reference_mode(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir, copy_images=False)
    for record in project.list_images():
        assert not (project.images_dir / record.file_name).exists()
        assert record.external_path
        assert project.image_path(record).exists()


def test_import_persists_annotations_per_image(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    reopened = open_project(project.root)
    ds = reopened.to_dataset()
    assert len(ds.annotations) == 4
    assert len([a for a in ds.annotations if a.segmentation]) == 1


def test_import_yolo_autodetected(tmp_path):
    yolo_dir = write_sample_yolo_dir(tmp_path / "yolo")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, yolo_dir)
    assert summary.format == "yolo"
    assert summary.num_images == 3


def test_import_zip(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    zip_path = tmp_path / "upload.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in coco_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(coco_dir))
    project = create_project(tmp_path / "proj")
    summary = import_zip(project, zip_path)
    assert summary.num_images == 3


def test_import_zip_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.txt", "boom")
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="unsafe path"):
        import_zip(project, zip_path)


def test_import_undetectable_format_is_explicit(tmp_path):
    (tmp_path / "mystery").mkdir()
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="Could not detect"):
        import_dataset(project, tmp_path / "mystery")


def test_convert_coco_to_yolo_and_back(tmp_path):
    # E1-S2: CVAT-style conversion without a project
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    yolo_yaml = convert_dataset(coco_dir, tmp_path / "yolo", to_format="yolo")
    assert yolo_yaml.name == "data.yaml"
    convert_dataset(tmp_path / "yolo", tmp_path / "coco2", to_format="coco")
    from horos.core.formats.coco import read_coco

    dataset, _ = read_coco(tmp_path / "coco2")
    assert len(dataset.annotations) == 4


def test_convert_to_same_format_is_rejected(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    with pytest.raises(DatasetFormatError, match="already in format"):
        convert_dataset(coco_dir, tmp_path / "out", to_format="coco")


def test_import_zip_roboflow_layout(tmp_path):
    # Regression: Roboflow YOLO zips (data.yaml with ../<split>/images paths)
    from test_format_yolo import _rewrite_yaml_roboflow_style

    yolo_dir = write_sample_yolo_dir(tmp_path / "yolo")
    _rewrite_yaml_roboflow_style(yolo_dir)
    zip_path = tmp_path / "upload.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in yolo_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(yolo_dir))
    project = create_project(tmp_path / "proj")
    summary = import_zip(project, zip_path)
    assert summary.format == "yolo"
    assert summary.num_images == 3


def test_import_zip_refuses_files_outside_archive(tmp_path):
    # A crafted data.yaml must not be able to pull server-side files into the project
    from helpers.data import make_image

    outside = tmp_path / "outside" / "images"
    make_image(outside / "secret.png")
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.yaml", f"train: {outside}\nnames: ['x']\n")
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="outside the archive"):
        import_zip(project, zip_path)


def test_import_zip_voc_gets_targeted_error(tmp_path):
    from helpers.data import make_image

    src = tmp_path / "voc"
    make_image(src / "img1.png")
    (src / "img1.xml").write_text(
        "<annotation><filename>img1.png</filename></annotation>", encoding="utf-8"
    )
    zip_path = tmp_path / "voc.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in src.rglob("*"):
            zf.write(path, path.relative_to(src))
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="Pascal VOC"):
        import_zip(project, zip_path)


def test_import_zip_darknet_gets_targeted_error(tmp_path):
    from helpers.data import make_image

    src = tmp_path / "darknet"
    make_image(src / "img1.png")
    (src / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (src / "_darknet.labels").write_text("helmet\n", encoding="utf-8")
    zip_path = tmp_path / "darknet.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in src.rglob("*"):
            zf.write(path, path.relative_to(src))
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="Darknet"):
        import_zip(project, zip_path)
