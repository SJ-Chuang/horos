"""E1-T6: the validator names each problem class explicitly (E1-S4)."""

from helpers.data import make_image, sample_dataset

from horos.core.dataset import Annotation, Category, Dataset, ImageRecord
from horos.core.validate import validate_dataset


def _clean_dataset(tmp_path):
    ds = sample_dataset()
    for record in ds.images:
        make_image(tmp_path / record.file_name, record.width, record.height)
    return ds


def test_clean_dataset_passes(tmp_path):
    report = validate_dataset(_clean_dataset(tmp_path), images_root=tmp_path)
    assert report.ok
    assert report.issues == []


def test_missing_image_file(tmp_path):
    ds = _clean_dataset(tmp_path)
    (tmp_path / "b.png").unlink()
    report = validate_dataset(ds, images_root=tmp_path)
    assert not report.ok
    issue = next(i for i in report.issues if i.kind == "missing_image_file")
    assert "b.png" in issue.message
    assert issue.image_id == 2


def test_bbox_out_of_bounds(tmp_path):
    ds = _clean_dataset(tmp_path)
    ds.annotations[0] = ds.annotations[0].model_copy(
        update={"bbox": (60.0, 40.0, 20.0, 20.0)}  # exceeds 64x48
    )
    report = validate_dataset(ds, images_root=tmp_path)
    issue = next(i for i in report.issues if i.kind == "bbox_out_of_bounds")
    assert "64x48" in issue.message
    assert issue.annotation_id == ds.annotations[0].id


def test_invalid_box_size(tmp_path):
    ds = _clean_dataset(tmp_path)
    ds.annotations[0] = ds.annotations[0].model_copy(update={"bbox": (5.0, 5.0, 0.0, 4.0)})
    report = validate_dataset(ds, images_root=tmp_path)
    issue = next(i for i in report.issues if i.kind == "invalid_box_size")
    assert "non-positive" in issue.message


def test_unknown_category(tmp_path):
    ds = _clean_dataset(tmp_path)
    ds.annotations[0] = ds.annotations[0].model_copy(update={"category_id": 42})
    report = validate_dataset(ds, images_root=tmp_path)
    issue = next(i for i in report.issues if i.kind == "unknown_category")
    assert "42" in issue.message and "[1, 2]" in issue.message


def test_invalid_polygon(tmp_path):
    ds = _clean_dataset(tmp_path)
    ds.annotations[3] = ds.annotations[3].model_copy(
        update={"segmentation": [[1.0, 2.0, 3.0]]}
    )
    report = validate_dataset(ds, images_root=tmp_path)
    issue = next(i for i in report.issues if i.kind == "invalid_polygon")
    assert "3 coordinates" in issue.message


def test_non_contiguous_category_ids_is_warning_not_error():
    ds = Dataset(
        categories=[Category(id=1, name="a"), Category(id=7, name="b")],
        images=[ImageRecord(id=1, file_name="x.png", width=10, height=10)],
        annotations=[Annotation(id=1, image_id=1, category_id=1, bbox=(0, 0, 2, 2))],
    )
    report = validate_dataset(ds)
    issue = next(i for i in report.issues if i.kind == "non_contiguous_category_ids")
    assert issue.level == "warning"
    assert report.ok  # warnings alone do not fail validation


def test_counts_summarize_issue_kinds(tmp_path):
    ds = _clean_dataset(tmp_path)
    ds.annotations[0] = ds.annotations[0].model_copy(update={"category_id": 42})
    ds.annotations[1] = ds.annotations[1].model_copy(update={"bbox": (0.0, 0.0, -1.0, 5.0)})
    report = validate_dataset(ds, images_root=tmp_path)
    counts = report.counts()
    assert counts["unknown_category"] == 1
    assert counts["invalid_box_size"] == 1
