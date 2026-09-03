"""dataset.validate_fix: clamp auto-fixable out-of-bounds boxes (E1-S4).

The fixer must repair exactly what the report marks `fixable` — sub-pixel /
few-pixel annotation-tool overshoots — and leave genuinely broken boxes for a
human.
"""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import fix_validation_issues, import_dataset, validate_project
from horos.api.project import create_project


@pytest.fixture
def project(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    return project


def _set_bbox(project, image_id, bbox):
    """Give the image's first annotation the bbox; returns the annotation id."""
    stored = project.load_annotations(image_id)
    target = stored.annotations[0]
    updated = [
        a.model_copy(update={"bbox": bbox}) if a.id == target.id else a
        for a in stored.annotations
    ]
    project.save_annotations(image_id, updated, expected_version=stored.version)
    return target.id


def test_fix_clamps_jitter_and_clears_the_report(project):
    record = project.list_images()[0]  # 64x48
    ann_id = _set_bbox(project, record.id, (0.5, -0.4, 64.0, 12.0))

    result = fix_validation_issues(project)

    assert result.num_fixed == 1
    fixed = result.fixed[0]
    assert (fixed.image_id, fixed.annotation_id) == (record.id, ann_id)
    assert fixed.before == (0.5, -0.4, 64.0, 12.0)
    x, y, w, h = fixed.after
    assert x >= 0 and y >= 0 and x + w <= 64 and y + h <= 48
    assert w > 0 and h > 0
    # persisted, not just reported
    saved = next(
        a for a in project.load_annotations(record.id).annotations if a.id == ann_id
    )
    assert saved.bbox == fixed.after
    assert result.report.ok
    assert not any(i.kind == "bbox_out_of_bounds" for i in result.report.issues)


def test_fix_leaves_genuinely_broken_boxes_alone(project):
    record = project.list_images()[0]
    ann_id = _set_bbox(project, record.id, (60.0, 40.0, 20.0, 20.0))  # 16px out

    result = fix_validation_issues(project)

    assert result.num_fixed == 0 and result.fixed == []
    saved = next(
        a for a in project.load_annotations(record.id).annotations if a.id == ann_id
    )
    assert saved.bbox == (60.0, 40.0, 20.0, 20.0)
    assert not result.report.ok  # still an error for a human to look at


def test_fix_repairs_exactly_the_fixable_issues(project):
    images = project.list_images()
    _set_bbox(project, images[0].id, (0.2, 4.0, 64.0, 12.0))  # jitter -> fixable
    _set_bbox(project, images[1].id, (60.0, 40.0, 20.0, 20.0))  # broken -> not

    report = validate_project(project)
    fixable = [i for i in report.issues if i.fixable]
    assert len(fixable) == 1

    result = fix_validation_issues(project)
    assert result.num_fixed == len(fixable)
    assert {(f.image_id, f.annotation_id) for f in result.fixed} == {
        (i.image_id, i.annotation_id) for i in fixable
    }


def test_fix_is_idempotent(project):
    record = project.list_images()[0]
    _set_bbox(project, record.id, (0.5, 4.0, 64.0, 12.0))
    assert fix_validation_issues(project).num_fixed == 1
    assert fix_validation_issues(project).num_fixed == 0
