"""E2-T2: bbox create/edit/delete through the annotation API."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.annotate import get_annotations, save_annotations
from horos.api.dataset import import_dataset
from horos.api.project import create_project, open_project
from horos.errors import DatasetValidationError, ProjectError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _fresh_image(project, tmp_path):
    from helpers.data import make_image

    src = make_image(tmp_path / "fresh.png", 64, 48)
    return project.add_image(src, width=64, height=48, split="train")


def test_add_bbox(project, tmp_path):
    record = _fresh_image(project, tmp_path)
    cat = project.categories[0]
    view = get_annotations(project, record.id)
    saved = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "bbox": (10, 10, 20, 15)}],
        expected_version=view.version,
    )
    assert saved.version == view.version + 1
    assert saved.annotations[0].bbox == (10, 10, 20, 15)


def test_edit_and_delete_flow(project, tmp_path):
    record = _fresh_image(project, tmp_path)
    cat = project.categories[0]
    v0 = get_annotations(project, record.id).version
    v1 = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "bbox": (10, 10, 20, 15)}],
        expected_version=v0,
    )
    # move/resize = replace the set with the edited box
    v2 = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "bbox": (12, 8, 25, 18)}],
        expected_version=v1.version,
    )
    assert v2.annotations[0].bbox == (12, 8, 25, 18)
    # delete = save the set without it
    v3 = save_annotations(project, record.id, [], expected_version=v2.version)
    assert v3.annotations == [] and v3.version == v2.version + 1


def test_persists_across_reopen(project, tmp_path):
    record = _fresh_image(project, tmp_path)
    cat = project.categories[0]
    save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "bbox": (1, 2, 3, 4)}],
        expected_version=0,
    )
    reopened = open_project(project.root)
    assert get_annotations(reopened, record.id).annotations[0].bbox == (1, 2, 3, 4)


def test_degenerate_bbox_is_rejected(project, tmp_path):
    record = _fresh_image(project, tmp_path)
    cat = project.categories[0]
    with pytest.raises(DatasetValidationError, match="degenerate box"):
        save_annotations(
            project,
            record.id,
            [{"category_id": cat.id, "bbox": (10, 10, 0, 15)}],
            expected_version=0,
        )


def test_unknown_category_is_rejected(project, tmp_path):
    record = _fresh_image(project, tmp_path)
    with pytest.raises(DatasetValidationError, match="unknown category"):
        save_annotations(
            project,
            record.id,
            [{"category_id": 999, "bbox": (1, 1, 2, 2)}],
            expected_version=0,
        )


def test_unknown_image_is_explicit(project):
    with pytest.raises(ProjectError, match="No image"):
        get_annotations(project, 999)
