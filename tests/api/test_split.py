"""E1-T8: split management — keep existing splits, or re-split with a seed."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset, resplit
from horos.api.project import create_project
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    return project


def test_import_preserves_existing_splits(project):
    splits = {i.file_name: i.split for i in project.list_images()}
    assert sorted(splits.values()) == ["train", "train", "valid"]


def test_resplit_is_deterministic_under_seed(project):
    counts1 = resplit(project, train=0.4, valid=0.3, test=0.3, seed=7)
    first = {i.id: i.split for i in project.list_images()}
    counts2 = resplit(project, train=0.4, valid=0.3, test=0.3, seed=7)
    second = {i.id: i.split for i in project.list_images()}
    assert first == second
    assert counts1 == counts2


def test_resplit_ratio_shapes_assignment(project):
    counts = resplit(project, train=1.0, valid=0.0, test=0.0, seed=1)
    assert counts == {"train": 3, "valid": 0, "test": 0}
    assert all(i.split == "train" for i in project.list_images())


def test_resplit_rejects_bad_ratios(project):
    with pytest.raises(ProjectError, match="sum to 1.0"):
        resplit(project, train=0.9, valid=0.9, test=0.1)


def test_resplit_rejects_empty_project(tmp_path):
    project = create_project(tmp_path / "empty_proj")
    with pytest.raises(ProjectError, match="no images"):
        resplit(project)


def test_resplit_uses_attributes_not_symlinks(project):
    # R7: no symlinks anywhere in split handling
    resplit(project, seed=3)
    links = [p for p in project.root.rglob("*") if p.is_symlink()]
    assert links == []
