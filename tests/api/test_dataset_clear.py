"""Dataset page "delete all": clear_dataset empties images and annotations,
keeps classes unless asked, and refuses without the project-name confirm."""

from __future__ import annotations

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.annotate import claim_image
from horos.api.dataset import clear_dataset, import_dataset
from horos.api.project import create_project, open_project
from horos.cli import main
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj", name="demo")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def test_clear_removes_images_and_annotations_but_keeps_classes(project):
    files = [project.image_path(r) for r in project.list_images()]
    assert all(f.is_file() for f in files)
    summary = clear_dataset(project, confirm="demo")
    assert (summary.deleted_images, summary.deleted_annotations) == (3, 4)
    assert summary.deleted_categories == 0 and summary.skipped_claimed == []
    reopened = open_project(project.root)
    assert reopened.list_images() == []
    assert [c.name for c in reopened.categories] == ["forklift", "pallet"]
    assert not any(f.exists() for f in files)
    assert not any(project.annotations_dir.glob("*.json"))
    assert project.runs_dir.is_dir()  # runs are never touched


def test_clear_can_drop_the_class_list_too(project):
    summary = clear_dataset(project, confirm="demo", keep_categories=False)
    assert summary.deleted_categories == 2
    assert open_project(project.root).categories == []


def test_wrong_confirmation_deletes_nothing(project):
    with pytest.raises(ProjectError, match="confirm must equal the project name"):
        clear_dataset(project, confirm="Demo")
    with pytest.raises(ProjectError):
        clear_dataset(project, confirm="")
    assert len(project.list_images()) == 3


def test_images_claimed_by_another_session_survive_and_classes_stay(project):
    claim_image(project, 1, "other-session")
    summary = clear_dataset(project, confirm="demo", keep_categories=False, session_id="me")
    assert summary.skipped_claimed == [1] and summary.deleted_images == 2
    assert summary.deleted_annotations == 2  # image 1 kept its two boxes
    assert summary.deleted_categories == 0  # a kept image still needs its classes
    assert [r.id for r in project.list_images()] == [1]


def test_cli_clear_requires_yes(project, capsys):
    assert main(["clear", "--project", str(project.root)]) != 0
    assert "--yes" in capsys.readouterr().err
    assert len(project.list_images()) == 3
    assert main(["clear", "--project", str(project.root), "--yes", "--drop-classes"]) == 0
    import json

    body = json.loads(capsys.readouterr().out)
    assert body["deleted_images"] == 3 and body["deleted_categories"] == 2
