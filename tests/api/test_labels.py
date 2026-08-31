"""E2-T4: category management (add, rename, recolor, delete)."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.labels import add_category, delete_category, update_category
from horos.api.project import create_project, open_project
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def test_add_category(project):
    cat = add_category(project, "person", color="#ff0000")
    assert cat.name == "person" and cat.color == "#ff0000"
    reopened = open_project(project.root)
    assert "person" in {c.name for c in reopened.categories}


def test_add_assigns_fresh_id_and_default_color(project):
    existing_ids = {c.id for c in project.categories}
    cat = add_category(project, "person")
    assert cat.id not in existing_ids
    assert cat.color.startswith("#")


def test_duplicate_name_is_rejected(project):
    with pytest.raises(ProjectError, match="already exists"):
        add_category(project, "forklift")


def test_empty_name_is_rejected(project):
    with pytest.raises(ProjectError, match="not be empty"):
        add_category(project, "   ")


def test_rename(project):
    target = next(c for c in project.categories if c.name == "forklift")
    updated = update_category(project, target.id, name="lift_truck")
    assert updated.name == "lift_truck"
    assert {c.name for c in open_project(project.root).categories} == {
        "lift_truck",
        "pallet",
    }


def test_rename_to_taken_name_is_rejected(project):
    target = next(c for c in project.categories if c.name == "forklift")
    with pytest.raises(ProjectError, match="already exists"):
        update_category(project, target.id, name="pallet")


def test_recolor_keeps_name(project):
    target = project.categories[0]
    updated = update_category(project, target.id, color="#123456")
    assert updated.color == "#123456" and updated.name == target.name


def test_delete_unreferenced(project):
    cat = add_category(project, "person")
    assert delete_category(project, cat.id) == 0
    assert cat.id not in {c.id for c in open_project(project.root).categories}


def test_delete_referenced_is_refused(project):
    target = next(c for c in project.categories if c.name == "forklift")
    with pytest.raises(ProjectError, match="force=True"):
        delete_category(project, target.id)


def test_forced_delete_cascades_and_bumps_versions(project):
    target = next(c for c in project.categories if c.name == "forklift")
    affected = [
        r.id
        for r in project.list_images()
        if any(
            a.category_id == target.id
            for a in project.load_annotations(r.id).annotations
        )
    ]
    versions_before = {i: project.load_annotations(i).version for i in affected}
    deleted = delete_category(project, target.id, force=True)
    assert deleted == 2
    for image_id in affected:
        stored = project.load_annotations(image_id)
        assert all(a.category_id != target.id for a in stored.annotations)
        assert stored.version == versions_before[image_id] + 1


def test_unknown_id_is_explicit(project):
    with pytest.raises(ProjectError, match="No category"):
        update_category(project, 999, name="x")
