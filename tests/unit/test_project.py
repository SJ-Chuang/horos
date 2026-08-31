"""E1-T1: Project object and on-disk structure — create, load, validate."""

import pytest
from helpers.data import make_image

from horos.core.dataset import Annotation, Category
from horos.core.project import Project
from horos.errors import AnnotationConflictError, ProjectError


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / "proj", name="demo")


def test_create_builds_expected_structure(project):
    assert (project.root / "horos.json").exists()
    assert (project.root / "images.json").exists()
    assert project.images_dir.is_dir()
    assert project.annotations_dir.is_dir()
    assert project.runs_dir.is_dir()
    assert project.manifest.name == "demo"


def test_create_refuses_existing_project(project):
    with pytest.raises(ProjectError, match="already exists"):
        Project.create(project.root)


def test_create_refuses_nonempty_directory(tmp_path):
    (tmp_path / "junk.txt").write_text("x")
    with pytest.raises(ProjectError, match="non-empty"):
        Project.create(tmp_path)


def test_open_roundtrip(project):
    reopened = Project.open(project.root)
    assert reopened.manifest.name == "demo"


def test_open_missing_project_is_explicit(tmp_path):
    with pytest.raises(ProjectError, match="No horos project"):
        Project.open(tmp_path / "nope")


def test_open_corrupt_manifest_is_explicit(project):
    (project.root / "horos.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectError, match="Corrupt project manifest"):
        Project.open(project.root)


def test_open_detects_missing_directories(project):
    import shutil

    shutil.rmtree(project.annotations_dir)
    with pytest.raises(ProjectError, match="missing directories"):
        Project.open(project.root)


def test_categories_roundtrip_and_get_default_colors(project):
    project.set_categories([Category(id=1, name="forklift")])
    reopened = Project.open(project.root)
    assert reopened.categories[0].color  # default color assigned
    with pytest.raises(ProjectError, match="Duplicate category name"):
        project.set_categories(
            [Category(id=1, name="a"), Category(id=2, name="a")]
        )


def test_add_image_copies_by_default(project, tmp_path):
    src = make_image(tmp_path / "src" / "cat.png", 64, 48)
    record = project.add_image(src, width=64, height=48)
    assert (project.images_dir / record.file_name).exists()
    assert record.external_path is None
    assert project.image_path(record).parent == project.images_dir


def test_add_image_reference_mode_keeps_source_in_place(project, tmp_path):
    src = make_image(tmp_path / "src" / "cat.png", 64, 48)
    record = project.add_image(src, width=64, height=48, copy=False)
    assert not (project.images_dir / record.file_name).exists()
    assert project.image_path(record) == src.resolve()


def test_add_image_resolves_name_collisions(project, tmp_path):
    a = make_image(tmp_path / "one" / "cat.png")
    b = make_image(tmp_path / "two" / "cat.png")
    r1 = project.add_image(a, width=64, height=48)
    r2 = project.add_image(b, width=64, height=48)
    assert r1.file_name != r2.file_name
    assert r1.id != r2.id


def test_annotation_save_load_roundtrip(project, tmp_path):
    src = make_image(tmp_path / "cat.png")
    record = project.add_image(src, width=64, height=48)
    ann = Annotation(id=1, image_id=record.id, category_id=1, bbox=(1, 2, 3, 4))
    saved = project.save_annotations(record.id, [ann], expected_version=0)
    assert saved.version == 1
    loaded = project.load_annotations(record.id)
    assert loaded.version == 1
    assert loaded.annotations == [ann]


def test_annotation_optimistic_lock_conflict(project, tmp_path):
    # E2-T8 foundation: stale writers must get a conflict, not clobber.
    src = make_image(tmp_path / "cat.png")
    record = project.add_image(src, width=64, height=48)
    ann = Annotation(id=1, image_id=record.id, category_id=1, bbox=(1, 2, 3, 4))
    project.save_annotations(record.id, [ann], expected_version=0)
    with pytest.raises(AnnotationConflictError, match="another session"):
        project.save_annotations(record.id, [ann], expected_version=0)


def test_to_dataset_assembles_everything(project, tmp_path):
    project.set_categories([Category(id=1, name="forklift")])
    src = make_image(tmp_path / "cat.png")
    record = project.add_image(src, width=64, height=48)
    ann = Annotation(id=1, image_id=record.id, category_id=1, bbox=(1, 2, 3, 4))
    project.save_annotations(record.id, [ann], expected_version=0)
    ds = project.to_dataset()
    assert len(ds.images) == 1 and len(ds.annotations) == 1
    assert ds.categories[0].name == "forklift"
