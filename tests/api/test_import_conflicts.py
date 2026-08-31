"""File-name conflict handling on import: hash-based duplicate skip plus the
ask/overwrite/skip/rename policies (design decision: two-phase confirm)."""

import pytest
from helpers.data import make_image, write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.errors import ImportConflictError, ProjectError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _variant_coco_dir(tmp_path):
    """Same file names as the sample dir, but image 'a.png' has different pixels."""
    variant = write_sample_coco_dir(tmp_path / "coco2")
    make_image(variant / "train" / "a.png", 64, 48, color=(1, 2, 3))
    return variant


def test_reimport_identical_dataset_skips_everything(project, tmp_path):
    summary = import_dataset(project, write_sample_coco_dir(tmp_path / "same"))
    assert summary.num_images == 0
    assert summary.duplicates_skipped == 3
    assert summary.conflict_files == []
    assert len(project.list_images()) == 3


def test_ask_raises_with_conflict_list_and_writes_nothing(project, tmp_path):
    with pytest.raises(ImportConflictError) as exc:
        import_dataset(project, _variant_coco_dir(tmp_path))
    assert exc.value.conflicts == ["a.png"]
    assert exc.value.details == {"conflicts": ["a.png"]}
    # nothing was imported, not even the non-conflicting images
    assert len(project.list_images()) == 3


def test_overwrite_replaces_image_and_annotations(project, tmp_path):
    original = next(r for r in project.list_images() if r.file_name == "a.png")
    original_bytes = (project.images_dir / "a.png").read_bytes()
    summary = import_dataset(
        project, _variant_coco_dir(tmp_path), on_conflict="overwrite"
    )
    assert summary.overwritten == 1
    assert summary.duplicates_skipped == 2
    records = project.list_images()
    assert len(records) == 3
    replaced = next(r for r in records if r.file_name == "a.png")
    assert replaced.id == original.id  # record identity survives
    assert (project.images_dir / "a.png").read_bytes() != original_bytes


def test_skip_keeps_existing(project, tmp_path):
    original_bytes = (project.images_dir / "a.png").read_bytes()
    summary = import_dataset(project, _variant_coco_dir(tmp_path), on_conflict="skip")
    assert summary.conflicts_skipped == 1
    assert (project.images_dir / "a.png").read_bytes() == original_bytes
    assert len(project.list_images()) == 3


def test_rename_imports_both(project, tmp_path):
    summary = import_dataset(project, _variant_coco_dir(tmp_path), on_conflict="rename")
    assert summary.renamed == 1
    names = {r.file_name for r in project.list_images()}
    assert "a.png" in names and "a_1.png" in names


def test_bad_policy_is_explicit(project, tmp_path):
    with pytest.raises(ProjectError, match="on_conflict"):
        import_dataset(project, tmp_path, on_conflict="merge")
