"""dataset.delete_images: bulk image deletion behind the grid selection UI."""

import pytest
from helpers.data import make_image, write_sample_coco_dir

from horos.api.annotate import claim_image
from horos.api.dataset import dataset_stats, delete_images, import_dataset
from horos.api.project import create_project
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    return project


def test_delete_removes_index_annotations_and_file(project):
    record = project.list_images()[0]
    image_file = project.images_dir / record.file_name
    annotation_file = project.annotations_dir / f"{record.id}.json"
    assert image_file.exists() and annotation_file.exists()

    summary = delete_images(project, [record.id])

    assert summary.deleted == [record.id] and summary.skipped_claimed == []
    assert not image_file.exists()
    assert not annotation_file.exists()
    assert all(i.id != record.id for i in project.list_images())
    # the dataset snapshot no longer carries its annotations
    assert dataset_stats(project).num_images == 2


def test_externally_referenced_images_keep_their_source_file(project, tmp_path):
    source = make_image(tmp_path / "external.png")
    record = project.add_image(source, width=64, height=48, copy=False)

    summary = delete_images(project, [record.id])

    assert summary.deleted == [record.id]
    assert source.exists()  # only the reference is dropped, never the source


def test_images_claimed_by_another_session_are_skipped(project):
    first, second = project.list_images()[:2]
    claim_image(project, first.id, session_id="someone-else")

    summary = delete_images(project, [first.id, second.id], session_id="me")

    assert summary.deleted == [second.id]
    assert summary.skipped_claimed == [first.id]
    assert any(i.id == first.id for i in project.list_images())


def test_own_claim_does_not_block_deletion(project):
    record = project.list_images()[0]
    claim_image(project, record.id, session_id="me")
    summary = delete_images(project, [record.id], session_id="me")
    assert summary.deleted == [record.id]


def test_unknown_id_fails_before_deleting_anything(project):
    ids = [i.id for i in project.list_images()]
    with pytest.raises(ProjectError, match="999"):
        delete_images(project, [ids[0], 999])
    assert len(project.list_images()) == 3  # nothing was deleted
