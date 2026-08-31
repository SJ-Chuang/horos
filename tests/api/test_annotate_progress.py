"""E2-T6: per-action persistence and resume-where-you-left-off."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.annotate import annotation_progress, image_queue, save_annotations
from horos.api.dataset import import_dataset
from horos.api.project import create_project, open_project


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def test_progress_counters(project, tmp_path):
    from helpers.data import make_image

    # the sample dataset annotates all 3 images; add one unannotated
    project.add_image(make_image(tmp_path / "new.png", 64, 48), width=64, height=48)
    progress = annotation_progress(project)
    assert progress.total_images == 4
    assert progress.annotated_images == 3
    assert progress.unannotated_images == 1
    assert progress.total_annotations == 4


def test_each_action_is_immediately_durable(project, tmp_path):
    from helpers.data import make_image

    record = project.add_image(
        make_image(tmp_path / "new.png", 64, 48), width=64, height=48
    )
    cat = project.categories[0]
    save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "bbox": (1, 1, 5, 5)}],
        expected_version=0,
    )
    # a brand-new Project object (fresh browser / new session) sees the write
    resumed = open_project(project.root)
    assert annotation_progress(resumed).annotated_images == 4


def test_queue_resumes_at_first_unannotated(project, tmp_path):
    from helpers.data import make_image

    project.add_image(make_image(tmp_path / "new.png", 64, 48), width=64, height=48)
    queue = image_queue(project)
    head = queue[0]
    assert not head.annotated
    # annotate it; the queue moves on to nothing-left-unannotated
    cat = project.categories[0]
    save_annotations(
        project,
        head.image.id,
        [{"category_id": cat.id, "bbox": (1, 1, 4, 4)}],
        expected_version=0,
    )
    requeued = image_queue(project)
    assert all(i.annotated for i in requeued)
    # annotated images sort behind while in unannotated_first mode
    assert requeued[0].image.file_name <= requeued[-1].image.file_name
