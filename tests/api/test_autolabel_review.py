"""E3-T5: pre-labels land as pending, distinguishable from manual work, and the
review actions (accept / reject / accept-at-threshold) resolve them."""

import pytest
from helpers.data import make_image, write_sample_coco_dir
from helpers.fake_backend import FakeOpenVocabBackend

from horos.api.autolabel import (
    PromptSpec,
    assist_image,
    autolabel_events,
    review_pending,
)
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.errors import ProjectError

SPEC = PromptSpec(prompts={"forklift": ["forklift"], "person": ["person"]})


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    proj.add_image(make_image(tmp_path / "u1.png", 64, 48), width=64, height=48)
    return proj


def _run(project, **kw):
    events = list(autolabel_events(project, SPEC, backend=FakeOpenVocabBackend(), **kw))
    assert events[-1].type == "completed", events[-1]
    return events


def _target(project):
    return next(
        r for r in project.list_images()
        if any(a.status == "pending" for a in project.load_annotations(r.id).annotations)
    )


def test_prelabels_are_pending_and_auto(project):
    _run(project)
    record = _target(project)
    pendings = [
        a for a in project.load_annotations(record.id).annotations
        if a.status == "pending"
    ]
    assert len(pendings) == 2
    assert all(a.source == "auto" and a.score is not None for a in pendings)


def test_new_classes_are_created_for_prompts(project):
    _run(project)
    assert "person" in {c.name for c in project.categories}


def test_rerun_replaces_pendings_not_confirmed(project):
    _run(project, only_unannotated=False)
    record = project.list_images()[0]  # an image with confirmed annotations
    before = project.load_annotations(record.id)
    confirmed = [a for a in before.annotations if a.status == "confirmed"]
    _run(project, only_unannotated=False)  # second pass
    after = project.load_annotations(record.id)
    assert [a for a in after.annotations if a.status == "confirmed"] == confirmed
    assert sum(a.status == "pending" for a in after.annotations) == 2  # not 4


def test_accept_all(project):
    _run(project)
    record = _target(project)
    assert review_pending(project, record.id, "accept") == 2
    stored = project.load_annotations(record.id)
    assert all(a.status == "confirmed" for a in stored.annotations)
    # accepted pre-labels keep their provenance (E3-T5: distinguishable)
    assert any(a.source == "auto" for a in stored.annotations)


def test_reject_all(project):
    _run(project)
    record = _target(project)
    assert review_pending(project, record.id, "reject") == 2
    assert all(
        a.status != "pending"
        for a in project.load_annotations(record.id).annotations
    )


def test_accept_at_threshold_drops_the_rest(project):
    # fake scores are 0.9 (forklift) and 0.8 (person)
    _run(project)
    record = _target(project)
    assert review_pending(project, record.id, "accept", min_score=0.85) == 1
    stored = project.load_annotations(record.id)
    assert sum(a.status == "confirmed" for a in stored.annotations) == 1
    assert sum(a.status == "pending" for a in stored.annotations) == 0


def test_accept_by_ids(project):
    _run(project)
    record = _target(project)
    pendings = [
        a for a in project.load_annotations(record.id).annotations
        if a.status == "pending"
    ]
    assert review_pending(project, record.id, "accept", ann_ids=[pendings[0].id]) == 1
    stored = project.load_annotations(record.id)
    assert sum(a.status == "pending" for a in stored.annotations) == 1


def test_bad_action_is_explicit(project):
    with pytest.raises(ProjectError, match="accept"):
        review_pending(project, 1, "merge")


def test_assist_single_image(project):
    record = project.list_images()[0]
    result = assist_image(project, record.id, SPEC, backend=FakeOpenVocabBackend())
    assert result.image_id == record.id
    assert sum(a.status == "pending" for a in result.annotations) == 2
    # confirmed annotations on the image were preserved
    assert any(a.status == "confirmed" for a in result.annotations)


def test_pending_writes_are_clamped_to_image_bounds(tmp_path):
    """OWLv2 boxes routinely overshoot the frame (the balloon-det warnings):
    pending annotations must land clipped; fully-outside detections drop."""
    from helpers.data import make_image

    from horos.api.autolabel import _write_pending
    from horos.api.project import create_project

    project = create_project(tmp_path / "proj")
    project.add_image(make_image(tmp_path / "img.png", 100, 80), width=100, height=80)
    image = project.list_images()[0]
    cat_ids = {"person": 1}
    from horos.core.dataset import Category

    project.set_categories([Category(id=1, name="person", color="#fff")])

    written = _write_pending(
        project,
        image.id,
        [
            ("person", (-2.5, 10.0, 50.0, 60.0), 0.9),    # clips at x=0
            ("person", (60.0, 70.0, 50.0, 30.0), 0.8),    # clips at right/bottom
            ("person", (200.0, 10.0, 20.0, 20.0), 0.7),   # fully outside → drop
        ],
        cat_ids,
        polygons=[[-3.0, 12.0, 40.0, 8.0, 30.0, 90.0], None, None],
    )
    assert written == 2
    stored = project.load_annotations(image.id).annotations
    for ann in stored:
        x, y, w, h = ann.bbox
        assert x >= 0 and y >= 0 and x + w <= 100 and y + h <= 80
        for poly in ann.segmentation:
            assert all(0 <= v <= 100 for v in poly[0::2])
            assert all(0 <= v <= 80 for v in poly[1::2])


def test_manual_save_clamps_out_of_bounds_input(tmp_path):
    from helpers.data import make_image

    from horos.api.annotate import get_annotations, save_annotations
    from horos.api.project import create_project
    from horos.core.dataset import Category

    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="x", color="#fff")])
    project.add_image(make_image(tmp_path / "img.png", 100, 80), width=100, height=80)
    image = project.list_images()[0]
    view = get_annotations(project, image.id)
    saved = save_annotations(
        project, image.id,
        [{"category_id": 1, "bbox": (-5.0, -5.0, 50.0, 100.0)}],
        expected_version=view.version,
    )
    assert saved.annotations[0].bbox == (0.0, 0.0, 45.0, 80.0)
