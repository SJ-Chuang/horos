"""E2-T8: optimistic locking + soft-claims. Cross-platform by construction —
the lock is a version check over an atomic rename, no fcntl anywhere (R7)."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.annotate import claim_image, release_claim, save_annotations
from horos.api.dataset import import_dataset
from horos.api.project import create_project, open_project
from horos.errors import AnnotationConflictError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def test_second_writer_gets_conflict(project):
    record = project.list_images()[0]
    cat = project.categories[0]
    base = project.load_annotations(record.id).version
    # two sessions load the same version, both edit
    session_a = open_project(project.root)
    session_b = open_project(project.root)
    save_annotations(
        session_a,
        record.id,
        [{"category_id": cat.id, "bbox": (1, 1, 2, 2)}],
        expected_version=base,
    )
    with pytest.raises(AnnotationConflictError, match="another session"):
        save_annotations(
            session_b,
            record.id,
            [{"category_id": cat.id, "bbox": (5, 5, 6, 6)}],
            expected_version=base,
        )
    # the losing session re-bases on the current version and succeeds
    current = session_b.load_annotations(record.id)
    saved = save_annotations(
        session_b,
        record.id,
        list(current.annotations) + [{"category_id": cat.id, "bbox": (5, 5, 6, 6)}],
        expected_version=current.version,
    )
    assert len(saved.annotations) == 2


def test_claim_denied_while_held(project):
    target = project.list_images()[0].id
    assert claim_image(project, target, "session-a").granted
    denial = claim_image(project, target, "session-b")
    assert not denial.granted and denial.held_by == "session-a"


def test_claim_renewal_and_release(project):
    target = project.list_images()[0].id
    first = claim_image(project, target, "session-a")
    renewed = claim_image(project, target, "session-a")
    assert renewed.granted and renewed.expires_at >= first.expires_at
    # only the holder can release
    assert not release_claim(project, target, "session-b")
    assert release_claim(project, target, "session-a")
    assert claim_image(project, target, "session-b").granted


def test_expired_claim_stops_steering(project):
    target = project.list_images()[0].id
    claim_image(project, target, "session-a", ttl_seconds=0.0)
    # expired immediately: another session may claim
    assert claim_image(project, target, "session-b").granted


def test_claim_is_advisory_not_a_write_lock(project):
    # correctness comes from the version check; a claim never blocks a write
    record = project.list_images()[0]
    cat = project.categories[0]
    claim_image(project, record.id, "session-a")
    version = project.load_annotations(record.id).version
    saved = save_annotations(
        project,
        record.id,
        [{"category_id": cat.id, "bbox": (1, 1, 2, 2)}],
        expected_version=version,
    )
    assert saved.version == version + 1
