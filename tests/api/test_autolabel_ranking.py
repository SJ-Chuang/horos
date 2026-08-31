"""E3-T6: uncertainty ranking — least-confident pre-labels first."""

import pytest
from helpers.data import make_image
from helpers.fake_backend import FakeOpenVocabBackend

from horos.api.annotate import image_queue
from horos.api.autolabel import PromptSpec, autolabel_events, pending_summary
from horos.api.project import create_project

SPEC = PromptSpec(prompts={"forklift": ["forklift"]})


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    for name in ("sure.png", "middling.png", "shaky.png"):
        proj.add_image(make_image(tmp_path / name, 64, 48), width=64, height=48)
    backend = FakeOpenVocabBackend(
        score_by_name={"sure.png": 0.95, "middling.png": 0.6, "shaky.png": 0.2}
    )
    events = list(autolabel_events(proj, SPEC, backend=backend))
    assert events[-1].type == "completed"
    return proj


def test_least_confident_first(project):
    ranking = pending_summary(project)
    assert [s.file_name for s in ranking] == ["shaky.png", "middling.png", "sure.png"]
    assert ranking[0].mean_score == pytest.approx(0.2)
    assert ranking[0].num_pending == 1


def test_queue_pending_mode_matches_ranking(project):
    queue = image_queue(project, mode="pending")
    assert [i.image.file_name for i in queue] == [
        "shaky.png", "middling.png", "sure.png",
    ]
    assert all(i.num_pending == 1 for i in queue)
    # pending pre-labels do not count as "annotated"
    assert all(not i.annotated for i in queue)


def test_images_without_pendings_are_excluded(project):
    from horos.api.autolabel import review_pending

    target = pending_summary(project)[0]
    review_pending(project, target.image_id, "accept")
    assert target.image_id not in {s.image_id for s in pending_summary(project)}
    assert len(image_queue(project, mode="pending")) == 2
