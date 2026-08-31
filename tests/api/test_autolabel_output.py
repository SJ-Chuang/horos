"""Autolabel output modes: bbox (native) vs polygon (SAM-refined boxes)."""

import pytest
from helpers.data import make_image
from helpers.fake_backend import FakeOpenVocabBackend, FakeRefinerBackend

from horos.api.autolabel import PromptSpec, assist_image, autolabel_events
from horos.api.project import create_project
from horos.errors import ProjectError

SPEC = PromptSpec(prompts={"forklift": ["forklift"], "person": ["person"]})


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    proj.add_image(make_image(tmp_path / "a.png", 64, 48), width=64, height=48)
    return proj


def _pendings(project):
    record = project.list_images()[0]
    return [
        a for a in project.load_annotations(record.id).annotations
        if a.status == "pending"
    ]


def test_bbox_mode_writes_no_segmentation(project):
    events = list(autolabel_events(project, SPEC, backend=FakeOpenVocabBackend()))
    assert events[-1].type == "completed"
    assert all(a.segmentation == [] for a in _pendings(project))


def test_polygon_mode_attaches_refined_outline(project):
    refiner = FakeRefinerBackend()
    events = list(
        autolabel_events(
            project, SPEC, backend=FakeOpenVocabBackend(),
            output="polygon", refiner=refiner,
        )
    )
    assert events[-1].type == "completed"
    pendings = _pendings(project)
    assert len(pendings) == 2
    for ann in pendings:
        assert len(ann.segmentation) == 1
        assert len(ann.segmentation[0]) == 6  # fake refiner emits triangles
        # detector box is kept alongside the mask outline
        assert ann.bbox[2] > 0 and ann.bbox[3] > 0
    assert refiner.calls == [("a.png", 2)]
    assert events[0].config["output"] == "polygon"


def test_failed_mask_keeps_the_box(project):
    refiner = FakeRefinerBackend(fail_indices={0})
    list(
        autolabel_events(
            project, SPEC, backend=FakeOpenVocabBackend(),
            output="polygon", refiner=refiner,
        )
    )
    pendings = sorted(_pendings(project), key=lambda a: a.id)
    with_poly = [a for a in pendings if a.segmentation]
    without = [a for a in pendings if not a.segmentation]
    assert len(with_poly) == 1 and len(without) == 1  # box survived, not dropped


def test_invalid_output_is_explicit(project):
    with pytest.raises(ProjectError, match="output"):
        list(autolabel_events(project, SPEC, backend=FakeOpenVocabBackend(), output="mask"))


def test_assist_polygon_mode(project):
    record = project.list_images()[0]
    result = assist_image(
        project, record.id, SPEC,
        backend=FakeOpenVocabBackend(), refiner=FakeRefinerBackend(),
        output="polygon",
    )
    pendings = [a for a in result.annotations if a.status == "pending"]
    assert pendings and all(a.segmentation for a in pendings)
