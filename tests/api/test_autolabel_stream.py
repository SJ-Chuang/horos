"""E3-T3: batch autolabel as an R4 event stream + background job machinery."""

import threading
import time

import pytest
from helpers.data import make_image, write_sample_coco_dir
from helpers.fake_backend import FakeOpenVocabBackend

from horos.api import jobs
from horos.api.autolabel import PromptSpec, autolabel_events, start_autolabel
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.backends.base import dump_event, parse_event
from horos.errors import ProjectError

SPEC = PromptSpec(prompts={"forklift": ["forklift"], "person": ["person"]})


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    # two unannotated targets on top of the 3 annotated sample images
    for name in ("u1.png", "u2.png"):
        proj.add_image(make_image(tmp_path / name, 64, 48), width=64, height=48)
    return proj


def test_event_stream_shape(project):
    events = list(autolabel_events(project, SPEC, backend=FakeOpenVocabBackend()))
    assert events[0].type == "started"
    assert events[0].total == 2  # only the unannotated images by default
    types = [e.type for e in events[1:-1]]
    assert types.count("prediction") == 2 and types.count("progress") == 2
    assert events[-1].type == "completed"
    assert events[-1].result == {"images": 2, "annotations": 4}


def test_events_roundtrip_as_jsonl(project):
    for event in autolabel_events(project, SPEC, backend=FakeOpenVocabBackend()):
        line = dump_event(event)
        assert parse_event(line).type == event.type


def test_include_annotated(project):
    events = list(
        autolabel_events(
            project, SPEC, backend=FakeOpenVocabBackend(), only_unannotated=False
        )
    )
    assert events[0].total == 5


def test_cancel_between_images(project):
    cancel = threading.Event()
    backend = FakeOpenVocabBackend()
    stream = autolabel_events(project, SPEC, backend=backend, cancel=cancel)
    seen = [next(stream)]  # started
    seen.append(next(stream))  # first prediction
    cancel.set()
    seen.extend(stream)
    assert seen[-1].type == "completed"
    assert seen[-1].result["cancelled"] is True
    assert len(backend.calls) == 1  # second image never ran


def test_failure_terminates_with_failed_event(project):
    class Boom(FakeOpenVocabBackend):
        def infer_one(self, image, *, threshold=0.5):
            raise RuntimeError("boom")

    events = list(autolabel_events(project, SPEC, backend=Boom()))
    assert events[-1].type == "failed"
    assert "boom" in events[-1].message


def _wait_done(project, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = jobs.job_status(project, job_id)
        if status.state != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_background_job_lifecycle(project):
    job_id = start_autolabel(project, SPEC, backend=FakeOpenVocabBackend())
    status = _wait_done(project, job_id)
    assert status.state == "completed"
    assert status.events[0]["type"] == "started"
    assert status.events[-1]["type"] == "completed"
    # polling with `after` returns only the new events
    tail = jobs.job_status(project, job_id, after=status.num_events - 1)
    assert len(tail.events) == 1
    # the JSONL log survives for post-mortem inspection
    log = project.root / "jobs" / f"{job_id}.jsonl"
    assert log.exists()
    assert parse_event(log.read_text().splitlines()[-1]).type == "completed"


def test_single_concurrent_job(project):
    release = threading.Event()

    class Slow(FakeOpenVocabBackend):
        def infer_one(self, image, *, threshold=0.5):
            release.wait(5)
            return super().infer_one(image, threshold=threshold)

    job_id = start_autolabel(project, SPEC, backend=Slow())
    try:
        with pytest.raises(ProjectError, match="already running"):
            start_autolabel(project, SPEC, backend=FakeOpenVocabBackend())
    finally:
        release.set()
    _wait_done(project, job_id)


def test_cancel_running_job(project):
    step = threading.Event()

    class Slow(FakeOpenVocabBackend):
        def infer_one(self, image, *, threshold=0.5):
            step.wait(5)
            return super().infer_one(image, threshold=threshold)

    job_id = start_autolabel(project, SPEC, backend=Slow())
    assert jobs.cancel_job(project, job_id) is True
    step.set()
    status = _wait_done(project, job_id)
    assert status.state == "cancelled"
    assert jobs.cancel_job(project, job_id) is False  # already finished
