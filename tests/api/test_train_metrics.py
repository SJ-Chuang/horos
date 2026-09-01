"""E5-T4: metric streaming and persistence — events survive on disk as typed,
replayable JSONL, and polling pagination never loses or duplicates one."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status
from horos.backends.base import parse_event

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )


@pytest.fixture
def finished_run(tmp_path):
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=3)
    )
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline, "run never finished"
        time.sleep(0.2)
    return project, record.run_id


def test_events_persist_as_typed_jsonl(finished_run):
    project, run_id = finished_run
    log = project.root / "runs" / run_id / "events.jsonl"
    assert log.is_file()
    lines = [line for line in log.read_text("utf-8").splitlines() if line.strip()]
    parsed = [parse_event(line) for line in lines]  # every line round-trips (R4)
    types = [e.type for e in parsed]
    assert types[0] == "started" and types[-1] == "completed"
    metrics = [e for e in parsed if e.type == "metrics"]
    assert len(metrics) == 3  # one per epoch from the fake backend
    assert all("loss" in e.metrics for e in metrics)
    assert [e.step for e in metrics] == [1, 2, 3]


def test_status_replays_exactly_what_is_on_disk(finished_run):
    project, run_id = finished_run
    log = project.root / "runs" / run_id / "events.jsonl"
    on_disk = [line for line in log.read_text("utf-8").splitlines() if line.strip()]
    status = training_status(project, run_id)
    assert status.num_events == len(on_disk)
    assert [e["type"] for e in status.events] == [
        parse_event(line).type for line in on_disk
    ]


def test_pagination_covers_the_stream_without_gaps_or_overlap(finished_run):
    project, run_id = finished_run
    total = training_status(project, run_id).num_events
    collected = []
    cursor = 0
    while cursor < total:
        page = training_status(project, run_id, after=cursor)
        chunk = page.events[: max(1, len(page.events) // 2)]  # small pages
        collected.extend(chunk)
        cursor += len(chunk)
    assert len(collected) == total
    full = training_status(project, run_id).events
    assert [e["type"] for e in collected] == [e["type"] for e in full]
