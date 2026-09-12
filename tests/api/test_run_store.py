"""E7-T1: run metadata storage — the experiment.json sidecar next to run.json."""

from __future__ import annotations

import json

import pytest
from helpers.runs import completed_fake_run

from horos.api.evaluate import EvalReport
from horos.api.experiment import RunExtras, get_run_summary, read_extras, write_extras
from horos.api.train import read_record, write_record
from horos.errors import ProjectError


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return completed_fake_run(tmp_path_factory.mktemp("store"), epochs=3)


def _run_dir(project, record):
    return project.root / "runs" / record.run_id


def test_summary_joins_record_scores_and_fingerprint(run):
    project, record = run
    summary = get_run_summary(project, record.run_id)
    assert summary.run.run_id == record.run_id and summary.run.state == "completed"
    # scores come from the same run_scores() the report and `horos models` use
    assert summary.scores["loss"] == pytest.approx(1 / 3)
    assert summary.fingerprint == record.dataset_fingerprint
    assert summary.notes == "" and summary.tags == [] and summary.evals == {}


def test_terminal_scores_are_cached_in_the_sidecar(run):
    project, record = run
    run_dir = _run_dir(project, record)
    get_run_summary(project, record.run_id)
    sidecar = json.loads((run_dir / "experiment.json").read_text("utf-8"))
    assert sidecar["scores_state"] == "completed"
    assert sidecar["scores"]["loss"] == pytest.approx(1 / 3)

    # the cache is trusted while the state is unchanged ...
    events = run_dir / "events.jsonl"
    original = events.read_text("utf-8")
    events.write_text("", "utf-8")
    try:
        assert get_run_summary(project, record.run_id).scores["loss"] == pytest.approx(1 / 3)
    finally:
        events.write_text(original, "utf-8")

    # ... and dropped when the state moves on (a stopped run re-reads events)
    stopped = read_record(run_dir).model_copy(update={"state": "stopped", "pid": None})
    write_record(run_dir, stopped)
    try:
        events.write_text(
            original + json.dumps({"type": "metrics", "step": 4, "metrics": {"loss": 0.01}})
            + "\n", "utf-8",
        )
        summary = get_run_summary(project, record.run_id)
        assert summary.run.state == "stopped"
        assert summary.scores["loss"] == pytest.approx(0.01)
    finally:
        events.write_text(original, "utf-8")
        write_record(run_dir, record)
        write_extras(run_dir, RunExtras())


def test_old_runs_get_their_fingerprint_backfilled_once(run):
    project, record = run
    run_dir = _run_dir(project, record)
    legacy = record.model_copy(update={"dataset_fingerprint": None})
    write_record(run_dir, legacy)
    write_extras(run_dir, RunExtras())
    try:
        summary = get_run_summary(project, record.run_id)
        assert summary.fingerprint is not None
        assert summary.fingerprint.digest == record.dataset_fingerprint.digest
        # remembered in the sidecar so the snapshot is hashed only once
        assert read_extras(run_dir).fingerprint == summary.fingerprint
    finally:
        write_record(run_dir, record)
        write_extras(run_dir, RunExtras())


def test_sidecar_never_touches_run_json(run):
    project, record = run
    run_dir = _run_dir(project, record)
    before = (run_dir / "run.json").read_bytes()
    write_extras(run_dir, RunExtras(notes="baseline", tags=["a"]))
    get_run_summary(project, record.run_id)
    assert (run_dir / "run.json").read_bytes() == before
    write_extras(run_dir, RunExtras())


def test_corrupt_sidecar_does_not_hide_the_run(run):
    project, record = run
    run_dir = _run_dir(project, record)
    (run_dir / "experiment.json").write_text("{not json", "utf-8")
    summary = get_run_summary(project, record.run_id)
    assert summary.run.run_id == record.run_id and summary.notes == ""
    write_extras(run_dir, RunExtras())


def test_persisted_evaluations_surface_as_headline_metrics(run):
    project, record = run
    run_dir = _run_dir(project, record)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    report = EvalReport(
        run_id=record.run_id, split="test", created_at="2026-09-12T00:00:00+00:00",
        num_images=1, num_instances=1, map_5095=0.5, map_50=0.75, map_75=0.4,
        mar_100=0.6,
    )
    (eval_dir / "test.json").write_text(report.model_dump_json(), "utf-8")
    (eval_dir / "test.detections.json").write_text("{}", "utf-8")
    try:
        summary = get_run_summary(project, record.run_id)
        assert summary.evals == {
            "test": {"map_5095": 0.5, "map_50": 0.75, "map_75": 0.4, "mar_100": 0.6}
        }
    finally:
        (eval_dir / "test.json").unlink()
        (eval_dir / "test.detections.json").unlink()


def test_unknown_run_is_an_explicit_error(run):
    project, _ = run
    with pytest.raises(ProjectError, match="No such training run"):
        get_run_summary(project, "nope")
