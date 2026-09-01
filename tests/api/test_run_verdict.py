"""Post-training verdict rules: every finished run gets a conclusion, and
near-perfect numbers are themselves a flag (bias/leakage check)."""

from __future__ import annotations

from horos.api.train import RunRecord
from horos.api.verdict import build_verdict

SPLITS_OK = {"train": 800, "valid": 100, "test": 100}


def _record(state="completed", **kw) -> RunRecord:
    defaults = dict(
        run_id="run-1",
        model="rfdetr-nano",
        state=state,
        created_at="2026-09-01T00:00:00+00:00",
        config={"epochs": 5},
    )
    defaults.update(kw)
    return RunRecord(**defaults)


def _metrics(step, **values):
    return {"type": "metrics", "step": step, "metrics": values}


def _healthy_events(epochs=5):
    events = []
    for e in range(epochs):
        events.append(_metrics(e, **{"train/loss": 10.0 - e * 2.0}))
        events.append(_metrics(e, **{"val/loss": 9.0 - e * 1.5, "val/mAP_50": 0.3 + e * 0.1}))
    return events


def _titles(verdict, severity=None):
    return [
        f.title for f in verdict.findings
        if severity is None or f.severity == severity
    ]


def test_clean_run_has_no_red_flags():
    verdict = build_verdict(_record(), _healthy_events(), SPLITS_OK)
    assert not _titles(verdict, "critical") and not _titles(verdict, "warning")
    assert "No red flags" in verdict.summary


def test_near_perfect_metrics_are_flagged_for_bias():
    events = _healthy_events()
    events.append(_metrics(5, **{"val/mAP_50": 1.0, "val/loss": 0.5}))
    events.append(_metrics(5, **{"train/loss": 0.4}))
    verdict = build_verdict(_record(), events, SPLITS_OK)
    flagged = [f for f in verdict.findings if "near-perfect" in f.title]
    assert flagged and flagged[0].severity == "warning"
    assert "leak" in flagged[0].suggestion


def test_tiny_valid_split_is_critical():
    verdict = build_verdict(
        _record(), _healthy_events(), {"train": 619, "valid": 2, "test": 14}
    )
    critical = [f for f in verdict.findings if f.severity == "critical"]
    assert critical and "far too small" in critical[0].title
    assert verdict.summary == critical[0].title


def test_small_valid_split_is_a_warning():
    verdict = build_verdict(
        _record(), _healthy_events(), {"train": 800, "valid": 20, "test": 80}
    )
    assert any("small" in t.lower() for t in _titles(verdict, "warning"))


def test_overfitting_is_detected():
    events = []
    val_losses = [9.0, 8.0, 8.6, 9.4, 10.2]  # minimum at epoch 1, then rises
    for e in range(5):
        events.append(_metrics(e, **{"train/loss": 10.0 - e * 1.8}))
        events.append(_metrics(e, **{"val/loss": val_losses[e], "val/mAP_50": 0.4}))
    verdict = build_verdict(_record(), events, SPLITS_OK)
    diverged = [f for f in verdict.findings if "diverged" in f.title]
    assert diverged and "epoch 1" in diverged[0].detail


def test_still_improving_is_noted():
    events = []
    for e in range(5):  # train loss falling >3% each epoch, val still falling
        events.append(_metrics(e, **{"train/loss": 10.0 * (0.8 ** e)}))
        events.append(_metrics(e, **{"val/loss": 9.0 - e, "val/mAP_50": 0.5}))
    verdict = build_verdict(_record(), events, SPLITS_OK)
    assert any("still improving" in t for t in _titles(verdict, "info"))


def test_no_learning_is_flagged():
    events = []
    for e in range(3):
        events.append(_metrics(e, **{"train/loss": 8.0, "val/loss": 8.0,
                                     "val/mAP_50": 0.01}))
    verdict = build_verdict(_record(), events, SPLITS_OK)
    assert any("barely detects" in t for t in _titles(verdict, "warning"))


def test_missing_test_split_is_noted():
    verdict = build_verdict(
        _record(), _healthy_events(), {"train": 900, "valid": 100, "test": 0}
    )
    noted = [f for f in verdict.findings if "test split" in f.title]
    assert noted and noted[0].severity == "info"


def test_failed_run_is_critical_with_log_pointer():
    verdict = build_verdict(_record(state="failed", error="CUDA OOM"), [], SPLITS_OK)
    critical = [f for f in verdict.findings if f.severity == "critical"]
    assert critical and "CUDA OOM" in critical[0].detail
    assert "worker.log" in critical[0].suggestion


def test_stopped_run_is_informational():
    verdict = build_verdict(_record(state="stopped"), _healthy_events(2), SPLITS_OK)
    assert any("Stopped" in t for t in _titles(verdict, "info"))


def test_findings_sorted_most_severe_first():
    verdict = build_verdict(
        _record(state="failed", error="boom"),
        _healthy_events(),
        {"train": 900, "valid": 100, "test": 0},
    )
    severities = [f.severity for f in verdict.findings]
    assert severities == sorted(
        severities, key={"critical": 0, "warning": 1, "info": 2}.get
    )


def test_run_verdict_reads_a_real_run_dir(tmp_path, monkeypatch):
    """Integration: verdict over an actual run produced by the fake backend."""
    import os
    import time
    from pathlib import Path

    from helpers.data import write_sample_coco_dir

    from horos.api import (
        create_project,
        import_dataset,
        run_verdict,
        start_training,
        training_status,
    )
    from horos.api.train import TrainRunConfig

    tests_root = Path(__file__).parent.parent
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(tests_root) + (os.pathsep + existing if existing else "")
    )
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project,
        TrainRunConfig(entrypoint_override="helpers.fake_backend:FakeBackend",
                       epochs=2),
    )
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)

    verdict = run_verdict(project, record.run_id)
    assert verdict.state == "completed" and verdict.summary
    # the sample dataset has a tiny valid split — the verdict must say so
    assert any(f.severity == "critical" for f in verdict.findings)


def test_overfit_suppresses_train_longer_advice():
    """A falling train loss during divergence is the overfit, not headroom."""
    events = []
    val_losses = [9.0, 8.0, 9.0, 11.0, 14.0]
    for e in range(5):
        events.append(_metrics(e, **{"train/loss": 10.0 * (0.8 ** e)}))
        events.append(_metrics(e, **{"val/loss": val_losses[e], "val/mAP_50": 0.3}))
    verdict = build_verdict(_record(), events, SPLITS_OK)
    assert any("diverged" in t for t in _titles(verdict))
    assert not any("still improving" in t for t in _titles(verdict))
