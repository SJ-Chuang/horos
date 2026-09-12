"""Experiment management (E7): which run was best, and why.

Design decisions (confirmed 2026-09-12):

- User-owned run metadata (notes, tags) and derived caches live in a sidecar
  `<run>/experiment.json` written only by this module. The training worker
  rewrites `run.json` on every state change, so putting user fields there
  would race with it and silently lose one side's write.
- Scores are the mAP/mAR/F1/loss values at the best checkpoint (the same
  `run_scores` the report and `horos models` use). They are cached in the
  sidecar once a run is terminal so listing runs stops re-parsing every
  events.jsonl; an active run is always re-read.
- Comparability is decided by the dataset fingerprint (E7-T2): two runs are
  comparable when their fingerprints are identical; a run is compared to the
  project's current data by fingerprinting today's dataset under the run's
  own class scope.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.api.train import (
    RunRecord,
    _read_events,
    _reconcile,
    _run_dir,
    list_runs,
    read_record,
)
from horos.core.fingerprint import DatasetFingerprint, fingerprint_snapshot
from horos.core.project import Project

__all__ = [
    "RunExtras",
    "RunSummary",
    "get_run_summary",
]

_EXTRAS_JSON = "experiment.json"
TERMINAL_STATES = ("completed", "failed", "stopped")
#: headline numbers copied out of a persisted evaluation report (E6-T3)
_EVAL_KEYS = ("map_5095", "map_50", "map_75", "mar_100")


class RunExtras(BaseModel):
    """The sidecar: user annotations plus caches derived from run artifacts."""

    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    updated_at: str | None = None
    #: cached run_scores() output, valid while the run stays in scores_state
    scores: dict[str, float] | None = None
    best_epoch: int | None = None
    scores_state: str | None = None
    #: fingerprint backfilled from the snapshot for runs older than E7-T2
    fingerprint: DatasetFingerprint | None = None


class RunSummary(BaseModel):
    """One run as the experiment view sees it: record + scores + metadata."""

    run: RunRecord
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    #: 1-based epoch the backend reported as best (None until known)
    best_epoch: int | None = None
    #: metrics at the best checkpoint, e.g. {"map50": 0.71, "loss": 0.4}
    scores: dict[str, float] = Field(default_factory=dict)
    #: persisted evaluation headline metrics per split (E6), e.g.
    #: {"test": {"map_50": 0.68, ...}}
    evals: dict[str, dict[str, float]] = Field(default_factory=dict)
    fingerprint: DatasetFingerprint | None = None


# ------------------------------------------------------------------ sidecar


def _extras_path(run_dir: Path) -> Path:
    return run_dir / _EXTRAS_JSON


def read_extras(run_dir: Path) -> RunExtras:
    path = _extras_path(run_dir)
    if not path.is_file():
        return RunExtras()
    try:
        return RunExtras.model_validate_json(path.read_text("utf-8"))
    except ValueError:
        # a corrupt sidecar must never hide the run itself
        return RunExtras()


def write_extras(run_dir: Path, extras: RunExtras) -> None:
    """Atomic replace, like run.json (R7: os.replace is atomic everywhere)."""
    tmp = run_dir / f"{_EXTRAS_JSON}.tmp"
    tmp.write_text(extras.model_dump_json(indent=2), "utf-8")
    os.replace(tmp, _extras_path(run_dir))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ scores


def _scores(run_dir: Path, record: RunRecord, extras: RunExtras) -> tuple[
    int | None, dict[str, float], bool
]:
    """(best epoch 1-based, scores, cache_dirty). Terminal runs are read once
    and cached in the sidecar; anything still moving is re-read."""
    if (
        extras.scores is not None
        and extras.scores_state == record.state
        and record.state in TERMINAL_STATES
    ):
        return extras.best_epoch, dict(extras.scores), False
    from horos.api.report import _series_from_events, run_scores

    events, _ = _read_events(run_dir)
    best_epoch, scores = run_scores(_series_from_events(events))
    best_1based = None if best_epoch is None else best_epoch + 1
    dirty = record.state in TERMINAL_STATES
    if dirty:
        extras.scores = scores
        extras.best_epoch = best_1based
        extras.scores_state = record.state
    return best_1based, scores, dirty


def _evals(run_dir: Path) -> dict[str, dict[str, float]]:
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return {}
    import json

    out: dict[str, dict[str, float]] = {}
    for path in sorted(eval_dir.glob("*.json")):
        if path.name.endswith(".detections.json"):
            continue
        try:
            report = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out[path.stem] = {
            key: float(report[key])
            for key in _EVAL_KEYS
            if isinstance(report.get(key), int | float)
        }
    return out


def _fingerprint(run_dir: Path, record: RunRecord, extras: RunExtras) -> tuple[
    DatasetFingerprint | None, bool
]:
    """(fingerprint, cache_dirty): the recorded one, else a snapshot backfill
    remembered in the sidecar so old runs are hashed only once."""
    if record.dataset_fingerprint is not None:
        return record.dataset_fingerprint, False
    if extras.fingerprint is not None:
        return extras.fingerprint, False
    computed = fingerprint_snapshot(run_dir / "dataset")
    if computed is None:
        return None, False
    extras.fingerprint = computed
    return computed, True


# ------------------------------------------------------------------ summaries


def _summarize(run_dir: Path, record: RunRecord) -> RunSummary:
    extras = read_extras(run_dir)
    best_epoch, scores, dirty_scores = _scores(run_dir, record, extras)
    fingerprint, dirty_fp = _fingerprint(run_dir, record, extras)
    if dirty_scores or dirty_fp:
        write_extras(run_dir, extras)
    return RunSummary(
        run=record,
        notes=extras.notes,
        tags=list(extras.tags),
        best_epoch=best_epoch,
        scores=scores,
        evals=_evals(run_dir),
        fingerprint=fingerprint,
    )


def _all_summaries(project: Project) -> list[RunSummary]:
    return [
        _summarize(_run_dir(project, record.run_id), record)
        for record in list_runs(project)
    ]


@capability(
    "experiment.run",
    summary="One run with its scores, evaluation metrics, notes and tags",
    web_route="/api/v1/experiments/runs/<run_id>",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos runs' prints every run with the same fields.",
)
def get_run_summary(project: Project, run_id: str) -> RunSummary:
    run_dir = _run_dir(project, run_id)
    return _summarize(run_dir, _reconcile(run_dir, read_record(run_dir)))
