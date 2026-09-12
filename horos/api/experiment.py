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
from horos.core.fingerprint import (
    DatasetFingerprint,
    compare_fingerprints,
    fingerprint_dataset,
    fingerprint_snapshot,
)
from horos.core.project import Project
from horos.errors import ProjectError

__all__ = [
    "RunExtras",
    "RunSummary",
    "Comparability",
    "RunComparison",
    "RunQueryResult",
    "compare_runs",
    "get_run_summary",
    "query_runs",
    "update_run_notes",
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


class Comparability(BaseModel):
    """Whether a run's metrics may be compared with the reference's (E7-S4).

    `reference` is a run id, or "project" for the project's data as it is
    today (fingerprinted under the run's own class scope)."""

    reference: str
    comparable: bool
    reason: str
    changed_splits: list[str] = Field(default_factory=list)
    classes_changed: bool = False


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
    #: against the query's reference; None when either side has no fingerprint
    comparability: Comparability | None = None


class RunQueryResult(BaseModel):
    runs: list[RunSummary] = Field(default_factory=list)
    sort_by: str
    descending: bool
    #: what every run's `comparability` was judged against
    reference: str = "project"
    #: every sort key valid for this project's runs right now (record fields,
    #: score keys, eval.<split>.<metric>) — the UI's sort menu is built from it
    sort_keys: list[str] = Field(default_factory=list)


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
def get_run_summary(
    project: Project, run_id: str, *, reference: str | None = "project"
) -> RunSummary:
    run_dir = _run_dir(project, run_id)
    summary = _summarize(run_dir, _reconcile(run_dir, read_record(run_dir)))
    if reference:
        summary.comparability = judge_comparability(project, summary, reference)
    return summary


# ------------------------------------------------------------------ notes/tags

_MAX_TAG = 40
_MAX_TAGS = 32
_MAX_NOTES = 20_000


def normalize_tags(tags) -> list[str]:
    """Trim, drop empties and duplicates (first occurrence wins), keep order.
    Commas are refused because tag lists travel comma-separated through the
    CLI and query strings."""
    out: list[str] = []
    for raw in tags or []:
        tag = str(raw).strip()
        if not tag:
            continue
        if "," in tag:
            raise ProjectError(f"Tag {tag!r} may not contain a comma")
        if len(tag) > _MAX_TAG:
            raise ProjectError(f"Tag {tag!r} is longer than {_MAX_TAG} characters")
        if tag.casefold() not in {t.casefold() for t in out}:
            out.append(tag)
    if len(out) > _MAX_TAGS:
        raise ProjectError(f"A run can carry at most {_MAX_TAGS} tags")
    return out


@capability(
    "experiment.annotate",
    summary="Set a run's notes, or replace / add / remove its tags",
    web_route="/api/v1/experiments/runs/<run_id>",
    web_methods=("PATCH",),
    cli="tag",
)
def update_run_notes(
    project: Project,
    run_id: str,
    *,
    notes: str | None = None,
    tags: list[str] | None = None,
    add_tags: list[str] | None = None,
    remove_tags: list[str] | None = None,
) -> RunSummary:
    """Every argument is optional and independent: `notes` replaces the
    notes, `tags` replaces the whole tag list, `add_tags` / `remove_tags`
    edit it in place (matching case-insensitively). Nothing else on the run
    is touched — the record stays the worker's."""
    run_dir = _run_dir(project, run_id)
    extras = read_extras(run_dir)
    changed = False
    if notes is not None:
        text = str(notes).rstrip()
        if len(text) > _MAX_NOTES:
            raise ProjectError(f"Notes are limited to {_MAX_NOTES} characters")
        changed |= text != extras.notes
        extras.notes = text
    if tags is not None:
        new_tags = normalize_tags(tags)
        changed |= new_tags != extras.tags
        extras.tags = new_tags
    if add_tags:
        merged = normalize_tags([*extras.tags, *add_tags])
        changed |= merged != extras.tags
        extras.tags = merged
    if remove_tags:
        drop = {t.strip().casefold() for t in remove_tags}
        kept = [t for t in extras.tags if t.casefold() not in drop]
        changed |= kept != extras.tags
        extras.tags = kept
    if changed:
        extras.updated_at = _now()
        write_extras(run_dir, extras)
    return _summarize(run_dir, _reconcile(run_dir, read_record(run_dir)))


# ------------------------------------------------------------------ query

_RECORD_SORT_KEYS = ("created_at", "run_id", "model", "state", "epochs_completed")


def _sort_value(summary: RunSummary, key: str):
    if key in _RECORD_SORT_KEYS:
        return getattr(summary.run, key)
    if key in summary.scores:
        return summary.scores[key]
    if key.startswith("eval."):
        _, _, rest = key.partition(".")
        split, _, metric = rest.partition(".")
        return summary.evals.get(split, {}).get(metric)
    return None


def available_sort_keys(summaries: list[RunSummary]) -> list[str]:
    keys: list[str] = list(_RECORD_SORT_KEYS)
    scores = sorted({k for s in summaries for k in s.scores})
    evals = sorted(
        f"eval.{split}.{metric}"
        for s in summaries
        for split, metrics in s.evals.items()
        for metric in metrics
    )
    for key in [*scores, *evals]:
        if key not in keys:
            keys.append(key)
    return keys


def sort_summaries(
    summaries: list[RunSummary], sort_by: str, *, descending: bool = True
) -> list[RunSummary]:
    """Runs lacking the key sort last in either direction; ties fall back to
    newest first so the order is stable across refreshes."""
    keys = available_sort_keys(summaries)
    if sort_by not in keys:
        raise ProjectError(
            f"Unknown sort key '{sort_by}'. Available for this project: {', '.join(keys)}"
        )
    present = [s for s in summaries if _sort_value(s, sort_by) is not None]
    missing = [s for s in summaries if _sort_value(s, sort_by) is None]
    present.sort(key=lambda s: s.run.created_at, reverse=True)
    present.sort(key=lambda s: _sort_value(s, sort_by), reverse=descending)
    missing.sort(key=lambda s: s.run.created_at, reverse=True)
    return [*present, *missing]


@capability(
    "experiment.runs",
    summary="List runs with scores, sorted by any metric, filtered by state or tag",
    web_route="/api/v1/experiments/runs",
    web_methods=("GET",),
    cli="runs",
)
def query_runs(
    project: Project,
    *,
    sort_by: str = "created_at",
    descending: bool = True,
    states: list[str] | None = None,
    tags: list[str] | None = None,
    reference: str | None = "project",
) -> RunQueryResult:
    """Every run of the project as a RunSummary. `states` keeps only runs in
    those states; `tags` keeps runs carrying ALL the given tags (matched
    case-insensitively). Sorting happens after filtering, and `sort_keys`
    lists what the project's runs can currently be sorted by. Each run's
    `comparability` is judged against `reference` (a run id, "project" for
    today's data, or None to skip the judgement)."""
    summaries = _all_summaries(project)
    if states:
        wanted = {s.strip() for s in states if s.strip()}
        summaries = [s for s in summaries if s.run.state in wanted]
    if tags:
        wanted_tags = {t.strip().casefold() for t in tags if t.strip()}
        summaries = [
            s for s in summaries
            if wanted_tags <= {t.casefold() for t in s.tags}
        ]
    ordered = sort_summaries(summaries, sort_by, descending=descending)
    if reference:
        judge = _comparability_judge(project, reference)
        for summary in ordered:
            summary.comparability = judge(summary)
    return RunQueryResult(
        runs=ordered,
        sort_by=sort_by,
        descending=descending,
        reference=reference or "",
        sort_keys=available_sort_keys(summaries),
    )


# ------------------------------------------------------------------ comparability


def _project_fingerprint(project: Project, record: RunRecord) -> DatasetFingerprint:
    """Today's project data under the run's own class scope, so a run trained
    on two of five classes is compared against those two classes today."""
    from horos.api.dataset import filter_dataset_categories

    dataset = project.to_dataset()
    categories = record.config.get("categories")
    if categories is not None:
        dataset = filter_dataset_categories(
            dataset,
            list(categories),
            include_background=bool(record.config.get("include_background", False)),
        )
    return fingerprint_dataset(dataset)


def _judge(
    summary: RunSummary, reference: str, reference_fp: DatasetFingerprint | None
) -> Comparability | None:
    if summary.fingerprint is None or reference_fp is None:
        return None
    diff = compare_fingerprints(summary.fingerprint, reference_fp)
    target = "the project's current data" if reference == "project" else f"run {reference}"
    if diff.identical:
        reason = f"Trained on the same data as {target}"
    else:
        reason = (
            f"Not directly comparable with {target}: {diff.describe()} — metrics "
            f"were measured on different data"
        )
    return Comparability(
        reference=reference,
        comparable=diff.identical,
        reason=reason,
        changed_splits=diff.changed_splits,
        classes_changed=diff.classes_changed,
    )


def _comparability_judge(project: Project, reference: str):
    """A callable judging summaries against one reference, resolving the
    reference once. For "project" the fingerprint depends on each run's class
    scope, so it is computed per distinct scope and memoized."""
    if reference == "project":
        cache: dict[str, DatasetFingerprint] = {}

        def judge(summary: RunSummary) -> Comparability | None:
            scope = repr((
                summary.run.config.get("categories"),
                summary.run.config.get("include_background", False),
            ))
            if scope not in cache:
                cache[scope] = _project_fingerprint(project, summary.run)
            return _judge(summary, reference, cache[scope])

        return judge

    ref_dir = _run_dir(project, reference)
    ref_fp, _ = _fingerprint(ref_dir, read_record(ref_dir), read_extras(ref_dir))
    return lambda summary: _judge(summary, reference, ref_fp)


def judge_comparability(
    project: Project, summary: RunSummary, reference: str = "project"
) -> Comparability | None:
    return _comparability_judge(project, reference)(summary)


# ------------------------------------------------------------------ side by side


class ComparisonRow(BaseModel):
    name: str
    #: one value per compared run, in request order (None = not recorded)
    values: list[object] = Field(default_factory=list)
    #: for hyperparameters: the derivation reason per run
    reasons: list[str | None] = Field(default_factory=list)
    differs: bool = False


class RunComparison(BaseModel):
    runs: list[RunSummary] = Field(default_factory=list)
    hparams: list[ComparisonRow] = Field(default_factory=list)
    metrics: list[ComparisonRow] = Field(default_factory=list)
    dataset: list[ComparisonRow] = Field(default_factory=list)


def _rows(names: list[str], per_run: list[dict], reasons: list[dict] | None = None):
    rows = []
    for name in names:
        values = [d.get(name) for d in per_run]
        rows.append(ComparisonRow(
            name=name,
            values=values,
            reasons=[r.get(name) for r in reasons] if reasons else [],
            differs=len({repr(v) for v in values}) > 1,
        ))
    return rows


@capability(
    "experiment.compare",
    summary="Compare runs side by side: hyperparameters, metrics, dataset",
    web_route="/api/v1/experiments/compare",
    web_methods=("GET",),
    cli="compare",
)
def compare_runs(project: Project, run_ids: list[str]) -> RunComparison:
    """Side-by-side table for 1..8 runs. Every run's comparability is judged
    against the FIRST run given; rows whose values differ are flagged so the
    UI can highlight what actually changed between runs (E7-S1)."""
    ids = [r for r in run_ids if r]
    if not ids:
        raise ProjectError("compare_runs needs at least one run id")
    if len(ids) > 8:
        raise ProjectError("compare_runs handles at most 8 runs at once")
    if len(set(ids)) != len(ids):
        raise ProjectError("compare_runs: the same run was given twice")
    summaries = [get_run_summary(project, run_id, reference=ids[0]) for run_id in ids]

    hparam_values, hparam_reasons = [], []
    for s in summaries:
        hparam_values.append({h.name: h.value for h in s.run.hparams})
        hparam_reasons.append({h.name: h.reason for h in s.run.hparams})
    hparam_names: list[str] = []
    for d in hparam_values:
        for name in d:
            if name not in hparam_names:
                hparam_names.append(name)

    metric_values = []
    for s in summaries:
        flat = {**s.scores}
        for split, metrics in s.evals.items():
            for key, value in metrics.items():
                flat[f"eval.{split}.{key}"] = value
        if s.best_epoch is not None:
            flat["best_epoch"] = s.best_epoch
        metric_values.append(flat)
    metric_names = sorted({k for d in metric_values for k in d})

    dataset_values = [
        {
            "images": s.run.dataset_images,
            "classes": ", ".join(s.run.dataset_classes),
            **{f"split.{k}": v for k, v in sorted(s.run.dataset_splits.items())},
            "fingerprint": s.fingerprint.digest if s.fingerprint else None,
        }
        for s in summaries
    ]
    dataset_names: list[str] = []
    for d in dataset_values:
        for name in d:
            if name not in dataset_names:
                dataset_names.append(name)

    return RunComparison(
        runs=summaries,
        hparams=_rows(hparam_names, hparam_values, hparam_reasons),
        metrics=_rows(metric_names, metric_values),
        dataset=_rows(dataset_names, dataset_values),
    )
