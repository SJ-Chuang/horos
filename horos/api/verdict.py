"""Post-training verdict: a rule-based conclusion for every finished run.

Every terminal run gets a summary plus concrete findings — including when the
numbers look perfect: a near-perfect score on a narrow validation set usually
means leakage or bias, not a solved problem, so "too good" is itself a flag.

Pure logic over the run's recorded events and split counts; no ML imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.project import Project

if TYPE_CHECKING:
    from horos.api.train import RunRecord

__all__ = ["Finding", "RunVerdict", "build_verdict", "run_verdict"]

Severity = Literal["info", "warning", "critical"]

#: validation sets below these sizes make metrics noise, not signal
_VALID_CRITICAL = 10
_VALID_WARNING = 30
#: a final val mAP@50 at or above this triggers the "too good" check
_PERFECT_MAP = 0.99
#: final val mAP@50 below this means the model effectively detects nothing
_NO_LEARNING_MAP = 0.05
#: val loss this far above its own minimum counts as divergence (overfit)
_OVERFIT_RATIO = 1.05
#: train loss still dropping at least this much per epoch = not converged
_STILL_IMPROVING = 0.03


class Finding(BaseModel):
    severity: Severity
    title: str
    detail: str
    suggestion: str


class RunVerdict(BaseModel):
    run_id: str
    state: str
    summary: str
    findings: list[Finding] = Field(default_factory=list)


def _series(events: list[dict], key: str) -> list[tuple[int, float]]:
    points = []
    for event in events:
        if event.get("type") == "metrics" and key in event.get("metrics", {}):
            points.append((event["step"], float(event["metrics"][key])))
    return points


def _final_map50(events: list[dict]) -> float | None:
    for key in ("val/mAP_50", "val/ema_mAP_50", "val/mAP"):
        points = _series(events, key)
        if points:
            return points[-1][1]
    return None


def build_verdict(
    record: RunRecord, events: list[dict], splits: dict[str, int]
) -> RunVerdict:
    findings: list[Finding] = []
    add = findings.append

    train_loss = _series(events, "train/loss")
    val_loss = _series(events, "val/loss")
    map50 = _final_map50(events)
    total = sum(splits.values()) or 1
    valid = splits.get("valid", 0)
    test = splits.get("test", 0)

    if record.state == "failed":
        add(Finding(
            severity="critical",
            title="Training failed",
            detail=record.error or "The worker reported no error message.",
            suggestion=f"Check runs/{record.run_id}/worker.log for the full "
                       f"traceback, then start a new run.",
        ))
    elif record.state == "stopped":
        done = train_loss[-1][0] + 1 if train_loss else 0
        add(Finding(
            severity="info",
            title="Stopped before finishing",
            detail=f"Interrupted after ~{done} of "
                   f"{record.config.get('epochs', '?')} planned epochs.",
            suggestion="The best checkpoint up to the stop is kept; results "
                       "below reflect a partially trained model.",
        ))

    # --- validation set size gates everything else -------------------------
    if 0 < valid < _VALID_CRITICAL:
        add(Finding(
            severity="critical",
            title="Validation set is far too small",
            detail=f"Only {valid} validation image(s) "
                   f"({valid / total:.1%} of the dataset).",
            suggestion="Every metric here — including which checkpoint was "
                       "kept as 'best' — is decided by a handful of images. "
                       "Re-split (e.g. 80/10/10) and retrain before drawing "
                       "any conclusion.",
        ))
    elif 0 < valid < _VALID_WARNING:
        add(Finding(
            severity="warning",
            title="Validation set is small",
            detail=f"{valid} validation images ({valid / total:.1%} of the "
                   f"dataset).",
            suggestion="Metrics will be noisy at this size; treat differences "
                       "of a few percent as ties. Consider a larger split.",
        ))

    # --- suspiciously perfect (the user asked for exactly this check) ------
    if map50 is not None and map50 >= _PERFECT_MAP:
        add(Finding(
            severity="warning",
            title="Metrics are near-perfect — verify before trusting",
            detail=f"Final val mAP@50 = {map50:.3f}.",
            suggestion="Perfect scores usually mean the validation set is too "
                       "easy or leaks from training: check for near-duplicate "
                       "frames across splits, and whether the validation "
                       "images cover the conditions you will deploy in "
                       "(lighting, angles, backgrounds, object sizes). "
                       "Validate on genuinely unseen data before shipping.",
        ))

    # --- no learning --------------------------------------------------------
    if map50 is not None and map50 < _NO_LEARNING_MAP and record.state == "completed":
        add(Finding(
            severity="warning",
            title="The model barely detects anything yet",
            detail=f"Final val mAP@50 = {map50:.3f}.",
            suggestion="Common causes in order: too few epochs for the "
                       "dataset size, annotation problems (wrong classes or "
                       "boxes), or objects too small for the input "
                       "resolution. Check a few predictions by hand first.",
        ))

    # --- overfitting ---------------------------------------------------------
    overfit = False
    if len(val_loss) >= 3 and len(train_loss) >= 3:
        val_values = [v for _, v in val_loss]
        val_min = min(val_values)
        best_epoch = val_loss[val_values.index(val_min)][0]
        train_dropping = train_loss[-1][1] < train_loss[0][1] * 0.9
        if val_values[-1] > val_min * _OVERFIT_RATIO and train_dropping:
            overfit = True
            add(Finding(
                severity="warning",
                title="Validation loss diverged from training loss",
                detail=f"val/loss bottomed out at epoch {best_epoch} "
                       f"({val_min:.3f}) and ended at {val_values[-1]:.3f} "
                       f"while train/loss kept falling.",
                suggestion="The model is memorizing the training set. The "
                           "kept 'best' checkpoint is from the val minimum, "
                           "so it is still usable — but more training data "
                           "or stronger augmentation is the real fix; more "
                           "epochs will not help.",
            ))

    # --- still improving ------------------------------------------------------
    # never suggest training longer when val loss already diverged: the falling
    # train loss is the overfitting, not headroom
    if record.state == "completed" and len(train_loss) >= 3 and not overfit:
        prev, last = train_loss[-2][1], train_loss[-1][1]
        val_still_ok = len(val_loss) < 2 or val_loss[-1][1] <= val_loss[-2][1]
        if prev > 0 and (prev - last) / prev >= _STILL_IMPROVING and val_still_ok:
            add(Finding(
                severity="info",
                title="Training was still improving when it ended",
                detail=f"train/loss fell {(prev - last) / prev:.1%} in the "
                       f"final epoch and val/loss had not turned upward.",
                suggestion="A longer run (or resuming from this checkpoint) "
                           "will likely improve the result.",
            ))

    # --- no held-out test set ---------------------------------------------------
    if test == 0:
        add(Finding(
            severity="info",
            title="No held-out test split",
            detail="The validation set both selected the best checkpoint and "
                   "produced the final numbers.",
            suggestion="That makes the reported metrics an optimistic "
                       "estimate. Keep a test split the model never "
                       "influences for the numbers you report.",
        ))

    order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order[f.severity])
    return RunVerdict(
        run_id=record.run_id,
        state=record.state,
        summary=_summary(record.state, findings),
        findings=findings,
    )


def _summary(state: str, findings: list[Finding]) -> str:
    critical = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]
    if critical:
        return critical[0].title
    if warnings:
        heads = "; ".join(f.title for f in warnings[:2])
        return f"Completed, but check: {heads}"
    if state == "completed":
        return ("No red flags — the metrics are as trustworthy as the "
                "validation set allows.")
    return f"Run ended in state '{state}'."


@capability(
    "train.verdict",
    summary="Rule-based conclusion and suggestions for a finished training run",
    web_route="/api/v1/train/runs/<run_id>/verdict",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos train' prints the verdict automatically when the "
                    "run finishes.",
)
def run_verdict(project: Project, run_id: str) -> RunVerdict:
    """Analyze a run's recorded events and split sizes into a conclusion.

    Computed on demand from events.jsonl — rules can improve over time and
    old runs get the improved analysis for free.
    """
    from horos.api.train import _run_dir, _split_counts, read_record

    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    from horos.api.train import _read_events

    events, _ = _read_events(run_dir)
    return build_verdict(record, events, _split_counts(run_dir, record))
