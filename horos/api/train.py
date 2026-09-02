"""Training run lifecycle (E5-T3): start, poll, stop.

Design decision (confirmed): training runs in a dedicated spawned subprocess
(`python -m horos.api.train_worker <run_dir>`), never inside the Flask/API
process — an OOM or crash cannot take the server down, stop is a real kill,
and the worker is spawn-safe by construction (E5-T6b). The worker appends R4
events to `<run_dir>/events.jsonl`; this module reads them back for polling.

Run artifacts live inside the project (confirmed): `<project>/runs/<run_id>/`
holds config.json, run.json, events.jsonl, the exported training dataset, and
checkpoints/ — a project directory carries its full experiment history (E7
scans this same layout later).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from horos.api.dataset import dataset_stats, export_dataset, filter_dataset_categories
from horos.api.hparams import DerivedValue, HyperparameterPlan, derive_plan
from horos.api.manifest import capability
from horos.api.system import ensure_supported
from horos.core.project import Project
from horos.core.registry import get_model_info
from horos.errors import LicenseError, ProjectError, UnknownModelError

logger = logging.getLogger(__name__)

__all__ = [
    "TrainRunConfig",
    "RunRecord",
    "TrainStatus",
    "delete_run",
    "derive_hyperparameters",
    "start_training",
    "training_status",
    "stop_training",
    "list_runs",
]

RunState = Literal["pending", "running", "completed", "failed", "stopped"]

_RUN_JSON = "run.json"
_CONFIG_JSON = "config.json"
_EVENTS_JSONL = "events.jsonl"
_STOP_FLAG = "stop.flag"
_WORKER_LOG = "worker.log"


class TrainRunConfig(BaseModel):
    """User-facing training request. Only `model` semantics are horos-level;
    everything else maps onto the backend-neutral TrainSpec.

    Hyperparameters left as None are derived from dataset statistics with a
    recorded reason (E5-T1); a set value is a user override and is marked as
    such in the plan without disturbing the other derivations (E5-T2).
    """

    model: str = "rfdetr-nano"
    epochs: int | None = None
    batch_size: int | None = None
    resolution: int | None = None
    device: str | None = None
    seed: int | None = None
    resume_from: str | None = None
    #: category names to train on; None = all. Unselected classes' objects
    #: become background in this run's dataset snapshot.
    categories: list[str] | None = None
    acknowledge_non_apache: bool = False
    #: expert passthrough to the backend's own knobs — applied last, wins (E5-S5)
    extra: dict[str, Any] = Field(default_factory=dict)
    #: testing hook: "module:ClassName" resolved instead of the model registry.
    #: Never set this in production code.
    entrypoint_override: str | None = None


class RunRecord(BaseModel):
    run_id: str
    model: str
    state: RunState
    created_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    #: derivation trail: every hyperparameter with its value, reason, and
    #: whether the user overrode it (E5-S2 — "why did the system pick this")
    hparams: list[DerivedValue] = Field(default_factory=list)
    hparam_notes: list[str] = Field(default_factory=list)
    device: str | None = None
    pid: int | None = None
    checkpoint: str | None = None
    error: str | None = None
    dataset_images: int = 0
    #: split sizes at training time — the verdict rules read these
    dataset_splits: dict[str, int] = Field(default_factory=dict)
    #: class names this run trained on (its snapshot's categories); resuming
    #: is only valid with exactly this set, so the UI locks the picker to it
    dataset_classes: list[str] = Field(default_factory=list)
    #: the full-state checkpoint (weights + optimizer + LR schedule; rfdetr's
    #: last.ckpt) — the right resume source. `checkpoint` stays the best
    #: weights for inference/export; resuming from it restarts the optimizer
    #: cold and the loss spikes for a few epochs.
    resume_checkpoint: str | None = None
    #: epochs this run actually finished — a resume's TOTAL epochs must exceed
    #: this or the trainer has nothing left to do
    epochs_completed: int | None = None


class TrainStatus(BaseModel):
    run: RunRecord
    events: list[dict[str, Any]] = Field(default_factory=list)  # events[after:]
    num_events: int = 0


# ------------------------------------------------------------------ run storage


def runs_root(project: Project) -> Path:
    return project.root / "runs"


def _run_dir(project: Project, run_id: str) -> Path:
    run_dir = runs_root(project) / run_id
    if not (run_dir / _RUN_JSON).is_file():
        raise ProjectError(f"No such training run: {run_id}")
    return run_dir


def read_record(run_dir: Path) -> RunRecord:
    return RunRecord.model_validate_json((run_dir / _RUN_JSON).read_text("utf-8"))


def write_record(run_dir: Path, record: RunRecord) -> None:
    """Atomic replace so a poll never reads a half-written run.json (R7:
    os.replace is atomic on POSIX and Windows alike)."""
    tmp = run_dir / f"{_RUN_JSON}.tmp"
    tmp.write_text(record.model_dump_json(indent=2), "utf-8")
    os.replace(tmp, run_dir / _RUN_JSON)


def _read_events(run_dir: Path, after: int = 0) -> tuple[list[dict[str, Any]], int]:
    path = run_dir / _EVENTS_JSONL
    if not path.is_file():
        return [], 0
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # a killed worker can leave a torn final line — not an error
                continue
    return events[after:], len(events)


# Popen handles for workers this process spawned. Needed for liveness: an
# unreaped child that died is a zombie, and os.kill(pid, 0) still succeeds on
# zombies — poll() both answers correctly and reaps. Workers spawned by an
# earlier server process get reparented to init on its exit, so the plain pid
# check below is accurate for them.
_PROCESSES: dict[str, subprocess.Popen] = {}


def _worker_alive(record: RunRecord) -> bool:
    process = _PROCESSES.get(record.run_id)
    if process is not None:
        return process.poll() is None
    return _pid_alive(record.pid)


def _run_class_names(run_dir: Path) -> list[str] | None:
    """The class list a run trained on, read from its dataset snapshot."""
    gt_path = run_dir / "dataset" / "train" / "_annotations.coco.json"
    if not gt_path.is_file():
        return None
    try:
        gt = json.loads(gt_path.read_text("utf-8"))
        return sorted(c["name"] for c in gt.get("categories", []))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _run_completed_epochs(run_dir: Path) -> int | None:
    """Epochs a run actually finished, from its event log (the relay emits a
    progress event with phase="epoch completed" and current=epoch+1 at each
    train-epoch end). Other progress events must NOT count: weight-download
    progress carries BYTES in `current`, which would read as hundreds of
    millions of epochs."""
    events, _ = _read_events(run_dir)
    completed = [
        e["current"]
        for e in events
        if e.get("type") == "progress"
        and e.get("phase") == "epoch completed"
        and isinstance(e.get("current"), int)
    ]
    return max(completed) if completed else None


def _snapshot_class_names(checkpoint: Path) -> list[str] | None:
    """Class list for a checkpoint inside a horos run
    (runs/<id>/checkpoints/x.pth → runs/<id>/dataset/train/...);
    None when the checkpoint is not inside a run (nothing to verify)."""
    return _run_class_names(checkpoint.parent.parent)


def _split_counts(run_dir: Path, record: RunRecord) -> dict[str, int]:
    """Split sizes at training time; older runs predate the recorded field,
    so fall back to counting the images exported into the run directory."""
    if record.dataset_splits:
        return record.dataset_splits
    from horos.core.formats import IMAGE_SUFFIXES

    counts = {}
    for split in ("train", "valid", "test"):
        split_dir = run_dir / "dataset" / split
        counts[split] = (
            sum(1 for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if split_dir.is_dir()
            else 0
        )
    return counts


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(ok) and exit_code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reconcile(run_dir: Path, record: RunRecord) -> RunRecord:
    """A run that claims to be active but whose worker is gone died without a
    terminal event (killed, OOM-killed, machine crash). Settle its state."""
    changed = False
    if not record.dataset_classes:  # backfill runs recorded before the field
        names = _run_class_names(run_dir)
        if names:
            record.dataset_classes = names
            changed = True
    if record.state not in ("pending", "running"):
        if record.resume_checkpoint is None:
            last = run_dir / "checkpoints" / "last.ckpt"
            if last.is_file():
                record.resume_checkpoint = str(last)
                changed = True
        if record.epochs_completed is None:
            completed = _run_completed_epochs(run_dir)
            if completed is not None:
                record.epochs_completed = completed
                changed = True
    if changed:
        write_record(run_dir, record)
    if record.state not in ("pending", "running") or _worker_alive(record):
        return record
    if (run_dir / _STOP_FLAG).is_file():
        record.state = "stopped"
    else:
        record.state = "failed"
        record.error = (
            "The training worker exited without reporting a result "
            f"(see {run_dir / _WORKER_LOG})."
        )
    write_record(run_dir, record)
    return record


# ---------------------------------------------------------------------- API


@capability(
    "train.derive",
    summary="Derive hyperparameters from dataset statistics, with reasons",
    web_route="/api/v1/train/derive",
    web_methods=("POST",),
    cli=None,
    not_cli_because=(
        "'horos train' derives automatically and records the plan in run.json; "
        "a standalone preview command ships with the training UI (E5-T8)."
    ),
)
def derive_hyperparameters(
    project: Project, config: TrainRunConfig | None = None
) -> HyperparameterPlan:
    """Rule-based plan (E5-T1): dataset statistics + model metadata + memory
    probe in, values with reasons out. Values set on `config` are honored as
    user overrides (E5-T2). The UI shows this plan before a run starts (E5-S2)."""
    from horos.backends.memory import probe_memory

    config = config or TrainRunConfig()
    try:
        model_info = get_model_info(config.model)
    except UnknownModelError:
        model_info = None  # testing backends: derive without model metadata
    memory = probe_memory(config.device.partition(":")[0] if config.device else None)
    if config.categories is not None:
        # the rules must see the data this run will actually train on
        from horos.core.stats import compute_stats

        stats = compute_stats(
            filter_dataset_categories(project.to_dataset(), config.categories)
        )
    else:
        stats = dataset_stats(project)
    return derive_plan(
        stats,
        model=config.model,
        model_info=model_info,
        memory=memory,
        overrides={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "resolution": config.resolution,
        },
    )


@capability(
    "train.start",
    summary="Start a training run in a dedicated worker subprocess",
    web_route="/api/v1/train",
    web_methods=("POST",),
    cli="train",
)
def start_training(project: Project, config: TrainRunConfig | None = None) -> RunRecord:
    """Export the dataset into a new run directory and spawn the worker.

    Returns immediately with the run's record; poll with `training_status`.
    One active run per project — a second start is refused, never queued.
    """
    config = config or TrainRunConfig()
    ensure_supported("training")

    if config.entrypoint_override is None:
        info = get_model_info(config.model)
        if info.requires_acknowledgement and not config.acknowledge_non_apache:
            raise LicenseError(
                f"Model '{config.model}' is licensed under {info.weights_license}, "
                f"not Apache 2.0 (see {info.license_url}). Pass "
                f"acknowledge_non_apache=True if you have reviewed and accept it."
            )

    active = next(
        (r for r in list_runs(project) if r.state in ("pending", "running")), None
    )
    if active is not None:
        raise ProjectError(
            f"Training run {active.run_id} is already {active.state}. horos runs "
            f"one training at a time — stop it first or wait for it to finish."
        )

    dataset = project.to_dataset()
    if config.categories is not None:
        dataset = filter_dataset_categories(dataset, config.categories)
    # allocation follows the project's split assignment (the Dataset page):
    # train trains, valid validates, test stays untouched for later evaluation
    split_counts = {
        split: len(dataset.images_in_split(split))
        for split in ("train", "valid", "test")
    }
    train_count, valid_count = split_counts["train"], split_counts["valid"]
    if train_count == 0 or valid_count == 0:
        raise ProjectError(
            f"Training needs a non-empty train and valid split (found "
            f"train={train_count}, valid={valid_count}). Re-split on the "
            f"Dataset page (or resplit() / 'horos split') first."
        )
    if config.categories is not None:
        train_ids = {i.id for i in dataset.images_in_split("train")}
        if not any(a.image_id in train_ids for a in dataset.annotations):
            raise ProjectError(
                f"The selected categories {config.categories} have no "
                f"annotations in the train split — nothing to learn from."
            )

    if config.resume_from:
        # the checkpoint's class head has a fixed shape: resuming with a
        # different class set fails deep inside the backend with a raw
        # state_dict size-mismatch — refuse it here with the actual fix
        source_names = _snapshot_class_names(Path(config.resume_from))
        new_names = sorted({c.name for c in dataset.categories})
        if source_names is not None and source_names != new_names:
            raise ProjectError(
                f"Cannot resume: the checkpoint was trained on classes "
                f"{source_names}, but this run would train on {new_names}. "
                f"The class head's shape is fixed by the checkpoint — select "
                f"exactly the source run's classes, or start a fresh run "
                f"(without resume) to train on the new class set."
            )

    # Fill every unset hyperparameter from the rule-based plan (E5-T1) and
    # hand the worker a fully resolved config; the plan (values + reasons)
    # goes into the run metadata so the "why" survives with the run (E5-S2).
    plan = derive_hyperparameters(project, config)
    spec_values = plan.spec_fields()
    resolved = config.model_copy(
        update={
            "epochs": spec_values["epochs"],
            "batch_size": spec_values["batch_size"],
            "resolution": spec_values.get("resolution", config.resolution),
            # derived backend knobs ride in extra; the user's own extra wins
            "extra": {**plan.extra_fields(), **config.extra},
        }
    )

    if config.resume_from:
        # epochs is the TOTAL count and the trainer restores the checkpoint's
        # epoch — a total at or below what is already done raises a raw
        # MisconfigurationException deep in the backend; refuse it up front
        source_dir = Path(config.resume_from).parent.parent
        completed = None
        if (source_dir / _RUN_JSON).is_file():
            source_record = read_record(source_dir)
            completed = (
                source_record.epochs_completed
                or _run_completed_epochs(source_dir)
            )
        if (
            completed
            and resolved.epochs is not None
            and resolved.epochs <= completed
        ):
            raise ProjectError(
                f"Cannot resume: the checkpoint already completed {completed} "
                f"epochs and epochs is the TOTAL count — {resolved.epochs} "
                f"leaves nothing to train. Set epochs above {completed} "
                f"(e.g. {completed + max(10, completed // 2)})."
            )

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = runs_root(project) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # The run keeps the exact data it trained on — reproducible by design,
    # at the cost of copying images per run (dataset fingerprints in E7 will
    # let identical exports be shared).
    export_dataset(
        project, run_dir / "dataset", format="coco", categories=config.categories
    )

    (run_dir / _CONFIG_JSON).write_text(resolved.model_dump_json(indent=2), "utf-8")
    record = RunRecord(
        run_id=run_id,
        model=config.model,
        state="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
        config=resolved.model_dump(exclude={"entrypoint_override"}),
        hparams=plan.derivations,
        hparam_notes=plan.notes,
        device=config.device,
        dataset_images=len(dataset.images),
        dataset_splits=split_counts,
        dataset_classes=sorted({c.name for c in dataset.categories}),
    )
    write_record(run_dir, record)

    with (run_dir / _WORKER_LOG).open("ab") as log:
        process = subprocess.Popen(  # noqa: S603 — our own module, our interpreter
            [sys.executable, "-m", "horos.api.train_worker", str(run_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    _PROCESSES[run_id] = process
    record.pid = process.pid
    record.state = "running"
    write_record(run_dir, record)
    logger.info("training run %s started (pid %d)", run_id, process.pid)
    return record


@capability(
    "train.status",
    summary="Poll a training run: state plus events after a given index",
    web_route="/api/v1/train/runs/<run_id>",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos train' runs in the foreground and prints events directly.",
)
def training_status(project: Project, run_id: str, *, after: int = 0) -> TrainStatus:
    run_dir = _run_dir(project, run_id)
    record = _reconcile(run_dir, read_record(run_dir))
    events, num_events = _read_events(run_dir, after)
    return TrainStatus(run=record, events=events, num_events=num_events)


@capability(
    "train.stop",
    summary="Stop a running training; checkpoints from finished epochs remain",
    web_route="/api/v1/train/runs/<run_id>/stop",
    web_methods=("POST",),
    cli=None,
    not_cli_because="'horos train' runs in the foreground; Ctrl+C stops it.",
)
def stop_training(project: Project, run_id: str) -> bool:
    """Best weights so far survive (E5-S4): rfdetr checkpoints every epoch, so
    terminating the worker only loses the epoch in flight."""
    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    if record.state not in ("pending", "running"):
        return False
    (run_dir / _STOP_FLAG).touch()
    if record.pid is not None and _worker_alive(record):
        try:
            os.kill(record.pid, signal.SIGTERM)
        except OSError:  # already gone between the check and the kill
            pass
    return True


@capability(
    "train.delete",
    summary="Delete a training run and everything it stored",
    web_route="/api/v1/train/runs/<run_id>",
    web_methods=("DELETE",),
    cli=None,
    not_cli_because="Run housekeeping is a UI concern; 'rm -r runs/<id>' works too.",
)
def delete_run(project: Project, run_id: str) -> bool:
    """Remove runs/<id> — checkpoints, dataset snapshot, events, eval reports.
    An active run must be stopped first; deletion is permanent."""
    import shutil

    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    if record.state in ("pending", "running") and _worker_alive(record):
        raise ProjectError(
            f"Run {run_id} is still {record.state} — stop it before deleting."
        )
    _PROCESSES.pop(run_id, None)
    shutil.rmtree(run_dir)
    logger.info("deleted training run %s", run_id)
    return True


@capability(
    "train.runs",
    summary="List this project's training runs, newest first",
    web_route="/api/v1/train/runs",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Run listing and comparison ship with experiment management (E7).",
)
def list_runs(project: Project) -> list[RunRecord]:
    root = runs_root(project)
    if not root.is_dir():
        return []
    records = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if (run_dir / _RUN_JSON).is_file():
            records.append(_reconcile(run_dir, read_record(run_dir)))
    return records
