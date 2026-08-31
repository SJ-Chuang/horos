"""Training worker subprocess: `python -m horos.api.train_worker <run_dir>`.

Spawned by `horos.api.train.start_training` — never imported into the server's
request path. Spawn-safe by construction (R7/E5-T6b): everything runs under
the `__main__` guard, so Windows/macOS `spawn` re-imports of this module are
side-effect free, including the DataLoader workers rfdetr starts.

Contract with the parent:
- reads  <run_dir>/config.json   (a TrainRunConfig dump)
- writes <run_dir>/events.jsonl  (R4 events, one per line, appended live)
- writes <run_dir>/run.json      (terminal state, checkpoint path, error)
- trains into <run_dir>/checkpoints/ from <run_dir>/dataset/
- exits on SIGTERM, settling run.json to "stopped" first when it can
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def _load_backend(config):
    """Resolve the backend inside the worker process — the heavy imports never
    touch the parent (R1b stays true for the server)."""
    if config.entrypoint_override:
        import importlib

        module_name, _, class_name = config.entrypoint_override.partition(":")
        backend_cls = getattr(importlib.import_module(module_name), class_name)
        return backend_cls(None, device=config.device)
    from horos.backends import get_backend

    return get_backend(
        config.model,
        acknowledge_non_apache=config.acknowledge_non_apache,
        device=config.device,
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m horos.api.train_worker <run_dir>", file=sys.stderr)  # noqa: T201
        return 2
    run_dir = Path(argv[0]).resolve()

    from horos.api.train import TrainRunConfig, read_record, write_record
    from horos.backends.base import RunFailed, TrainSpec, dump_event

    config = TrainRunConfig.model_validate_json(
        (run_dir / "config.json").read_text("utf-8")
    )
    record = read_record(run_dir)
    record.state = "running"
    record.pid = os.getpid()
    write_record(run_dir, record)

    def settle(state: str, *, checkpoint: str | None = None, error: str | None = None):
        current = read_record(run_dir)
        current.state = state  # type: ignore[assignment]
        current.checkpoint = checkpoint or current.checkpoint
        current.error = error
        write_record(run_dir, current)

    def on_sigterm(signum, frame):  # noqa: ANN001, ARG001
        # Checkpoints from finished epochs are already on disk (E5-S4).
        settle("stopped")
        os._exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)

    events_path = run_dir / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as log:

        def emit(event) -> None:
            log.write(dump_event(event) + "\n")
            log.flush()

        try:
            backend = _load_backend(config)
            spec = TrainSpec(
                dataset_dir=run_dir / "dataset",
                output_dir=run_dir / "checkpoints",
                epochs=config.epochs,
                batch_size=config.batch_size,
                resolution=config.resolution,
                device=config.device,
                seed=config.seed,
                resume_from=Path(config.resume_from) if config.resume_from else None,
                extra=config.extra,
            )
            final_state, checkpoint, error = "failed", None, "no terminal event"
            for event in backend.train(spec):
                emit(event)
                if event.type == "completed":
                    final_state = "completed"
                    checkpoint = event.result.get("checkpoint")
                    error = None
                elif event.type == "failed":
                    final_state = "failed"
                    error = event.message
            settle(final_state, checkpoint=checkpoint, error=error)
            return 0 if final_state == "completed" else 1
        except Exception as exc:  # noqa: BLE001 — last-resort net under the stream
            emit(RunFailed(error_code=getattr(exc, "code", "backend_error"),
                           message=str(exc)))
            settle("failed", error=str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
