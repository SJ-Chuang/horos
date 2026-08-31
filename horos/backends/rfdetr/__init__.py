"""RF-DETR backend (E4-T4) — the only place allowed to import `rfdetr` (R1).

Training goes through rfdetr's PyTorch Lightning stack (the `rfdetr[train]`
extra). rfdetr exposes no user callback hook in 1.9.4, so per-epoch progress
and metrics are captured by wrapping `rfdetr.training.build_trainer` and
appending one Lightning callback that relays into horos's R4 event types —
acceptable because R5 pins the version exactly.

All ML imports happen lazily on first use (R1b).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from horos.backends.base import (
    Event,
    ExportSpec,
    ImagePrediction,
    MetricsUpdated,
    ModelBackend,
    PredictedInstance,
    PredictionReady,
    ProgressUpdated,
    RunCompleted,
    RunFailed,
    RunStarted,
    TrainSpec,
    translate_backend_errors,
)
from horos.errors import BackendError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

_MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}

# Best-first order among the files rfdetr training writes to output_dir.
_CHECKPOINT_PREFERENCE = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "last.ckpt",
)


def _train_kwargs(spec: TrainSpec) -> dict[str, Any]:
    """Map the backend-neutral TrainSpec onto rfdetr.train() keyword arguments.

    horos owns progress reporting (R4), so rfdetr's own loggers and progress
    bar are off by default; `spec.extra` is applied last so an expert override
    wins over every derived value (E5-S5).
    """
    kwargs: dict[str, Any] = {
        "dataset_dir": str(spec.dataset_dir),
        "output_dir": str(spec.output_dir),
        "epochs": spec.epochs,
        "batch_size": spec.batch_size,
        "tensorboard": False,
        "progress_bar": None,
        "run_test": False,
        "early_stopping": False,
        "log_per_class_metrics": False,
    }
    if spec.resolution is not None:
        kwargs["resolution"] = spec.resolution
    if spec.device is not None:
        kwargs["device"] = spec.device
    if spec.seed is not None:
        kwargs["seed"] = spec.seed
    if spec.resume_from is not None:
        kwargs["resume"] = str(spec.resume_from)
    kwargs.update(spec.extra)
    return kwargs


def _detections_to_instances(detections: Any) -> list[PredictedInstance]:
    """supervision.Detections (xyxy) → PredictedInstance list (COCO xywh)."""
    instances: list[PredictedInstance] = []
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        instances.append(
            PredictedInstance(
                bbox=(x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)),
                score=float(confidence[i]) if confidence is not None else 1.0,
                category_id=int(class_id[i]) if class_id is not None else 0,
            )
        )
    return instances


def _best_checkpoint(output_dir: Path) -> Path | None:
    for name in _CHECKPOINT_PREFERENCE:
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    return None


class RFDETRBackend(ModelBackend):
    family = "rfdetr"

    def __init__(
        self,
        info: ModelInfo,
        *,
        device: str | None = None,
        checkpoint: Path | None = None,
    ):
        super().__init__(info, device=device, checkpoint=checkpoint)
        self._model = None  # loaded lazily on first real use

    # ------------------------------------------------------------------ model
    def _resolve_device(self) -> str:
        from horos.backends.device import select_device

        prefer = self.device.partition(":")[0] if self.device else None
        return select_device(prefer).torch_device  # type: ignore[arg-type]

    def _model_class(self):
        import rfdetr  # noqa: PLC0415 — the sanctioned import site (R1)

        class_name = _MODEL_CLASSES.get(self.info.key)
        if class_name is None:
            raise BackendError(
                f"No RF-DETR class mapping for model '{self.info.key}' "
                f"(rfdetr 1.9.4 ships {sorted(_MODEL_CLASSES)})",
                backend=self.family,
            )
        return getattr(rfdetr, class_name)

    def _load(self):
        if self._model is not None:
            return self._model
        with translate_backend_errors(self.family):
            from horos.backends import env

            env.check_environment()
            device = self._resolve_device()
            if self.checkpoint is not None:
                from rfdetr.detr import RFDETR

                # trust_checkpoint: these are horos's own training outputs
                self._model = RFDETR.from_checkpoint(
                    self.checkpoint, trust_checkpoint=True, device=device
                )
            else:
                self._model = self._model_class()(device=device)
        return self._model

    # --------------------------------------------------------------- training
    def train(self, spec: TrainSpec) -> Iterator[Event]:
        import queue as queue_mod
        import threading

        kwargs = _train_kwargs(spec)
        if kwargs.get("device") is None:
            kwargs["device"] = self._resolve_device()
        yield RunStarted(
            config={k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
        )

        try:
            with translate_backend_errors(self.family):
                try:
                    import pytorch_lightning as pl
                    import rfdetr.training as rf_training
                except ImportError as exc:
                    raise BackendError(
                        "RF-DETR training needs the training extras: "
                        'pip install "rfdetr[train]==1.9.4" '
                        "(or run: horos doctor --fix)",
                        backend=self.family,
                    ) from exc

                model = self._model_class()(device=kwargs["device"])
                events: queue_mod.Queue = queue_mod.Queue()

                class _EventRelay(pl.Callback):
                    def on_train_epoch_start(self, trainer, module):  # noqa: ANN001
                        events.put(
                            ProgressUpdated(
                                current=trainer.current_epoch,
                                total=trainer.max_epochs,
                                phase=(
                                    f"epoch {trainer.current_epoch + 1}"
                                    f"/{trainer.max_epochs}"
                                ),
                            )
                        )

                    def on_validation_epoch_end(self, trainer, module):  # noqa: ANN001
                        if trainer.sanity_checking:
                            return
                        metrics = {
                            key: float(value)
                            for key, value in trainer.callback_metrics.items()
                        }
                        if metrics:
                            events.put(
                                MetricsUpdated(
                                    step=trainer.current_epoch, metrics=metrics
                                )
                            )

                    def on_train_epoch_end(self, trainer, module):  # noqa: ANN001
                        events.put(
                            ProgressUpdated(
                                current=trainer.current_epoch + 1,
                                total=trainer.max_epochs,
                                phase="epoch completed",
                            )
                        )

                original_build_trainer = rf_training.build_trainer

                def build_trainer_with_relay(*args, **kw):  # noqa: ANN002, ANN003
                    trainer = original_build_trainer(*args, **kw)
                    trainer.callbacks.append(_EventRelay())
                    return trainer

                failure: list[BaseException] = []

                def run_training() -> None:
                    try:
                        model.train(**kwargs)
                    except BaseException as exc:  # noqa: BLE001 — relayed to the stream
                        failure.append(exc)

                rf_training.build_trainer = build_trainer_with_relay
                try:
                    worker = threading.Thread(
                        target=run_training, name="horos-rfdetr-train"
                    )
                    worker.start()
                    while worker.is_alive() or not events.empty():
                        try:
                            yield events.get(timeout=1.0)
                        except queue_mod.Empty:
                            continue
                    worker.join()
                finally:
                    rf_training.build_trainer = original_build_trainer

                if failure:
                    raise failure[0]

                checkpoint = _best_checkpoint(Path(kwargs["output_dir"]))
                if checkpoint is None:
                    raise BackendError(
                        "Training finished but no checkpoint was written to "
                        f"{kwargs['output_dir']}",
                        backend=self.family,
                    )
        except Exception as exc:  # noqa: BLE001 — R4: the stream terminates itself
            code = getattr(exc, "code", "backend_error")
            yield RunFailed(error_code=code, message=str(exc))
            return

        yield RunCompleted(result={"checkpoint": str(checkpoint)})

    # -------------------------------------------------------------- inference
    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        model = self._load()
        with translate_backend_errors(self.family):
            from PIL import Image

            with Image.open(image) as im:
                width, height = im.size
            detections = model.predict(str(image), threshold=threshold)
            return ImagePrediction(
                image=str(image),
                width=width,
                height=height,
                instances=_detections_to_instances(detections),
            )

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        paths = [Path(p) for p in images]
        yield RunStarted(total=len(paths), config={"model": self.info.key})
        for index, path in enumerate(paths):
            prediction = self.infer_one(path, threshold=threshold)
            yield PredictionReady(index=index, prediction=prediction)
            yield ProgressUpdated(current=index + 1, total=len(paths), phase="inference")
        yield RunCompleted(result={"images": len(paths)})

    # ----------------------------------------------------------------- export
    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        raise BackendError(
            "RF-DETR export is not implemented yet (E8 lands with the deployment "
            "phase P4).",
            backend=self.family,
        )
