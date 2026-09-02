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

# EMA factor applied to the monitored mAP before best-checkpoint comparison
# under the "smoothed_map" criterion (rfdetr's own smooth_alpha knob). Chosen
# so one noisy validation spike on a tiny valid split cannot lock in a bad
# checkpoint, while a real improvement still wins within ~2 epochs.
_SMOOTH_ALPHA = 0.6


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
    if spec.checkpoint_criterion == "smoothed_map":
        kwargs["smooth_alpha"] = _SMOOTH_ALPHA
    # "loss" has no train kwarg — the checkpoint callback is re-pointed after
    # build_trainer instead (see _repoint_checkpoint_monitor)
    kwargs.update(spec.extra)
    return kwargs


def _detections_to_instances(
    detections: Any, class_names: list[str] | None = None
) -> list[PredictedInstance]:
    """supervision.Detections (xyxy) → PredictedInstance list (COCO xywh).

    Fine-tuned rfdetr emits 0-based indices into its class list — NOT the
    training dataset's COCO category ids. The name is the portable identity,
    so it is attached to every instance (see PredictedInstance)."""
    instances: list[PredictedInstance] = []
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        label = int(class_id[i]) if class_id is not None else 0
        name = None
        if class_names is not None and 0 <= label < len(class_names):
            name = class_names[label]
        instances.append(
            PredictedInstance(
                bbox=(x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)),
                score=float(confidence[i]) if confidence is not None else 1.0,
                category_id=label,
                category_name=name,
            )
        )
    return instances


def _epoch_metrics(callback_metrics: Any, *, train_side: bool) -> dict[str, float]:
    """One epoch's Lightning callback_metrics, split into the train-side or
    val-side slice.

    Lightning runs validation BEFORE `on_train_epoch_end`, and the epoch-
    aggregated `train/*` values are only published in that later hook — so a
    relay that reads everything at validation end reports train metrics one
    epoch late, misses epoch 0, and silently drops the final epoch. Each side
    must be captured in its own hook.
    """
    picked: dict[str, float] = {}
    for key, value in callback_metrics.items():
        if key.startswith("train/") != train_side:
            continue
        try:
            picked[key] = float(value)
        except (TypeError, ValueError):
            continue
    return picked


def _repoint_checkpoint_monitor(callback: Any, monitor: str, mode: str) -> None:
    """Re-target an already-constructed Lightning ModelCheckpoint.

    rfdetr 1.9.4 hardcodes the best-model monitor to val mAP inside
    build_trainer; the "loss" criterion needs val/loss with mode=min. Mode
    lives in the parent's name-mangled init helper (kth_value must be reset
    alongside), so it is re-run here — acceptable against a pinned version
    (R5)."""
    callback.monitor = monitor
    callback._ModelCheckpoint__init_monitor_mode(mode)  # noqa: SLF001


class _BestTracker:
    """Answers "which epoch do the saved best weights come from" by watching
    the actual BestModelCallback state each epoch — exact under every
    criterion (raw, smoothed, loss) and across the regular/EMA tracks,
    without re-implementing any comparison logic.
    """

    def __init__(self) -> None:
        self.callback: Any = None
        self._last_regular: float | None = None
        self._last_ema: float | None = None
        self._regular_epoch: int | None = None
        self._ema_epoch: int | None = None
        self._last_emitted: tuple[int, bool] | None = None

    def observe(self, epoch: int) -> dict[str, float] | None:
        """Metrics to publish when the best checkpoint changed, else None."""
        cb = self.callback
        if cb is None:
            return None
        score = getattr(cb, "best_model_score", None)
        regular = float(score) if score is not None else None
        ema = float(getattr(cb, "_best_ema", 0.0) or 0.0)
        if regular is not None and regular != self._last_regular:
            self._last_regular, self._regular_epoch = regular, epoch
        if ema and ema != self._last_ema:
            self._last_ema, self._ema_epoch = ema, epoch
        # mirror on_fit_end's winner rule: EMA wins on strict >, compared
        # against the raw (un-smoothed) regular value when smoothing is on
        raw_regular = regular if not getattr(cb, "_smooth_alpha", 0.0) else None
        if raw_regular is None:
            raw_regular = float(getattr(cb, "_best_raw_regular", 0.0) or 0.0)
        is_ema = (
            getattr(cb, "_monitor_ema", None) is not None
            and self._ema_epoch is not None
            and ema > raw_regular
        )
        best_epoch = self._ema_epoch if is_ema else self._regular_epoch
        if best_epoch is None or (best_epoch, is_ema) == self._last_emitted:
            return None
        self._last_emitted = (best_epoch, is_ema)
        return {"best/epoch": float(best_epoch), "best/is_ema": float(is_ema)}


def _default_weights_filename(model_class: Any) -> str | None:
    """The variant's published default `pretrain_weights` filename; None when
    no usable string default can be found (then rfdetr's own paths handle it).

    Most variants declare `_model_config_class`; RFDETRLarge instead overrides
    `get_model_config()` (its declared class attribute is the bare ModelConfig,
    default None), so the config default is checked first and the unbound
    method — which ignores `self` in 1.9.4 (R5) — is the fallback."""
    config_class = getattr(model_class, "_model_config_class", None)
    fields = getattr(config_class, "model_fields", None) or {}
    default = getattr(fields.get("pretrain_weights"), "default", None)
    if isinstance(default, str):
        return default
    try:
        config = model_class.get_model_config(model_class)
        default = getattr(config, "pretrain_weights", None)
    except Exception:  # noqa: BLE001 — best effort; downloading stays rfdetr's job
        return None
    return default if isinstance(default, str) else None


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

    def _pretrained_weights_events(self) -> Iterator[Event]:
        """Pre-fetch the variant's pretrained weights with R4 progress events.

        rfdetr downloads its default weights silently inside model
        construction, so the first run of each size sat on a bare `started`
        event for minutes and looked hung. Fetching the same file into
        rfdetr's own cache first (same path, MD5-checked) makes rfdetr skip
        its download, and horos owns the progress stream."""
        from rfdetr.assets.model_weights import ModelWeights, get_model_cache_dir

        filename = _default_weights_filename(self._model_class())
        if filename is None or Path(filename).is_absolute():
            return
        target = Path(get_model_cache_dir()) / filename
        if target.is_file():
            return
        asset = ModelWeights.from_filename(filename)
        if asset is None:
            return  # unknown to the registry — leave rfdetr's fallbacks to it

        from rfdetr.utilities.files import _validate_file_md5

        from horos.backends.weights import download_events

        path = yield from download_events(
            asset.url,
            filename=filename,
            dest_dir=target.parent,
            label=f"downloading {filename}",
        )
        if asset.md5_hash and not _validate_file_md5(str(path), asset.md5_hash):
            path.unlink(missing_ok=True)
            raise BackendError(
                f"Downloaded weights {filename} failed MD5 validation; the "
                "corrupt file was removed — check the network and retry.",
                backend=self.family,
            )

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

                yield from self._pretrained_weights_events()
                model = self._model_class()(device=kwargs["device"])
                events: queue_mod.Queue = queue_mod.Queue()
                tracker = _BestTracker()

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
                        metrics = _epoch_metrics(
                            trainer.callback_metrics, train_side=False
                        )
                        if metrics:
                            events.put(
                                MetricsUpdated(
                                    step=trainer.current_epoch, metrics=metrics
                                )
                            )

                    def on_train_epoch_end(self, trainer, module):  # noqa: ANN001
                        # train/* is only published in this hook (see
                        # _epoch_metrics) — capture it here, on its own epoch
                        metrics = _epoch_metrics(
                            trainer.callback_metrics, train_side=True
                        )
                        if metrics:
                            events.put(
                                MetricsUpdated(
                                    step=trainer.current_epoch, metrics=metrics
                                )
                            )
                        # checkpointing ran during validation (before this
                        # hook) — the tracker now sees the settled best state
                        best = tracker.observe(trainer.current_epoch)
                        if best:
                            events.put(
                                MetricsUpdated(
                                    step=trainer.current_epoch, metrics=best
                                )
                            )
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
                    best_cb = next(
                        (
                            c
                            for c in trainer.callbacks
                            if isinstance(c, rf_training.BestModelCallback)
                        ),
                        None,
                    )
                    if best_cb is not None:
                        if spec.checkpoint_criterion == "loss":
                            _repoint_checkpoint_monitor(best_cb, "val/loss", "min")
                            # the EMA track still measures mAP; comparing a
                            # loss against an mAP for best_total is meaningless
                            best_cb._monitor_ema = None  # noqa: SLF001
                        tracker.callback = best_cb
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
            class_names = list(getattr(model, "class_names", None) or [])
            return ImagePrediction(
                image=str(image),
                width=width,
                height=height,
                instances=_detections_to_instances(detections, class_names or None),
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
