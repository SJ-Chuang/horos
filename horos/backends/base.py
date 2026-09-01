"""Abstract backend interface (E4-T2) and the shared event types (E4-T3, R4).

Every model dependency lives behind subclasses of `ModelBackend` in
`horos/backends/<family>/`. This module itself must import no ML library —
it is imported by the lazy loader before any backend is resolved.

R4: long-running work reports progress through these event types and nothing
else. Backends must not invent their own reporting format.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, TypeAdapter

from horos.errors import BackendError, BackendOutOfMemoryError, HorosError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

# ---------------------------------------------------------------- inference payloads


class PredictedInstance(BaseModel):
    """One predicted object. bbox is absolute-pixel COCO xywh.

    `category_id` is in the backend's own label space (e.g. rfdetr emits
    0-based indices into its class list); `category_name` is the portable
    identity — callers matching predictions against a dataset must map by
    name, never by raw id."""

    bbox: tuple[float, float, float, float]
    score: float
    category_id: int
    category_name: str | None = None
    segmentation: list[list[float]] | None = None


class ImagePrediction(BaseModel):
    image: str  # path or identifier the caller passed in
    width: int | None = None
    height: int | None = None
    instances: list[PredictedInstance] = Field(default_factory=list)


# --------------------------------------------------------------------------- events


class _EventBase(BaseModel):
    ts: float = Field(default_factory=time.time)
    run_id: str | None = None


class RunStarted(_EventBase):
    type: Literal["started"] = "started"
    total: int | None = None  # total units of work, if known up front
    config: dict[str, Any] = Field(default_factory=dict)


class ProgressUpdated(_EventBase):
    type: Literal["progress"] = "progress"
    current: int
    total: int | None = None
    phase: str = ""  # e.g. "epoch 2/10", "downloading weights"
    message: str = ""


class MetricsUpdated(_EventBase):
    type: Literal["metrics"] = "metrics"
    step: int
    metrics: dict[str, float]


class WarningRaised(_EventBase):
    type: Literal["warning"] = "warning"
    message: str


class PredictionReady(_EventBase):
    """One item of a batch inference finished; payload is an ImagePrediction dump."""

    type: Literal["prediction"] = "prediction"
    index: int
    prediction: ImagePrediction


class RunCompleted(_EventBase):
    type: Literal["completed"] = "completed"
    result: dict[str, Any] = Field(default_factory=dict)


class RunFailed(_EventBase):
    type: Literal["failed"] = "failed"
    error_code: str = "backend_error"
    message: str = ""


Event = Annotated[
    RunStarted
    | ProgressUpdated
    | MetricsUpdated
    | WarningRaised
    | PredictionReady
    | RunCompleted
    | RunFailed,
    Field(discriminator="type"),
]

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(data: dict[str, Any] | str | bytes) -> Event:
    """Parse a serialized event back into its typed form (for SSE / JSONL)."""
    if isinstance(data, (str, bytes)):
        return _event_adapter.validate_json(data)
    return _event_adapter.validate_python(data)


def dump_event(event: Event) -> str:
    """One-line JSON, suitable for JSONL streams and SSE data fields."""
    return event.model_dump_json()


# ------------------------------------------------------------------------- specs


class TrainSpec(BaseModel):
    """Backend-neutral training request. Backends map these onto their own knobs
    and must reject (not ignore) anything they cannot honor."""

    dataset_dir: Path
    output_dir: Path
    epochs: int
    batch_size: int
    resolution: int | None = None
    device: str | None = None  # resolved via backends/device.py when None
    seed: int | None = None
    resume_from: Path | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


ExportFormat = Literal["onnx", "tensorrt", "tflite"]


class ExportSpec(BaseModel):
    format: ExportFormat
    output_dir: Path
    options: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- interface


class ModelBackend(ABC):
    """The only surface upper layers may program against (R1).

    Construction must be cheap and must not load weights or import the heavy
    dependency yet — first use does (R1b is about `import horos`, but keeping
    construction light keeps registry-driven UIs snappy too).
    """

    family: ClassVar[str]

    def __init__(
        self,
        info: ModelInfo,
        *,
        device: str | None = None,
        checkpoint: Path | None = None,
    ):
        self.info = info
        self.device = device
        # When set, inference loads these trained weights instead of the
        # model's pretrained defaults. Ignored by backends that never train.
        self.checkpoint = checkpoint

    # -- training ------------------------------------------------------------
    @abstractmethod
    def train(self, spec: TrainSpec) -> Iterator[Event]:
        """Run training, yielding events (R4). Final event is RunCompleted with
        result["checkpoint"] pointing at the best weights, or RunFailed."""

    # -- inference -----------------------------------------------------------
    @abstractmethod
    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        """Single-image inference, synchronous."""

    @abstractmethod
    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        """Batch inference as an event stream: ProgressUpdated + PredictionReady
        per image, terminated by RunCompleted/RunFailed."""

    # -- export --------------------------------------------------------------
    @abstractmethod
    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        """Export a trained checkpoint. RunCompleted carries result["artifact"]."""


class OpenVocabularyBackend(ModelBackend):
    """Zero-shot detectors driven by text prompts (OWLv2 and successors).

    `category_id` in predictions indexes into the prompt list passed to
    `configure_prompts` — the caller owns the prompt→class mapping (E3-T2)."""

    @abstractmethod
    def configure_prompts(self, prompts: list[str]) -> None:
        """Set the text prompts used by subsequent infer_one/infer_batch calls."""


class BoxToMaskBackend(ModelBackend):
    """Promptable segmenters that turn detection boxes into masks (SAM and
    successors) — the polygon output path of autolabel."""

    @abstractmethod
    def polygons_for_boxes(
        self, image: Path, boxes: list[tuple[float, float, float, float]]
    ) -> list[list[float] | None]:
        """One flat [x1, y1, x2, y2, ...] polygon per COCO-xywh box, in the
        same order; None where no usable mask came back."""


# ------------------------------------------------------------------ error bridge


@contextmanager
def translate_backend_errors(backend: str):
    """Wrap backend-library calls so upper layers only ever see HorosError (E4-T5).

    OOM is recognized structurally where possible and by message otherwise —
    torch.cuda.OutOfMemoryError cannot be imported here without violating R1b.
    """
    try:
        yield
    except HorosError:
        raise
    except MemoryError as exc:
        raise BackendOutOfMemoryError(
            f"[{backend}] out of memory: {exc}", backend=backend
        ) from exc
    except Exception as exc:  # noqa: BLE001 — the whole point is to catch everything
        message = str(exc)
        if "out of memory" in message.lower():
            raise BackendOutOfMemoryError(
                f"[{backend}] out of memory: {message}", backend=backend
            ) from exc
        raise BackendError(
            f"[{backend}] {type(exc).__name__}: {message}", backend=backend
        ) from exc
