"""RF-DETR backend — the only place allowed to import `rfdetr` (R1).

P0 status: adapter skeleton. Construction and environment checking work; the
train/infer/export bodies are E4-T4 and land with P2 (training). Each method
already goes through `translate_backend_errors` so upper layers never see a
raw rfdetr exception.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from horos.backends.base import (
    Event,
    ExportSpec,
    ImagePrediction,
    ModelBackend,
    TrainSpec,
    translate_backend_errors,
)
from horos.errors import BackendError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

_NOT_YET = (
    "RF-DETR {op} is not implemented yet (E4-T4 lands with the training phase P2). "
    "The backend interface and registry entry are in place; only the adapter body "
    "is pending."
)


class RFDETRBackend(ModelBackend):
    family = "rfdetr"

    def __init__(self, info: "ModelInfo", *, device: str | None = None):
        super().__init__(info, device=device)
        self._model = None  # loaded lazily on first real use

    def _load(self):
        if self._model is not None:
            return self._model
        with translate_backend_errors(self.family):
            from horos.backends import env

            env.check_environment()
            import rfdetr  # noqa: F401, PLC0415 — the sanctioned import site

            raise BackendError(_NOT_YET.format(op="model loading"), backend=self.family)

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        raise BackendError(_NOT_YET.format(op="training"), backend=self.family)

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        raise BackendError(_NOT_YET.format(op="inference"), backend=self.family)

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        raise BackendError(_NOT_YET.format(op="batch inference"), backend=self.family)

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        raise BackendError(_NOT_YET.format(op="export"), backend=self.family)
