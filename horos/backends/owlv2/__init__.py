"""OWLv2 backend — the only place allowed to import `transformers` (R1).

P0 status: adapter skeleton. The real zero-shot autolabel implementation is
E3-T1 and lands with P1.
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
)
from horos.errors import BackendError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

_NOT_YET = (
    "OWLv2 {op} is not implemented yet (E3-T1 lands with the autolabel phase P1)."
)


class OWLv2Backend(ModelBackend):
    family = "owlv2"

    def __init__(self, info: "ModelInfo", *, device: str | None = None):
        super().__init__(info, device=device)
        self._model = None

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        raise BackendError(
            "OWLv2 is a zero-shot autolabeling model; it is not trainable in horos.",
            backend=self.family,
        )

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        raise BackendError(_NOT_YET.format(op="inference"), backend=self.family)

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        raise BackendError(_NOT_YET.format(op="batch inference"), backend=self.family)

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        raise BackendError(
            "OWLv2 export is not supported; it is an autolabeling backend only.",
            backend=self.family,
        )
