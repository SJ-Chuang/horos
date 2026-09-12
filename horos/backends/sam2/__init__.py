"""SAM 2.1 backend — interactive point/box segmentation (SAM-T1).

`facebook/sam2.1-hiera-tiny` / `-small` (code and weights Apache 2.0) through
transformers' Sam2Model / Sam2Processor: no new dependency. The encoder runs
once per image (`embed`, ~0.2 s on a desktop GPU) and every click then costs
only the prompt decoder (tens of milliseconds), which is what makes
click-to-segment usable in the annotator. Like SAM v1 it is a segmenter, not
a detector: it never proposes objects on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from horos.backends import weights
from horos.backends.base import (
    MODEL_LOAD_LOCK,
    Event,
    ExportSpec,
    ImageEmbedding,
    ImagePrediction,
    PromptableSegmenter,
    SegmentPrompt,
    SegmentResult,
    TrainSpec,
    translate_backend_errors,
)
from horos.backends.sam.promptable import TransformersPromptableMixin
from horos.errors import BackendError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

_NOT_A_DETECTOR = (
    "SAM 2.1 segments what you point at; it does not {op} on its own."
)


class SAM2Backend(TransformersPromptableMixin, PromptableSegmenter):
    family = "sam2"
    _needs_reshaped_sizes = False

    def __init__(
        self,
        info: ModelInfo,
        *,
        device: str | None = None,
        checkpoint: Path | None = None,
    ):
        super().__init__(info, device=device, checkpoint=checkpoint)
        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is not None:
            return
        with MODEL_LOAD_LOCK, translate_backend_errors(self.family):
            if self._model is not None:  # another thread loaded it while we waited
                return
            import torch  # noqa: F401 — resolved lazily on first use (R1b)
            from transformers import Sam2Model, Sam2Processor

            from horos.backends.device import select_device

            self.device = select_device(self.device).torch_device
            cache = str(weights.hf_cache_dir())
            self._processor = Sam2Processor.from_pretrained(self.info.hf_id, cache_dir=cache)
            self._model = Sam2Model.from_pretrained(self.info.hf_id, cache_dir=cache).to(
                self.device
            )
            self._model.eval()

    def embed(self, image: Path) -> ImageEmbedding:
        with translate_backend_errors(self.family):
            return TransformersPromptableMixin.embed(self, image)

    def segment(self, embedding: ImageEmbedding, prompt: SegmentPrompt) -> SegmentResult:
        with translate_backend_errors(self.family):
            return TransformersPromptableMixin.segment(self, embedding, prompt)

    # -- not a detector -------------------------------------------------------
    def train(self, spec: TrainSpec) -> Iterator[Event]:
        raise BackendError(_NOT_A_DETECTOR.format(op="train"), backend=self.family)

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        raise BackendError(_NOT_A_DETECTOR.format(op="detect"), backend=self.family)

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        raise BackendError(_NOT_A_DETECTOR.format(op="detect"), backend=self.family)

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        raise BackendError(_NOT_A_DETECTOR.format(op="export"), backend=self.family)
