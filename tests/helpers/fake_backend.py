"""A fully functional fake backend used to test the interface contract (E4-T2)
and lazy loading (E4-T11) without any ML dependency."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

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
    RunStarted,
    TrainSpec,
    translate_backend_errors,
)

IMPORTED_MARKER = {"count": 0}
IMPORTED_MARKER["count"] += 1  # increments once per real import of this module


class FakeBackend(ModelBackend):
    family = "fake"

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        yield RunStarted(total=spec.epochs, config={"epochs": spec.epochs})
        for epoch in range(spec.epochs):
            yield ProgressUpdated(current=epoch + 1, total=spec.epochs, phase="train")
            yield MetricsUpdated(step=epoch + 1, metrics={"loss": 1.0 / (epoch + 1)})
        yield RunCompleted(result={"checkpoint": str(spec.output_dir / "best.fake")})

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        return ImagePrediction(
            image=str(image),
            instances=[
                PredictedInstance(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, category_id=0)
            ],
        )

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        images = list(images)
        yield RunStarted(total=len(images))
        for i, image in enumerate(images):
            yield PredictionReady(index=i, prediction=self.infer_one(image))
            yield ProgressUpdated(current=i + 1, total=len(images), phase="infer")
        yield RunCompleted(result={"count": len(images)})

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        yield RunStarted()
        yield RunCompleted(
            result={"artifact": str(spec.output_dir / f"model.{spec.format}")}
        )


class ExplodingBackend(FakeBackend):
    """Raises a foreign exception inside the translation wrapper (E4-T5)."""

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        with translate_backend_errors(self.family):
            raise RuntimeError("simulated library failure")

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        with translate_backend_errors(self.family):
            raise RuntimeError("CUDA error: out of memory (simulated)")
