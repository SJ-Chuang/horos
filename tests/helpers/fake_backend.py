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
        import time

        from horos.backends.base import RunFailed

        yield RunStarted(total=spec.epochs, config={"epochs": spec.epochs})
        # spec.extra is the expert passthrough; the fakes read pacing and
        # failure switches from it so subprocess tests can steer behavior.
        oom_above = spec.extra.get("oom_above_batch")
        if oom_above is not None and spec.batch_size > int(oom_above):
            yield RunFailed(
                error_code="backend_out_of_memory",
                message=f"simulated OOM at batch {spec.batch_size}",
            )
            return
        for epoch in range(spec.epochs):
            if spec.extra.get("sleep_per_epoch"):
                time.sleep(float(spec.extra["sleep_per_epoch"]))
            yield ProgressUpdated(current=epoch + 1, total=spec.epochs, phase="train")
            yield MetricsUpdated(step=epoch + 1, metrics={"loss": 1.0 / (epoch + 1)})
        if spec.extra.get("fail"):
            yield RunFailed(error_code="backend_error", message="simulated failure")
            return
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = spec.output_dir / "best.fake"
        checkpoint.write_bytes(b"fake-weights")
        result = {"checkpoint": str(checkpoint), "batch_size": spec.batch_size}
        if spec.resume_from is not None:
            result["resumed_from"] = str(spec.resume_from)
        yield RunCompleted(result=result)

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


class FakeOpenVocabBackend(FakeBackend):
    """Deterministic open-vocabulary backend for the autolabel tests (E3).

    Default behavior: one detection per configured prompt, boxes spread along
    x, scores 0.9, 0.8, ... per prompt index. `score_by_name` overrides the
    base score per image file name so ranking tests can control uncertainty.
    """

    family = "fake-openvocab"

    def __init__(self, info=None, *, device=None, score_by_name=None):
        if info is not None:
            super().__init__(info, device=device)
        else:
            self.info = None
            self.device = device
        self.prompts: list[str] = []
        self.score_by_name = score_by_name or {}
        self.calls: list[str] = []

    def configure_prompts(self, prompts):
        self.prompts = list(prompts)

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        self.calls.append(Path(image).name)
        base = self.score_by_name.get(Path(image).name, 0.9)
        instances = [
            PredictedInstance(
                bbox=(10.0 + 30.0 * i, 10.0, 20.0, 20.0),
                score=max(base - 0.1 * i, 0.01),
                category_id=i,
            )
            for i in range(len(self.prompts))
        ]
        return ImagePrediction(image=str(image), instances=instances)


class FakeRefinerBackend:
    """Deterministic box->polygon refiner for the autolabel polygon tests:
    each box becomes a triangle inside it; boxes named in `fail_indices`
    return None (mask failure -> the box must survive as a box)."""

    family = "fake-refiner"

    def __init__(self, *, fail_indices=()):
        self.fail_indices = set(fail_indices)
        self.calls: list[tuple[str, int]] = []

    def polygons_for_boxes(self, image, boxes):
        self.calls.append((Path(image).name, len(boxes)))
        out = []
        for i, (x, y, w, h) in enumerate(boxes):
            if i in self.fail_indices:
                out.append(None)
            else:
                out.append([x, y, x + w, y, x + w / 2, y + h])
        return out


def _spawn_probe_child(marker_path: str) -> None:
    """Runs in a spawn-context child process — must be module-level picklable."""
    Path(marker_path).write_text("spawned-child-ran", encoding="utf-8")


class SpawnProbeBackend(FakeBackend):
    """Mimics a DataLoader worker: starts a spawn-context child during
    training (E5-T6b). If the training worker were not `__main__`-guarded,
    the spawn re-import would re-execute it and the run would corrupt itself.
    """

    family = "fake-spawn"

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        import multiprocessing

        yield RunStarted(total=spec.epochs, config={"epochs": spec.epochs})
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        marker = spec.output_dir / "spawn_marker.txt"
        ctx = multiprocessing.get_context("spawn")  # the Windows/macOS default
        child = ctx.Process(target=_spawn_probe_child, args=(str(marker),))
        child.start()
        child.join(30)
        if not marker.is_file():
            from horos.backends.base import RunFailed

            yield RunFailed(error_code="backend_error",
                            message="spawned child never ran")
            return
        checkpoint = spec.output_dir / "best.fake"
        checkpoint.write_bytes(b"fake-weights")
        yield RunCompleted(result={"checkpoint": str(checkpoint)})
