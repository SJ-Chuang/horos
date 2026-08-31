"""E4-T4: RF-DETR adapter unit tests — the pure mapping logic, no ML runtime.

The full training path is exercised end-to-end by test_train_e2e.py; these
tests pin down the spec→rfdetr translation and result conversion, which is
where a silent regression would corrupt runs without failing loudly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from horos.backends.base import TrainSpec
from horos.backends.rfdetr import (
    _MODEL_CLASSES,
    _best_checkpoint,
    _detections_to_instances,
    _train_kwargs,
)
from horos.core.registry import list_models


def _spec(**overrides):
    defaults = dict(
        dataset_dir=Path("/data/ds"),
        output_dir=Path("/data/out"),
        epochs=5,
        batch_size=2,
    )
    defaults.update(overrides)
    return TrainSpec(**defaults)


def test_every_registered_rfdetr_model_has_a_class_mapping():
    keys = {m.key for m in list_models() if m.family == "rfdetr"}
    assert keys == set(_MODEL_CLASSES)


def test_train_kwargs_maps_spec_fields():
    kwargs = _train_kwargs(
        _spec(resolution=384, device="cpu", seed=7, resume_from=Path("/ckpt.pth"))
    )
    assert kwargs["dataset_dir"] == str(Path("/data/ds"))
    assert kwargs["output_dir"] == str(Path("/data/out"))
    assert kwargs["epochs"] == 5 and kwargs["batch_size"] == 2
    assert kwargs["resolution"] == 384 and kwargs["device"] == "cpu"
    assert kwargs["seed"] == 7 and kwargs["resume"] == str(Path("/ckpt.pth"))


def test_train_kwargs_omits_unset_optionals_and_silences_rfdetr_reporting():
    kwargs = _train_kwargs(_spec())
    for absent in ("resolution", "device", "seed", "resume"):
        assert absent not in kwargs
    # horos owns progress reporting (R4)
    assert kwargs["tensorboard"] is False
    assert kwargs["progress_bar"] is None
    assert kwargs["early_stopping"] is False


def test_train_kwargs_extra_overrides_win():
    kwargs = _train_kwargs(
        _spec(extra={"epochs": 99, "tensorboard": True, "lr": 1e-5})
    )
    assert kwargs["epochs"] == 99  # expert override beats the derived value (E5-S5)
    assert kwargs["tensorboard"] is True
    assert kwargs["lr"] == 1e-5


def test_detections_convert_to_coco_xywh():
    detections = SimpleNamespace(
        xyxy=[(10.0, 20.0, 30.0, 60.0), (0.0, 0.0, 5.0, 5.0)],
        confidence=[0.9, 0.4],
        class_id=[2, 0],
    )
    instances = _detections_to_instances(detections)
    assert instances[0].bbox == (10.0, 20.0, 20.0, 40.0)
    assert instances[0].score == 0.9 and instances[0].category_id == 2
    assert instances[1].bbox == (0.0, 0.0, 5.0, 5.0)


def test_detections_tolerate_missing_confidence_and_class():
    detections = SimpleNamespace(
        xyxy=[(1.0, 1.0, 2.0, 2.0)], confidence=None, class_id=None
    )
    (instance,) = _detections_to_instances(detections)
    assert instance.score == 1.0 and instance.category_id == 0


def test_best_checkpoint_preference_order(tmp_path):
    assert _best_checkpoint(tmp_path) is None
    (tmp_path / "last.ckpt").touch()
    assert _best_checkpoint(tmp_path).name == "last.ckpt"
    (tmp_path / "checkpoint_best_ema.pth").touch()
    assert _best_checkpoint(tmp_path).name == "checkpoint_best_ema.pth"
    (tmp_path / "checkpoint_best_total.pth").touch()
    assert _best_checkpoint(tmp_path).name == "checkpoint_best_total.pth"
