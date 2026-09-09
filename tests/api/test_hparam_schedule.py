"""E5: LR schedule, augmentation and mosaic derivation rules.

These rules exist because of a concrete failure mode: constant full LR plus a
flip-only augmentation pipeline drives small-dataset fine-tunes into a
catastrophic late-run collapse (train loss falls, val mAP goes to ~0).
"""

from __future__ import annotations

from test_hparam_derive import _plan, _stats

from horos.api.hparams import _AUG_HEAVY, _AUG_STANDARD


def test_scheduler_is_cosine_with_floor_for_every_size():
    for n in (30, 300, 5000):
        plan = _plan(_stats(num_images=n))
        assert plan.values["lr_scheduler"] == "cosine"
        assert plan.values["lr_scheduler_kwargs"] == {"min_factor": 0.01}


def test_lr_halved_below_100_images():
    assert _plan(_stats(num_images=74)).values["lr"] == 5e-5
    assert _plan(_stats(num_images=300)).values["lr"] == 1e-4


def test_warmup_three_epochs_on_small_datasets():
    assert _plan(_stats(num_images=74)).values["warmup_epochs"] == 3.0
    # ≥500 images falls back to the instance-count rule
    big = _plan(_stats(num_images=800))
    assert big.values["warmup_epochs"] == 0.0


def test_early_stopping_enabled_with_size_scaled_patience():
    small = _plan(_stats(num_images=74))
    assert small.values["early_stopping"] is True
    assert small.values["early_stopping_patience"] == 15
    assert small.values["early_stopping_use_ema"] is True
    assert _plan(_stats(num_images=5000)).values["early_stopping_patience"] == 10


def test_augmentation_tier_follows_dataset_size():
    assert _plan(_stats(num_images=74)).values["aug_config"] == _AUG_STANDARD
    assert _plan(_stats(num_images=5000)).values["aug_config"] == _AUG_HEAVY
    # deterministic pixels across platforms (R7)
    assert _plan(_stats(num_images=74)).values["augmentation_backend"] == "cpu"


def test_mosaic_is_opt_in_for_every_dataset_size():
    # A/B-tested (balloon, seed 42): ratio 0.5 degraded val mAP and
    # confidence calibration, so the derived default is always 0
    for n in (74, 5000):
        plan = _plan(_stats(num_images=n))
        assert plan.values["mosaic_ratio"] == 0.0


def test_mosaic_ratio_is_api_field_not_backend_extra():
    plan = _plan(_stats(num_images=74), overrides={"mosaic_ratio": 0.5})
    assert plan.api_fields() == {"mosaic_ratio": 0.5}
    assert "mosaic_ratio" not in plan.extra_fields()
    # the backend-bound knobs stay in extra
    for key in ("lr_scheduler", "aug_config", "early_stopping"):
        assert key in plan.extra_fields()


def test_mosaic_ratio_override_wins_without_disturbing_the_rest():
    plan = _plan(_stats(num_images=74), overrides={"mosaic_ratio": 0.5})
    assert plan.values["mosaic_ratio"] == 0.5
    entry = next(d for d in plan.derivations if d.name == "mosaic_ratio")
    assert entry.overridden
    assert plan.values["lr_scheduler"] == "cosine"  # unrelated values intact


def test_user_extra_can_override_derived_schedule(tmp_path):
    # E5-S5: config.extra is applied after plan.extra_fields() in start_training
    plan = _plan(_stats(num_images=74))
    merged = {**plan.extra_fields(), **{"early_stopping": False}}
    assert merged["early_stopping"] is False
