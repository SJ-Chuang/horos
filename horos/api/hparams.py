"""Rule-based hyperparameter derivation (E5-T1) and overrides (E5-T2).

Pure logic: dataset statistics (E1-T7) + model metadata + a memory probe go
in, a plan with per-value reasons comes out. Every derived value records WHY
it was chosen — the plan is stored in the run metadata so users can see
"why did the system pick this resolution" (E5-S2). Search-based HPO is out of
scope for v1 by design (§6 E5).

Overrides: a user-supplied value replaces the derived one and is marked
`overridden`; the other derivations are computed exactly as before — partial
overrides never shift the rest of the plan. `extra` passthrough to the backend
still applies last on top of everything (E5-S5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from horos.backends.memory import MemoryInfo
    from horos.core.registry import ModelInfo
    from horos.core.stats import DatasetStats

__all__ = ["DerivedValue", "HyperparameterPlan", "derive_plan"]

#: effective batch size (batch × grad accumulation) the schedule is tuned for
_TARGET_EFFECTIVE_BATCH = 16
#: keep this share of measured available memory; the rest absorbs spikes
_MEMORY_HEADROOM = 0.7
#: assumed usable memory when availability is unknown (CPU / probe failure)
_CONSERVATIVE_BUDGET_GB = 3.0
#: reference cost: ~GB per sample for a ~30M-param model at 384px
_BASE_SAMPLE_GB = 1.5
_BASE_RESOLUTION = 384
_BASE_PARAMS_M = 30.5

#: TrainSpec-level knobs; everything else derived goes through spec.extra
_SPEC_FIELDS = ("epochs", "batch_size", "resolution")


class DerivedValue(BaseModel):
    name: str
    value: Any
    reason: str
    overridden: bool = False


class HyperparameterPlan(BaseModel):
    model: str
    #: final effective values, overrides already applied
    values: dict[str, Any] = Field(default_factory=dict)
    #: one entry per value, in derivation order, each with its reason
    derivations: list[DerivedValue] = Field(default_factory=list)
    #: honest non-actionables (e.g. imbalance the backend has no knob for)
    notes: list[str] = Field(default_factory=list)

    def spec_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if k in _SPEC_FIELDS}

    def extra_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if k not in _SPEC_FIELDS}


def _derive_epochs(stats: DatasetStats) -> tuple[int, str]:
    n = stats.num_images
    for limit, epochs in ((100, 100), (500, 60), (2000, 40), (10000, 25)):
        if n < limit:
            return epochs, (
                f"{n} images: small datasets need more passes to converge "
                f"(<{limit} images → {epochs} epochs)"
            )
    return 15, f"{n} images: large dataset, 15 epochs suffice per pass volume"


def _derive_resolution(
    stats: DatasetStats, model_info: ModelInfo | None
) -> tuple[int, str] | None:
    if model_info is None:
        return None  # unknown model: leave the backend default alone
    base = model_info.input_resolution
    area = stats.relative_area
    if area is not None and area.median < 0.01:
        # +128 keeps rfdetr's divisible-by-64 constraint for every base size
        return base + 128, (
            f"median object covers {area.median:.2%} of its image (<1%): "
            f"small objects need more pixels — raised {base} → {base + 128}"
        )
    return base, f"model's native input resolution ({base}px), objects are not tiny"


def _derive_batch(
    memory: MemoryInfo, model_info: ModelInfo | None, resolution: int | None
) -> tuple[int, str]:
    params = model_info.params_millions if model_info else _BASE_PARAMS_M
    res = resolution or _BASE_RESOLUTION
    per_sample = (
        _BASE_SAMPLE_GB
        * (res / _BASE_RESOLUTION) ** 2
        * math.sqrt(params / _BASE_PARAMS_M)
    )
    if memory.available_gb is not None:
        budget = memory.available_gb * _MEMORY_HEADROOM
        budget_reason = (
            f"{memory.available_gb:.1f} GB available on {memory.kind} "
            f"({memory.source}), {_MEMORY_HEADROOM:.0%} budgeted"
        )
    else:
        budget = _CONSERVATIVE_BUDGET_GB
        budget_reason = (
            f"memory availability unknown on {memory.kind} ({memory.source}): "
            f"conservative {_CONSERVATIVE_BUDGET_GB:g} GB budget"
        )
    raw = budget / per_sample
    batch = 2 ** int(math.log2(raw)) if raw >= 1 else 1
    batch = max(1, min(batch, 16))
    return batch, (
        f"{budget_reason}; ~{per_sample:.1f} GB/sample at {res}px "
        f"→ batch {batch} (power of two, capped at 16)"
    )


def derive_plan(
    stats: DatasetStats,
    *,
    model: str,
    model_info: ModelInfo | None,
    memory: MemoryInfo,
    overrides: dict[str, Any] | None = None,
) -> HyperparameterPlan:
    """Apply the derivation rules; `overrides` values (non-None) win per-key
    without disturbing how the other keys are derived (E5-T2)."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    plan = HyperparameterPlan(model=model)

    def put(name: str, value: Any, reason: str) -> Any:
        if name in overrides:
            value = overrides[name]
            entry = DerivedValue(
                name=name, value=value, reason="user override", overridden=True
            )
        else:
            entry = DerivedValue(name=name, value=value, reason=reason)
        plan.values[name] = value
        plan.derivations.append(entry)
        return value

    epochs, why = _derive_epochs(stats)
    put("epochs", epochs, why)

    resolution: int | None = None
    derived_res = _derive_resolution(stats, model_info)
    if derived_res is not None:
        resolution = put("resolution", *derived_res)
    elif "resolution" in overrides:
        resolution = put("resolution", None, "")

    batch, why = _derive_batch(memory, model_info, resolution)
    batch = put("batch_size", batch, why)

    put(
        "grad_accum_steps",
        max(1, round(_TARGET_EFFECTIVE_BATCH / batch)),
        f"accumulate gradients to an effective batch of {_TARGET_EFFECTIVE_BATCH} "
        f"(batch {batch} × accumulation)",
    )

    populated = [c for c in stats.per_class if c.instances > 0]
    weakest = min(populated, key=lambda c: c.instances, default=None)
    if weakest is not None and weakest.instances < 100:
        put(
            "warmup_epochs",
            1.0,
            f"weakest class '{weakest.name}' has only {weakest.instances} "
            f"instances (<100): one warmup epoch stabilizes early training",
        )
    else:
        put("warmup_epochs", 0.0, "every class has ≥100 instances, no warmup needed")

    put(
        "num_workers",
        0 if stats.num_images < 200 else 2,
        (
            f"{stats.num_images} images (<200): single-process data loading — "
            f"worker spawn overhead outweighs the gain"
            if stats.num_images < 200
            else f"{stats.num_images} images: 2 loader workers"
        ),
    )

    if stats.imbalance_ratio is not None and stats.imbalance_ratio > 3.0:
        plan.notes.append(
            f"Class imbalance is {stats.imbalance_ratio:.1f}:1. The current "
            f"backend exposes no sampling-strategy knob; consider adding data "
            f"for the underrepresented classes."
        )
    return plan
