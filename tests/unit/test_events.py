"""E4-T3 (R4): typed progress events, serializable for JSONL / SSE."""

import pytest
from pydantic import ValidationError

from horos.backends.base import (
    ImagePrediction,
    MetricsUpdated,
    PredictionReady,
    ProgressUpdated,
    RunCompleted,
    RunFailed,
    RunStarted,
    WarningRaised,
    dump_event,
    parse_event,
)

ALL_EVENTS = [
    RunStarted(total=10, config={"epochs": 10}),
    ProgressUpdated(current=3, total=10, phase="epoch 3/10"),
    MetricsUpdated(step=3, metrics={"loss": 0.5, "mAP": 0.7}),
    WarningRaised(message="something looks off"),
    PredictionReady(index=0, prediction=ImagePrediction(image="a.jpg")),
    RunCompleted(result={"checkpoint": "best.pt"}),
    RunFailed(error_code="backend_error", message="boom"),
]


@pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: e.type)
def test_events_roundtrip_through_json(event):
    line = dump_event(event)
    assert "\n" not in line  # JSONL-safe
    parsed = parse_event(line)
    assert type(parsed) is type(event)
    assert parsed.model_dump(exclude={"ts"}) == event.model_dump(exclude={"ts"})


def test_parse_event_dispatches_on_type_discriminator():
    parsed = parse_event({"type": "progress", "current": 1, "total": 5})
    assert isinstance(parsed, ProgressUpdated)


def test_events_reject_missing_required_fields():
    with pytest.raises(ValidationError):
        parse_event({"type": "metrics"})  # step + metrics required


def test_events_have_timestamps():
    event = ProgressUpdated(current=1)
    assert event.ts > 0


def test_event_kinds_cover_r4_minimum():
    # R4: started, progress, metrics, warning, completed, failed — at minimum
    kinds = {e.type for e in ALL_EVENTS}
    assert {"started", "progress", "metrics", "warning", "completed", "failed"} <= kinds
