"""Training routes (E5-T9, first slice). Thin by rule (R2): validate, call horos.api."""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

import horos.api as api
from horos.api.train import TrainRunConfig
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("train", __name__, url_prefix="/api/v1")


@bp.post("/train")
def start_training():
    body = request.get_json(silent=True) or {}
    try:
        config = TrainRunConfig.model_validate(body)
    except ValidationError as exc:
        raise ProjectError(f"Invalid training config: {exc}") from exc
    record = api.start_training(_project(), config)
    return jsonify(record.model_dump()), 202


@bp.post("/train/derive")
def derive_hyperparameters():
    body = request.get_json(silent=True) or {}
    try:
        config = TrainRunConfig.model_validate(body)
    except ValidationError as exc:
        raise ProjectError(f"Invalid training config: {exc}") from exc
    plan = api.derive_hyperparameters(_project(), config)
    return jsonify(plan.model_dump())


@bp.get("/train/runs")
def list_runs():
    # advance=True: a history refresh doubles as the queue's heartbeat
    return jsonify([r.model_dump() for r in api.list_runs(_project(), advance=True)])


@bp.get("/train/runs/<run_id>")
def training_status(run_id: str):
    status = api.training_status(
        _project(), run_id, after=request.args.get("after", 0, type=int)
    )
    return jsonify(status.model_dump())


@bp.post("/train/runs/<run_id>/stop")
def stop_training(run_id: str):
    return jsonify({"stopped": api.stop_training(_project(), run_id)})


@bp.delete("/train/runs/<run_id>")
def delete_run(run_id: str):
    return jsonify({"deleted": api.delete_run(_project(), run_id)})


@bp.patch("/train/runs/<run_id>")
def update_queued_run(run_id: str):
    body = request.get_json(silent=True) or {}
    record = api.update_queued_run(_project(), run_id, body)
    return jsonify(record.model_dump())


@bp.get("/train/runs/<run_id>/verdict")
def run_verdict(run_id: str):
    return jsonify(api.run_verdict(_project(), run_id).model_dump())
