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


@bp.get("/train/runs")
def list_runs():
    return jsonify([r.model_dump() for r in api.list_runs(_project())])


@bp.get("/train/runs/<run_id>")
def training_status(run_id: str):
    status = api.training_status(
        _project(), run_id, after=request.args.get("after", 0, type=int)
    )
    return jsonify(status.model_dump())


@bp.post("/train/runs/<run_id>/stop")
def stop_training(run_id: str):
    return jsonify({"stopped": api.stop_training(_project(), run_id)})
