"""Experiment routes (E7-T7). Thin by rule (R2): validate, call horos.api."""

from __future__ import annotations

from flask import Blueprint, jsonify

import horos.api as api
from horos.web.routes.autolabel import _project

bp = Blueprint("experiment", __name__, url_prefix="/api/v1")


@bp.get("/experiments/runs/<run_id>")
def run_summary(run_id: str):
    return jsonify(api.get_run_summary(_project(), run_id).model_dump())
