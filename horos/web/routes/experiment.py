"""Experiment routes (E7-T7). Thin by rule (R2): validate, call horos.api."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import horos.api as api
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("experiment", __name__, url_prefix="/api/v1")


@bp.get("/experiments/runs/<run_id>")
def run_summary(run_id: str):
    return jsonify(api.get_run_summary(_project(), run_id).model_dump())


def _str_list(body: dict, key: str) -> list[str] | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProjectError(f"'{key}' must be a list of strings")
    return value


@bp.patch("/experiments/runs/<run_id>")
def update_run_notes(run_id: str):
    body = request.get_json(silent=True) or {}
    notes = body.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ProjectError("'notes' must be a string")
    summary = api.update_run_notes(
        _project(),
        run_id,
        notes=notes,
        tags=_str_list(body, "tags"),
        add_tags=_str_list(body, "add_tags"),
        remove_tags=_str_list(body, "remove_tags"),
    )
    return jsonify(summary.model_dump())
