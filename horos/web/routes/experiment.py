"""Experiment routes (E7-T7). Thin by rule (R2): validate, call horos.api."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import horos.api as api
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("experiment", __name__, url_prefix="/api/v1")


def _csv(name: str) -> list[str] | None:
    raw = request.args.get(name)
    return None if raw is None else [part for part in raw.split(",") if part.strip()]


@bp.get("/experiments/runs")
def query_runs():
    desc = request.args.get("desc", "1")
    if desc not in ("0", "1", "true", "false"):
        raise ProjectError("'desc' must be 0/1 or true/false")
    result = api.query_runs(
        _project(),
        sort_by=request.args.get("sort", "created_at"),
        descending=desc in ("1", "true"),
        states=_csv("state"),
        tags=_csv("tag"),
    )
    return jsonify(result.model_dump())


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
