"""Export routes (E8-T8). Thin by rule (R2): validate params, call horos.api."""

from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file

import horos.api as api
from horos.api.export import export_file_path
from horos.web.routes.autolabel import _project

bp = Blueprint("export", __name__, url_prefix="/api/v1")


@bp.post("/train/runs/<run_id>/export/report")
def export_report(run_id: str):
    body = request.get_json(silent=True) or {}
    path = api.export_training_report(_project(), run_id, format=body.get("format", "png"))
    return jsonify({
        "name": path.name,
        "path": str(path),
        "download_url": f"/api/v1/train/runs/{run_id}/exports/{path.name}",
    })


@bp.post("/train/runs/<run_id>/export/model")
def export_model(run_id: str):
    body = request.get_json(silent=True) or {}
    job_id = api.start_model_export(
        _project(), run_id, format=body.get("format", "onnx"), options=body.get("options") or {}
    )
    return jsonify({"job_id": job_id}), 202


@bp.get("/train/runs/<run_id>/exports")
def list_exports(run_id: str):
    items = api.list_exports(_project(), run_id)
    return jsonify([
        a.model_dump() | {"download_url": f"/api/v1/train/runs/{run_id}/exports/{a.name}"}
        for a in items
    ])


@bp.get("/train/runs/<run_id>/exports/<name>")
def download_export(run_id: str, name: str):
    # export_file_path refuses anything outside <run>/exports/
    return send_file(export_file_path(_project(), run_id, name), as_attachment=True)
