"""Serve control routes (E8-T7). Thin by rule (R2): validate, call horos.api."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import horos.api as api
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("serve", __name__, url_prefix="/api/v1")


@bp.post("/serve")
def start_server():
    body = request.get_json(silent=True) or {}
    run_id = body.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ProjectError("Request body must include 'run_id'")
    port = body.get("port", 8080)
    if not isinstance(port, int):
        raise ProjectError("'port' must be an integer")
    threshold = body.get("threshold", 0.5)
    if not isinstance(threshold, int | float):
        raise ProjectError("'threshold' must be a number")
    status = api.start_server(
        _project(),
        run_id=run_id,
        format=str(body.get("format", "onnx")),
        host=str(body.get("host", "127.0.0.1")),
        port=port,
        threshold=float(threshold),
        device=body.get("device") or None,
    )
    return jsonify(status.model_dump()), 201


@bp.get("/serve")
def server_status():
    return jsonify(api.server_status(_project()).model_dump())


@bp.delete("/serve")
def stop_server():
    return jsonify(api.stop_server(_project()).model_dump())
