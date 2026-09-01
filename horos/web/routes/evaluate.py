"""Evaluation routes (E6-T9, first slice). Thin by rule (R2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, request

import horos.api as api
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("evaluate", __name__, url_prefix="/api/v1")


@bp.post("/train/runs/<run_id>/infer")
def infer_image(run_id: str):
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ProjectError("Attach an image as multipart field 'file'.")
    threshold = request.form.get("threshold", 0.05, type=float)
    suffix = Path(upload.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        upload.save(handle)
        temp_path = Path(handle.name)
    try:
        prediction = api.infer_image(
            _project(), run_id, temp_path, threshold=threshold
        )
        payload = prediction.model_dump()
        payload["image"] = upload.filename  # never leak the server temp path
        return jsonify(payload)
    finally:
        temp_path.unlink(missing_ok=True)


@bp.post("/train/runs/<run_id>/evaluate")
def start_evaluation(run_id: str):
    body = request.get_json(silent=True) or {}
    job_id = api.start_evaluation(
        _project(), run_id, split=body.get("split", "test")
    )
    return jsonify({"job_id": job_id}), 202


@bp.get("/train/runs/<run_id>/eval/<split>")
def get_eval_report(run_id: str, split: str):
    return jsonify(api.get_eval_report(_project(), run_id, split).model_dump())
