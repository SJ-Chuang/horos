"""Autolabel + job routes (E3-T9). Thin by rule (R2): validate params, call horos.api."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

import horos.api as api
from horos.api.autolabel import DEFAULT_MODEL, DEFAULT_NMS_IOU, DEFAULT_THRESHOLD
from horos.errors import ProjectError

bp = Blueprint("autolabel", __name__, url_prefix="/api/v1")


def _project():
    root = current_app.config.get("HOROS_PROJECT_ROOT")
    if not root:
        raise ProjectError(
            "This server is not bound to a project. Start it with "
            "'horos ui --project <dir>' or POST /api/v1/projects first."
        )
    return api.open_project(root)


def _body() -> dict:
    return request.get_json(silent=True) or {}


def _prompt_spec(body: dict) -> api.PromptSpec:
    prompts = body.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise ProjectError(
            "Request body must include 'prompts': {class_name: [prompt, ...]}"
        )
    return api.PromptSpec(
        prompts={
            str(cls): [str(p) for p in (plist if isinstance(plist, list) else [plist])]
            for cls, plist in prompts.items()
        }
    )


@bp.post("/autolabel")
def start_autolabel():
    body = _body()
    job_id = api.start_autolabel(
        _project(),
        _prompt_spec(body),
        model=str(body.get("model", DEFAULT_MODEL)),
        threshold=float(body.get("threshold", DEFAULT_THRESHOLD)),
        nms_iou=float(body.get("nms_iou", DEFAULT_NMS_IOU)),
        split=body.get("split") or None,
        only_unannotated=bool(body.get("only_unannotated", True)),
        output=str(body.get("output", "bbox")),
    )
    return jsonify({"job_id": job_id}), 202


@bp.get("/jobs/<job_id>")
def job_status(job_id: str):
    status = api.job_status(
        _project(), job_id, after=request.args.get("after", 0, type=int)
    )
    return jsonify(status.model_dump())


@bp.post("/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    return jsonify({"cancelled": api.cancel_job(_project(), job_id)})


@bp.post("/images/<int:image_id>/assist")
def assist(image_id: int):
    body = _body()
    result = api.assist_image(
        _project(),
        image_id,
        _prompt_spec(body),
        model=str(body.get("model", DEFAULT_MODEL)),
        threshold=float(body.get("threshold", DEFAULT_THRESHOLD)),
        nms_iou=float(body.get("nms_iou", DEFAULT_NMS_IOU)),
        output=str(body.get("output", "bbox")),
    )
    return jsonify(result.model_dump())


@bp.post("/images/<int:image_id>/review")
def review(image_id: int):
    body = _body()
    action = body.get("action")
    if action not in ("accept", "reject"):
        raise ProjectError("Request body must include 'action': accept | reject")
    ann_ids = body.get("ann_ids")
    if ann_ids is not None and not isinstance(ann_ids, list):
        raise ProjectError("'ann_ids' must be a list of annotation ids")
    count = api.review_pending(
        _project(),
        image_id,
        action,
        ann_ids=[int(i) for i in ann_ids] if ann_ids is not None else None,
        min_score=(
            float(body["min_score"]) if body.get("min_score") is not None else None
        ),
    )
    return jsonify({"action": action, "count": count})


@bp.get("/autolabel/pending")
def pending():
    return jsonify([s.model_dump() for s in api.pending_summary(_project())])
