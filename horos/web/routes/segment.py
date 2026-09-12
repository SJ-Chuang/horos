"""Interactive segmentation routes (SAM-T3). Thin by rule (R2)."""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

import horos.api as api
from horos.api.segment import DEFAULT_SEGMENTER, SegmentRequest
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("segment", __name__, url_prefix="/api/v1")


@bp.post("/images/<int:image_id>/segment")
def segment_image(image_id: int):
    body = request.get_json(silent=True) or {}
    try:
        spec = SegmentRequest.model_validate(body)
    except ValidationError as exc:
        raise ProjectError(f"Invalid segment request: {exc}") from exc
    return jsonify(api.segment_image(_project(), image_id, spec).model_dump())


@bp.post("/images/<int:image_id>/segment/prefetch")
def prefetch_embedding(image_id: int):
    body = request.get_json(silent=True) or {}
    model = body.get("model", DEFAULT_SEGMENTER)
    if not isinstance(model, str) or not model:
        raise ProjectError("'model' must be a model key")
    return jsonify(api.prefetch_embedding(_project(), image_id, model=model).model_dump())
