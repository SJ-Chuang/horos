"""Meta routes: models, platform capabilities, capability manifest, OpenAPI."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

import horos.api as api

bp = Blueprint("meta", __name__, url_prefix="/api/v1")


@bp.get("/models")
def models():
    # R3: license rides along everywhere models are listed
    return jsonify([m.model_dump() for m in api.list_models()])


@bp.get("/capabilities")
def capabilities():
    return jsonify(api.platform_capabilities().model_dump())


@bp.get("/meta/manifest")
def manifest():
    return jsonify([c.model_dump() for c in api.list_capabilities()])


@bp.get("/openapi.json")
def openapi():
    from horos.web.openapi import build_spec

    return jsonify(build_spec(current_app))
