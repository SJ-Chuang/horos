"""Dataset routes (E1-T9). Thin by rule (R2): validate params, call horos.api."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

import horos.api as api
from horos.errors import ProjectError

bp = Blueprint("data", __name__, url_prefix="/api/v1")


def _project():
    root = current_app.config.get("HOROS_PROJECT_ROOT")
    if not root:
        raise ProjectError(
            "This server is not bound to a project. Start it with "
            "'horos ui <dir>' or POST /api/v1/projects first."
        )
    return api.open_project(root)


def _body() -> dict:
    return request.get_json(silent=True) or {}


@bp.post("/projects")
def create_project():
    body = _body()
    path = body.get("path")
    if not path:
        raise ProjectError("Request body must include 'path'")
    project = api.create_project(path, name=body.get("name"))
    if not current_app.config.get("HOROS_PROJECT_ROOT"):
        current_app.config["HOROS_PROJECT_ROOT"] = str(project.root)
    return jsonify({"root": str(project.root), "name": project.manifest.name}), 201


@bp.get("/project")
def project_summary():
    project = _project()
    return jsonify(
        {
            "root": str(project.root),
            "name": project.manifest.name,
            "categories": [c.model_dump() for c in project.categories],
            "num_images": len(project.list_images()),
        }
    )


def _class_names(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        names = json.loads(raw)
    except ValueError as exc:
        raise ProjectError(f"'class_names' is not valid JSON: {exc}") from exc
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ProjectError("'class_names' must be a JSON array of strings")
    return names


@bp.post("/dataset/import")
def import_dataset():
    body = _body()
    source = body.get("path")
    if not source:
        raise ProjectError("Request body must include 'path' (dataset directory)")
    summary = api.import_dataset(
        _project(),
        source,
        format=body.get("format"),
        copy_images=bool(body.get("copy_images", True)),
        on_conflict=body.get("on_conflict", "ask"),
        class_names=body.get("class_names"),
    )
    return jsonify(summary.model_dump())


@bp.post("/dataset/upload")
def upload_dataset():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ProjectError("Attach the dataset zip as multipart field 'file'")
    with tempfile.TemporaryDirectory(prefix="horos_upload_") as tmp:
        zip_path = Path(tmp) / "upload.zip"
        upload.save(zip_path)
        summary = api.import_zip(
            _project(),
            zip_path,
            on_conflict=request.form.get("on_conflict", "ask"),
            class_names=_class_names(request.form.get("class_names")),
            # the UI shows an editable class-name dialog instead of placeholders
            require_class_names=True,
        )
    return jsonify(summary.model_dump())


@bp.post("/dataset/export")
def export_dataset():
    body = _body()
    out_dir = body.get("out_dir")
    if not out_dir:
        raise ProjectError("Request body must include 'out_dir'")
    written = api.export_dataset(
        _project(), out_dir, format=body.get("format", "coco")
    )
    return jsonify({"path": str(written)})


@bp.get("/dataset/validation")
def validation():
    report = api.validate_project(_project())
    return jsonify(report.model_dump() | {"ok": report.ok, "counts": report.counts()})


@bp.post("/dataset/validation/fix")
def validation_fix():
    result = api.fix_validation_issues(_project())
    return jsonify(result.model_dump() | {"ok": result.report.ok})


@bp.get("/dataset/stats")
def stats():
    return jsonify(api.dataset_stats(_project()).model_dump())


@bp.post("/dataset/split")
def split():
    body = _body()
    counts = api.resplit(
        _project(),
        train=float(body.get("train", 0.8)),
        valid=float(body.get("valid", 0.1)),
        test=float(body.get("test", 0.1)),
        seed=int(body.get("seed", 42)),
    )
    return jsonify(counts)


@bp.get("/images")
def images():
    return jsonify([i.model_dump() for i in api.list_images(_project())])


@bp.post("/images/delete")
def images_delete():
    body = _body()
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ProjectError("Request body must include a non-empty 'ids' list")
    summary = api.delete_images(
        _project(), [int(i) for i in ids], session_id=body.get("session")
    )
    return jsonify(summary.model_dump())
