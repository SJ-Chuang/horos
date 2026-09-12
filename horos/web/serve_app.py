"""The standalone inference service (E8-T7): what `horos serve` runs.

Thin by rule (R2): routes validate and delegate to the InferenceServer from
horos.api.serve. Deliberately not the project Web API — no project, no UI,
nothing but the model — so the same command deploys to a Jetson. Responses
carry permissive CORS headers so a browser page (the Lab page's Serve test
form, a customer's dashboard) can post images straight to it.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from horos.errors import HorosError, ProjectError
from horos.web.app import _status_for, error_payload

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def create_serve_app(server) -> Flask:
    app = Flask("horos.serve")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024**2  # one image per request

    @app.after_request
    def cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/")
    def usage():
        return jsonify({
            "service": "horos serve",
            "endpoints": {
                "GET /health": "liveness, loaded model, device, request count",
                "GET /model_card": "the model card shipped with the artifact "
                                   "(licence, classes, I/O contract)",
                "POST /predict": "multipart field 'file' (or a raw image body); form/query "
                                 "'threshold' (0-1); 'annotated=1' returns a JPEG overlay",
            },
            "model": server.health(),
        })

    @app.get("/health")
    def health():
        return jsonify(server.health())

    @app.get("/model_card")
    def model_card():
        return jsonify(server.source.card)

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return ("", 204)
        upload = request.files.get("file")
        raw = request.args.get("threshold", request.form.get("threshold"))
        try:
            threshold = None if raw in (None, "") else float(raw)
        except ValueError as exc:
            raise ProjectError(f"threshold must be a number, got {raw!r}") from exc
        annotated = (request.args.get("annotated") or request.form.get("annotated") or "0") \
            .lower() in ("1", "true", "yes")
        if upload is not None and upload.filename:
            name, data = upload.filename, upload.read()
        elif request.data and (request.mimetype or "").startswith("image/"):
            name, data = "image" + _suffix_for(request.mimetype), request.data
        else:
            raise ProjectError(
                "Send the image as multipart field 'file' or as a raw image/* body."
            )
        suffix = Path(name).suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            suffix = ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        try:
            prediction = server.predict(temp_path, threshold=threshold)
            if annotated:
                from horos.api.visualize import render_prediction_overlay

                image = render_prediction_overlay(temp_path, prediction)
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=90)
                buffer.seek(0)
                return send_file(buffer, mimetype="image/jpeg")
            payload = prediction.model_dump()
            payload["image"] = name  # never leak the server temp path
            payload["threshold"] = server.default_threshold if threshold is None else threshold
            return jsonify(payload)
        finally:
            temp_path.unlink(missing_ok=True)

    @app.errorhandler(HorosError)
    def handle_horos_error(exc: HorosError):
        logger.info("request failed: %s: %s", exc.code, exc)
        return jsonify(error_payload(exc.code, str(exc), exc.details)), _status_for(exc)

    @app.errorhandler(404)
    def handle_not_found(exc):
        return jsonify(error_payload("not_found", "No such endpoint")), 404

    @app.errorhandler(405)
    def handle_bad_method(exc):
        return jsonify(error_payload("method_not_allowed", str(exc))), 405

    @app.errorhandler(413)
    def handle_too_large(exc):
        return jsonify(error_payload("payload_too_large", "Image larger than 64 MB")), 413

    return app


def _suffix_for(mimetype: str) -> str:
    return {
        "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
        "image/bmp": ".bmp", "image/gif": ".gif", "image/tiff": ".tif",
    }.get(mimetype, ".png")
