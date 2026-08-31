"""OpenAPI spec generation (E9-T4).

Built from the live Flask url_map, enriched with capability-manifest summaries
where a route corresponds to a registered capability (E9-T1). Machine-readable
per E9-S4; not hand-maintained, so it cannot drift from the actual routes.
"""

from __future__ import annotations

from typing import Any

from flask import Flask

import horos
import horos.api as api


def build_spec(app: Flask) -> dict[str, Any]:
    summaries: dict[tuple[str, str], str] = {}
    for cap in api.list_capabilities():
        if cap.web_route:
            for method in cap.web_methods:
                summaries[(cap.web_route, method)] = cap.summary

    paths: dict[str, dict[str, Any]] = {}
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith("/api/"):
            continue
        methods = sorted(
            m for m in (rule.methods or set()) if m not in {"HEAD", "OPTIONS"}
        )
        entry = paths.setdefault(rule.rule, {})
        for method in methods:
            entry[method.lower()] = {
                "operationId": f"{rule.endpoint}_{method.lower()}",
                "summary": summaries.get((rule.rule, method), rule.endpoint),
                "responses": {
                    "200": {"description": "Success"},
                    "default": {
                        "description": "Error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                },
            }

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "horos Web API",
            "version": horos.__version__,
        },
        "paths": paths,
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                            },
                            "required": ["code", "message"],
                        }
                    },
                    "required": ["error"],
                }
            }
        },
    }
