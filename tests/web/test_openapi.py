"""E9-T4: machine-readable OpenAPI spec generated from the live routes."""

import pytest

from horos.web.app import create_app


@pytest.fixture
def spec():
    app = create_app()
    app.testing = True
    response = app.test_client().get("/api/v1/openapi.json")
    assert response.status_code == 200
    return response.get_json()


def test_spec_skeleton(spec):
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "horos Web API"
    assert spec["info"]["version"]


def test_spec_covers_every_api_route(spec):
    app = create_app()
    live = {
        r.rule
        for r in app.url_map.iter_rules()
        if r.rule.startswith("/api/")
    }
    assert live == set(spec["paths"])


def test_capability_summaries_flow_into_the_spec(spec):
    stats = spec["paths"]["/api/v1/dataset/stats"]["get"]
    assert "statistics" in stats["summary"].lower()


def test_error_schema_is_declared(spec):
    error = spec["components"]["schemas"]["Error"]
    assert error["properties"]["error"]["properties"]["code"]["type"] == "string"
    upload = spec["paths"]["/api/v1/dataset/upload"]["post"]
    assert upload["responses"]["default"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Error"
    }
