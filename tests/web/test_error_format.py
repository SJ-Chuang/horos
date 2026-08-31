"""E9-T3: one error shape for every /api response: {"error": {code, message}}."""

import pytest

from horos.web.app import create_app


@pytest.fixture
def client():
    app = create_app()  # deliberately unbound
    app.testing = True
    return app.test_client()


def _assert_error_shape(body):
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"]
    assert body["error"]["message"]


def test_horos_errors_map_to_the_unified_shape(client):
    response = client.get("/api/v1/dataset/stats")  # no project bound
    assert response.status_code == 400
    body = response.get_json()
    _assert_error_shape(body)
    assert body["error"]["code"] == "project_error"
    assert "horos ui --project" in body["error"]["message"]


def test_missing_body_field_is_a_clear_400(client):
    response = client.post("/api/v1/dataset/import", json={})
    assert response.status_code == 400
    _assert_error_shape(response.get_json())


def test_unknown_endpoint_is_json_not_html(client):
    response = client.get("/api/v1/no-such-thing")
    assert response.status_code == 404
    assert response.content_type.startswith("application/json")
    _assert_error_shape(response.get_json())


def test_wrong_method_is_json(client):
    response = client.get("/api/v1/dataset/import")  # POST-only
    assert response.status_code == 405
    _assert_error_shape(response.get_json())


def test_format_error_uses_stable_code(client, tmp_path):
    from horos.api import create_project

    project = create_project(tmp_path / "proj")
    app = create_app(project.root)
    app.testing = True
    (tmp_path / "mystery").mkdir()
    response = app.test_client().post(
        "/api/v1/dataset/import", json={"path": str(tmp_path / "mystery")}
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "dataset_format_error"
