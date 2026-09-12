"""E7-T7: experiment Web API endpoints (thin routes over horos.api.experiment)."""

from __future__ import annotations

import pytest
from helpers.experiments import project_with_runs, write_eval

from horos.api.experiment import RunExtras, write_extras
from horos.web.app import create_app


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    project, (r1, r2) = project_with_runs(tmp_path_factory.mktemp("routes"), epochs=(1, 2))
    write_eval(project, r2.run_id, map_50=0.8)
    app = create_app(project.root)
    app.testing = True
    return app.test_client(), project, r1, r2


@pytest.fixture(autouse=True)
def clean_sidecars(world):
    _, project, r1, r2 = world
    for record in (r1, r2):
        write_extras(project.root / "runs" / record.run_id, RunExtras())
    yield


def test_list_runs_sorted_and_filtered(world):
    client, project, r1, r2 = world
    body = client.get("/api/v1/experiments/runs").get_json()
    assert body["sort_by"] == "created_at" and body["descending"] is True
    assert [r["run"]["run_id"] for r in body["runs"]] == [r2.run_id, r1.run_id]
    assert "loss" in body["sort_keys"] and "eval.test.map_50" in body["sort_keys"]
    assert body["reference"] == "project"
    assert all(r["comparability"]["comparable"] for r in body["runs"])

    body = client.get("/api/v1/experiments/runs?sort=loss&desc=0").get_json()
    assert [r["scores"]["loss"] for r in body["runs"]] == [0.5, 1.0]

    body = client.get("/api/v1/experiments/runs?sort=eval.test.map_50").get_json()
    assert body["runs"][0]["run"]["run_id"] == r2.run_id
    assert body["runs"][0]["evals"]["test"]["map_50"] == 0.8

    body = client.get("/api/v1/experiments/runs?state=failed,stopped").get_json()
    assert body["runs"] == []

    body = client.get("/api/v1/experiments/runs?reference=none").get_json()
    assert all(r["comparability"] is None for r in body["runs"])


def test_bad_query_parameters_are_400_with_the_unified_shape(world):
    client, *_ = world
    response = client.get("/api/v1/experiments/runs?sort=bogus")
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert error["code"] == "project_error" and "Unknown sort key" in error["message"]
    assert client.get("/api/v1/experiments/runs?desc=maybe").status_code == 400


def test_patch_notes_and_tags_then_filter_by_tag(world):
    client, project, r1, r2 = world
    response = client.patch(
        f"/api/v1/experiments/runs/{r1.run_id}",
        json={"notes": "baseline run", "tags": ["baseline", "nano"]},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["notes"] == "baseline run" and body["tags"] == ["baseline", "nano"]

    body = client.patch(
        f"/api/v1/experiments/runs/{r1.run_id}",
        json={"add_tags": ["Sweep"], "remove_tags": ["nano"]},
    ).get_json()
    assert body["tags"] == ["baseline", "Sweep"]

    listing = client.get("/api/v1/experiments/runs?tag=sweep").get_json()
    assert [r["run"]["run_id"] for r in listing["runs"]] == [r1.run_id]

    single = client.get(f"/api/v1/experiments/runs/{r1.run_id}").get_json()
    assert single["notes"] == "baseline run" and single["run"]["state"] == "completed"


def test_patch_validates_its_body(world):
    client, project, r1, _ = world
    assert client.patch(
        f"/api/v1/experiments/runs/{r1.run_id}", json={"tags": "not-a-list"}
    ).status_code == 400
    assert client.patch(
        f"/api/v1/experiments/runs/{r1.run_id}", json={"notes": 42}
    ).status_code == 400
    assert client.patch(
        f"/api/v1/experiments/runs/{r1.run_id}", json={"tags": ["a,b"]}
    ).status_code == 400
    assert client.patch("/api/v1/experiments/runs/ghost", json={"notes": "x"}).status_code == 400


def test_compare_endpoint(world):
    client, project, r1, r2 = world
    body = client.get(f"/api/v1/experiments/compare?runs={r1.run_id},{r2.run_id}").get_json()
    assert [r["run"]["run_id"] for r in body["runs"]] == [r1.run_id, r2.run_id]
    epochs = next(row for row in body["hparams"] if row["name"] == "epochs")
    assert epochs["values"] == [1, 2] and epochs["differs"] is True
    loss = next(row for row in body["metrics"] if row["name"] == "loss")
    assert loss["values"] == [1.0, 0.5]
    assert body["runs"][1]["comparability"]["reference"] == r1.run_id

    assert client.get("/api/v1/experiments/compare").status_code == 400
    assert client.get("/api/v1/experiments/compare?runs=ghost").status_code == 400


def test_single_run_reference_parameter(world):
    client, project, r1, r2 = world
    body = client.get(f"/api/v1/experiments/runs/{r2.run_id}?reference={r1.run_id}").get_json()
    assert body["comparability"]["reference"] == r1.run_id
    assert body["comparability"]["comparable"] is True
