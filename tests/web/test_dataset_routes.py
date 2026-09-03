"""E1-T9: dataset Web API endpoints (thin routes over horos.api)."""

import io
import zipfile

import pytest
from helpers.data import write_sample_coco_dir

from horos.web.app import create_app


@pytest.fixture
def project_root(tmp_path):
    from horos.api import create_project, import_dataset

    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    return project.root


@pytest.fixture
def client(project_root):
    app = create_app(project_root)
    app.testing = True
    return app.test_client()


def _zip_of(directory) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for f in directory.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(directory))
    buffer.seek(0)
    return buffer


def test_project_summary(client):
    body = client.get("/api/v1/project").get_json()
    assert body["name"] == "proj"
    assert body["num_images"] == 3
    assert {c["name"] for c in body["categories"]} == {"forklift", "pallet"}


def test_stats_route(client):
    body = client.get("/api/v1/dataset/stats").get_json()
    assert body["num_images"] == 3
    assert body["num_annotations"] == 4


def test_validation_route(client):
    body = client.get("/api/v1/dataset/validation").get_json()
    assert body["ok"] is True
    assert body["issues"] == []


def test_validation_fix_route(client, project_root):
    from horos.api import open_project

    project = open_project(project_root)
    record = project.list_images()[0]  # 64x48
    stored = project.load_annotations(record.id)
    jittered = [
        stored.annotations[0].model_copy(update={"bbox": (0.5, 4.0, 64.0, 12.0)}),
        *stored.annotations[1:],
    ]
    project.save_annotations(record.id, jittered, expected_version=stored.version)

    report = client.get("/api/v1/dataset/validation").get_json()
    assert any(i["fixable"] for i in report["issues"])

    body = client.post("/api/v1/dataset/validation/fix").get_json()
    assert body["num_fixed"] == 1
    assert body["ok"] is True
    assert body["report"]["issues"] == []


def test_images_route(client):
    body = client.get("/api/v1/images").get_json()
    assert len(body) == 3
    assert {"id", "file_name", "width", "height", "split"} <= set(body[0])


def test_images_delete_route(client):
    ids = [i["id"] for i in client.get("/api/v1/images").get_json()]
    body = client.post(
        "/api/v1/images/delete", json={"ids": ids[:2], "session": "s1"}
    ).get_json()
    assert body["deleted"] == ids[:2] and body["skipped_claimed"] == []
    assert len(client.get("/api/v1/images").get_json()) == 1


def test_images_delete_route_requires_ids(client):
    response = client.post("/api/v1/images/delete", json={})
    assert response.status_code == 400
    assert "ids" in response.get_json()["error"]["message"]


def test_split_route(client):
    body = client.post(
        "/api/v1/dataset/split",
        json={"train": 1.0, "valid": 0.0, "test": 0.0, "seed": 5},
    ).get_json()
    assert body == {"train": 3, "valid": 0, "test": 0}


def test_upload_route(tmp_path):
    from horos.api import create_project

    project = create_project(tmp_path / "fresh")
    app = create_app(project.root)
    app.testing = True
    client = app.test_client()
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(coco_dir), "dataset.zip")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["num_images"] == 3
    assert body["instances_per_category"] == {"forklift": 2, "pallet": 2}


def test_import_by_path_route(tmp_path):
    from horos.api import create_project

    project = create_project(tmp_path / "fresh")
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    app = create_app(project.root)
    app.testing = True
    body = app.test_client().post(
        "/api/v1/dataset/import", json={"path": str(coco_dir)}
    ).get_json()
    assert body["format"] == "coco"
    assert body["num_images"] == 3


def test_export_route(client, tmp_path):
    body = client.post(
        "/api/v1/dataset/export",
        json={"out_dir": str(tmp_path / "out"), "format": "yolo"},
    ).get_json()
    assert body["path"].endswith("data.yaml")


def test_create_project_route(tmp_path):
    app = create_app()
    app.testing = True
    response = app.test_client().post(
        "/api/v1/projects", json={"path": str(tmp_path / "newproj"), "name": "n"}
    )
    assert response.status_code == 201
    assert response.get_json()["name"] == "n"


def test_models_route_carries_license(client):
    body = client.get("/api/v1/models").get_json()
    assert len(body) >= 4
    assert all(m["weights_license"] == "Apache-2.0" for m in body)


def test_capabilities_route(client):
    body = client.get("/api/v1/capabilities").get_json()
    features = {f["feature"] for f in body["features"]}
    assert "export_tensorrt" in features


def test_upload_conflict_flow(tmp_path, client, project_root):
    # phase 1: same names, different content -> 409 with the conflict list
    from helpers.data import make_image
    from helpers.data import write_sample_coco_dir as sample

    variant = sample(tmp_path / "variant")
    make_image(variant / "train" / "a.png", 64, 48, color=(1, 2, 3))
    post = lambda **extra: client.post(  # noqa: E731
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(variant), "dataset.zip"), **extra},
        content_type="multipart/form-data",
    )
    response = post()
    assert response.status_code == 409
    error = response.get_json()["error"]
    assert error["code"] == "import_conflict"
    assert error["details"]["conflicts"] == ["a.png"]
    # phase 2: retry with the chosen policy
    response = post(on_conflict="overwrite")
    assert response.status_code == 200
    body = response.get_json()
    assert body["overwritten"] == 1
    assert body["duplicates_skipped"] == 2


def test_upload_darknet_class_names_flow(tmp_path):
    from helpers.data import make_image

    from horos.api import create_project
    from horos.web.app import create_app as make_app

    src = tmp_path / "darknet"
    make_image(src / "img1.png")
    (src / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    app = make_app(create_project(tmp_path / "fresh").root)
    app.testing = True
    web = app.test_client()
    post = lambda **extra: web.post(  # noqa: E731
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(src), "dataset.zip"), **extra},
        content_type="multipart/form-data",
    )
    # phase 1: no _darknet.labels -> 422 with editable defaults
    response = post()
    assert response.status_code == 422
    error = response.get_json()["error"]
    assert error["code"] == "class_names_required"
    assert error["details"] == {"num_classes": 1, "default_names": ["0"]}
    # phase 2: retry with the names the user typed
    response = post(class_names='["helmet"]')
    assert response.status_code == 200
    assert response.get_json()["instances_per_category"] == {"helmet": 1}


def test_upload_bad_class_names_is_400(tmp_path, client):
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(tmp_path), "dataset.zip"), "class_names": "not json"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "class_names" in response.get_json()["error"]["message"]
