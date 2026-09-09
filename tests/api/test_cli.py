"""E9-T5: the CLI runs the whole E1 workflow without a browser (E9-S3)."""

import json
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.cli import main


def _run(capsys, *argv) -> tuple[int, dict | list]:
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip() else None)


def test_full_workflow(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    proj = tmp_path / "proj"

    code, body = _run(capsys, "init", str(proj), "--name", "demo")
    assert code == 0 and body["name"] == "demo"

    code, body = _run(capsys, "import", str(coco_dir), "--project", str(proj))
    assert code == 0 and body["num_images"] == 3

    code, body = _run(capsys, "stats", "--project", str(proj))
    assert code == 0 and body["num_annotations"] == 4

    code, body = _run(capsys, "split", "--project", str(proj),
                      "--train", "1.0", "--valid", "0.0", "--test", "0.0")
    assert code == 0 and body["train"] == 3

    code, body = _run(capsys, "export", str(tmp_path / "out"),
                      "--project", str(proj), "--format", "yolo")
    assert code == 0 and body["path"].endswith("data.yaml")


def test_validate_exit_code_reflects_dataset_health(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    proj = tmp_path / "proj"
    _run(capsys, "init", str(proj))
    _run(capsys, "import", str(coco_dir), "--project", str(proj))

    code, body = _run(capsys, "validate", "--project", str(proj))
    assert code == 0 and body["ok"] is True

    # break it: delete an image file
    from horos.api import open_project

    project = open_project(proj)
    (project.images_dir / project.list_images()[0].file_name).unlink()
    code, body = _run(capsys, "validate", "--project", str(proj))
    assert code == 1 and body["ok"] is False


def test_convert(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    code, body = _run(capsys, "convert", str(coco_dir), str(tmp_path / "yolo"),
                      "--to", "yolo")
    assert code == 0 and body["path"].endswith("data.yaml")


def test_convert_to_labelme(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    out = tmp_path / "labelme"
    code, body = _run(capsys, "convert", str(coco_dir), str(out), "--to", "labelme")
    assert code == 0 and body["path"] == str(out)
    assert (out / "train" / "a.json").is_file() and (out / "valid" / "c.json").is_file()


def test_catalog_lists_architectures_with_licenses(capsys):
    code, body = _run(capsys, "catalog")
    assert code == 0
    assert all(m["weights_license"] == "Apache-2.0" for m in body)


def test_models_lists_the_projects_trained_models(tmp_path, monkeypatch, capsys):
    from helpers.runs import completed_fake_run

    project, record = completed_fake_run(tmp_path, epochs=2)
    monkeypatch.chdir(project.root)
    code, body = _run(capsys, "models")
    assert code == 0 and [m["run_id"] for m in body] == [record.run_id]
    entry = body[0]
    assert entry["state"] == "completed" and entry["default"] is True
    assert entry["classes"] == ["forklift", "pallet"] and entry["epochs_completed"] == 2
    assert entry["scores"]["loss"] == pytest.approx(0.5)
    assert entry["checkpoint"].endswith("best.fake")


def test_models_without_all_hides_unfinished_runs(tmp_path, monkeypatch, capsys):
    import time

    from helpers.runs import FAKE, ensure_worker_can_import_helpers

    from horos.api import create_project, import_dataset
    from horos.api.train import TrainRunConfig, start_training, training_status

    ensure_worker_can_import_helpers()
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    failed = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=1, extra={"fail": True})
    )
    deadline = time.monotonic() + 60
    while training_status(project, failed.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    monkeypatch.chdir(project.root)
    code, body = _run(capsys, "models")
    assert code == 0 and body == []
    code, body = _run(capsys, "models", "--all")
    assert code == 0 and [m["state"] for m in body] == ["failed"]
    assert body[0]["default"] is False


def test_capabilities(capsys):
    code, body = _run(capsys, "capabilities")
    assert code == 0
    assert {f["feature"] for f in body["features"]} >= {"training", "export_tensorrt"}


def test_horos_errors_exit_2_with_stderr(tmp_path, capsys):
    code = main(["stats", "--project", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert code == 2
    assert "error [project_error]" in captured.err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_train_streams_events_and_exits_by_state(tmp_path, capsys, monkeypatch):
    """`horos train` runs in the foreground: it starts a run, prints the event
    stream as JSONL, and its exit code mirrors the terminal state (E5/E9-S3)."""
    import os

    from helpers.data import write_sample_coco_dir as _make

    import horos.api as api
    from horos.api.train import TrainRunConfig
    from horos.cli import main as cli_main

    tests_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", tests_root + (os.pathsep + existing if existing else "")
    )

    proj_dir = tmp_path / "proj"
    project = api.create_project(proj_dir)
    api.import_dataset(project, _make(tmp_path / "coco"))

    # the CLI builds the config itself; reroute it onto the fake backend
    original = api.start_training

    def with_fake(project, config):
        patched = TrainRunConfig(
            **config.model_dump(exclude={"entrypoint_override"}),
            entrypoint_override="helpers.fake_backend:FakeBackend",
        )
        return original(project, patched)

    monkeypatch.setattr(api, "start_training", with_fake)
    # the CLI refuses ML commands when torch/rfdetr are absent; the fake backend
    # needs no torch, so bypass the gate instead of tying this test to what is
    # installed (the default setup_local.sh venv has no ML stack)
    import horos.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_ml_preflight", lambda command: None)

    code = cli_main(["train", "--project", str(proj_dir), "--epochs", "2"])
    out = capsys.readouterr().out
    # events stream as one-line JSON; the final run record is pretty-printed —
    # decode the concatenated stream object by object
    decoder, pos, payloads = json.JSONDecoder(), 0, []
    while pos < len(out):
        remainder = out[pos:].lstrip()
        if not remainder:
            break
        obj, consumed = decoder.raw_decode(remainder)
        payloads.append(obj)
        pos += (len(out[pos:]) - len(remainder)) + consumed
    assert code == 0
    types = [p.get("type") for p in payloads]
    assert "started" in types and "completed" in types
    # the last JSON payload is the final run record
    assert payloads[-1]["state"] == "completed"


# ------------------------------------------------------- project & run discovery


def test_init_in_an_empty_directory_uses_it_directly(tmp_path, monkeypatch, capsys):
    """`horos init <name>` in an empty directory makes THAT directory the
    project — no pointless nesting (the common `mkdir x && cd x` flow)."""
    empty = tmp_path / "beds"
    empty.mkdir()
    monkeypatch.chdir(empty)
    code, body = _run(capsys, "init", "beds")
    assert code == 0
    assert Path(body["root"]) == empty.resolve() and body["name"] == "beds"
    assert (empty / "horos.json").is_file()


def test_init_ignores_dotfiles_when_deciding_emptiness(tmp_path, monkeypatch, capsys):
    # a fresh `git init` must not push the project into a subdirectory
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    monkeypatch.chdir(root)
    code, body = _run(capsys, "init", "repo")
    assert code == 0 and Path(body["root"]) == root.resolve()


def test_init_with_files_present_creates_a_subdirectory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("keep me", encoding="utf-8")
    monkeypatch.chdir(root)
    code, body = _run(capsys, "init", "proj")
    assert code == 0 and Path(body["root"]) == (root / "proj").resolve()
    assert (root / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_init_without_a_name_uses_the_current_directory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "unnamed"
    root.mkdir()
    monkeypatch.chdir(root)
    code, body = _run(capsys, "init")
    assert code == 0 and body["name"] == "unnamed"


def test_init_with_a_path_still_creates_that_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, body = _run(capsys, "init", "nested/deep/proj")
    assert code == 0 and Path(body["root"]).resolve() == (tmp_path / "nested/deep/proj").resolve()


def test_project_commands_find_the_project_from_a_subdirectory(tmp_path, monkeypatch, capsys):
    from horos.api import create_project, import_dataset

    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    monkeypatch.chdir(project.images_dir)  # a subdirectory of the project
    code, body = _run(capsys, "stats")
    assert code == 0 and body["num_images"] == 3


def test_missing_project_names_the_two_ways_out(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["stats"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--project" in captured.err and "horos init" in captured.err


def test_run_defaults_to_the_newest_completed_run(tmp_path, monkeypatch, capsys):
    from helpers.runs import completed_fake_run

    project, record = completed_fake_run(tmp_path, epochs=1)
    monkeypatch.chdir(project.root)
    code = main(["report", "--format", "xlsx"])
    captured = capsys.readouterr()  # one read: it drains both streams
    assert code == 0 and record.run_id in json.loads(captured.out)["path"]
    # the choice is announced on stderr, so stdout stays machine-readable
    assert f"using run {record.run_id}" in captured.err


def test_run_default_without_any_run_explains_itself(tmp_path, monkeypatch, capsys):
    from horos.api import create_project

    project = create_project(tmp_path / "empty_proj")
    monkeypatch.chdir(project.root)
    code = main(["report"])
    assert code == 2 and "no training runs yet" in capsys.readouterr().err
