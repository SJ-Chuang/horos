"""E7-T5: notes and tags on a run, from the API and the CLI."""

from __future__ import annotations

import json

import pytest
from helpers.runs import completed_fake_run

from horos.api.experiment import (
    RunExtras,
    get_run_summary,
    normalize_tags,
    update_run_notes,
    write_extras,
)
from horos.cli import main
from horos.errors import ProjectError


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return completed_fake_run(tmp_path_factory.mktemp("tags"), epochs=1)


@pytest.fixture(autouse=True)
def clean_sidecar(run):
    project, record = run
    write_extras(project.root / "runs" / record.run_id, RunExtras())
    yield


def test_notes_and_tags_round_trip(run):
    project, record = run
    summary = update_run_notes(
        project, record.run_id, notes="baseline, no mosaic  ", tags=["baseline", "nano"]
    )
    assert summary.notes == "baseline, no mosaic"
    assert summary.tags == ["baseline", "nano"]
    again = get_run_summary(project, record.run_id)
    assert (again.notes, again.tags) == (summary.notes, summary.tags)
    # the record itself is untouched: notes never leak into run.json
    run_json = json.loads((project.root / "runs" / record.run_id / "run.json").read_text("utf-8"))
    assert "notes" not in run_json and "tags" not in run_json


def test_add_and_remove_edit_in_place_case_insensitively(run):
    project, record = run
    update_run_notes(project, record.run_id, tags=["Baseline"])
    summary = update_run_notes(project, record.run_id, add_tags=["baseline", "sweep-1"])
    assert summary.tags == ["Baseline", "sweep-1"]  # duplicate ignored, first spelling kept
    summary = update_run_notes(project, record.run_id, remove_tags=["BASELINE"])
    assert summary.tags == ["sweep-1"]
    # notes untouched by tag edits
    update_run_notes(project, record.run_id, notes="keep me")
    assert update_run_notes(project, record.run_id, add_tags=["x"]).notes == "keep me"


def test_tag_normalization_rules():
    assert normalize_tags([" a ", "", "A", "b"]) == ["a", "b"]
    with pytest.raises(ProjectError, match="comma"):
        normalize_tags(["a,b"])
    with pytest.raises(ProjectError, match="longer than"):
        normalize_tags(["x" * 41])
    with pytest.raises(ProjectError, match="at most"):
        normalize_tags([f"t{i}" for i in range(40)])


def test_no_op_update_does_not_rewrite_the_sidecar(run):
    project, record = run
    sidecar = project.root / "runs" / record.run_id / "experiment.json"
    update_run_notes(project, record.run_id, notes="n", tags=["t"])
    stamp = json.loads(sidecar.read_text("utf-8"))["updated_at"]
    assert stamp is not None
    before = sidecar.read_bytes()
    update_run_notes(project, record.run_id, notes="n", add_tags=["T"])
    assert sidecar.read_bytes() == before


def test_unknown_run_is_refused(run):
    project, _ = run
    with pytest.raises(ProjectError, match="No such training run"):
        update_run_notes(project, "missing", notes="x")


def test_cli_tag_subcommand(run, capsys):
    project, record = run
    code = main([
        "tag", record.run_id, "--project", str(project.root),
        "--notes", "from the cli", "--add", "cli", "--add", "demo",
    ])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body == {"run_id": record.run_id, "notes": "from the cli", "tags": ["cli", "demo"]}

    code = main(["tag", record.run_id, "--project", str(project.root), "--set", "only,this"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["tags"] == ["only", "this"]

    code = main(["tag", record.run_id, "--project", str(project.root), "--remove", "only"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["tags"] == ["this"]
