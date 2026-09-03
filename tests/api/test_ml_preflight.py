"""CLI pre-flight gate: ML commands fail fast, with the fix, when the ML
stack is missing (`pip install horos` ships without it by design)."""

import horos.api.install as install_mod
from horos import cli


def test_ml_command_is_refused_with_a_pointer_to_horos_install(monkeypatch, capsys):
    monkeypatch.setattr(install_mod, "probe_missing", lambda: ["torch", "rfdetr"])
    code = cli.main(["train", "--project", "nope"])
    assert code == 2
    err = capsys.readouterr().err
    assert "horos install" in err
    assert "torch" in err and "rfdetr" in err


def test_ui_warns_but_still_runs(monkeypatch, capsys):
    # dataset management and annotation work without the ML stack, so `ui`
    # must start; here it proceeds past the gate to its own usage error
    monkeypatch.setattr(install_mod, "probe_missing", lambda: ["torch"])
    code = cli.main(["ui"])
    err = capsys.readouterr().err
    assert code == 2  # missing <project> argument — the gate let it through
    assert "usage: horos ui" in err
    assert "horos install" in err  # ...after warning about the missing stack


def test_non_ml_commands_never_probe(monkeypatch, capsys):
    def _boom():
        raise AssertionError("the gate must not probe for non-ML commands")

    monkeypatch.setattr(install_mod, "probe_missing", _boom)
    assert cli.main(["models"]) == 0
