"""The WebUI pages render and link to each other (R2: the UI is served by the
same app but only ever talks to /api/v1 from the browser)."""

from __future__ import annotations

import re

import pytest

from horos.web.app import create_app

PAGES = {
    "/": "index",
    "/annotate": "annotate",
    "/train": "train",
    "/evaluate": "evaluate",
    "/experiments": "experiments",
    "/lab": "lab",
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.mark.parametrize("path", sorted(PAGES))
def test_page_renders_and_links_to_every_other_page(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    html = response.get_data(as_text=True)
    for other in PAGES:
        assert f'href="{other}"' in html, f"{path} has no nav link to {other}"
    # exactly one nav entry is marked as the current page, and it is this one
    current = re.findall(r'<a href="([^"]+)"[^>]*aria-current="page"', html)
    assert current == [path]


def test_evaluate_page_holds_metrics_and_error_analysis_only(client):
    html = client.get("/evaluate").get_data(as_text=True)
    assert 'id="eval-btn"' in html and 'id="errors-panel"' in html
    assert 'id="dropzone"' not in html and 'id="gallery"' not in html
    assert 'href="/lab"' in html  # points the user to where uploads went


def test_lab_page_holds_the_upload_playground_only(client):
    html = client.get("/lab").get_data(as_text=True)
    assert 'id="dropzone"' in html and 'id="gallery"' in html
    assert 'id="eval-btn"' not in html and 'id="errors-panel"' not in html


def test_pages_only_call_the_web_api(client):
    # R2: every fetch() in the inline scripts goes through the /api/v1 prefix
    for path in PAGES:
        html = client.get(path).get_data(as_text=True)
        for call in re.findall(r'fetch\(\s*"([^"]+)"', html):
            assert call.startswith("/api/v1"), f"{path} fetches {call} directly"


def test_experiments_page_holds_the_comparison_table_and_editor(client):
    # E7-T6: the run table, the side-by-side panel and the notes/tags editor
    html = client.get("/experiments").get_data(as_text=True)
    for element in ("runs-table", "sort-select", "ref-select", "compare-panel",
                    "notes-input", "tags-input", "save-btn", "link-train"):
        assert f'id="{element}"' in html, element
    # E7-S2: the export flow is entered from the table via the Training page's deep link
    assert "/train#" in html
    # every endpoint the page talks to exists under /api/v1/experiments
    assert "/experiments/runs" in html and "/experiments/compare" in html


def test_class_manager_offers_merge_and_train_setup_folds_advanced_knobs(client):
    annotate = client.get("/annotate").get_data(as_text=True)
    assert 'id="merge-row"' in annotate and 'id="merge-target"' in annotate
    assert "/categories/merge" in annotate
    train = client.get("/train").get_data(as_text=True)
    # criterion, seed and the non-primary derived knobs live inside the fold
    details = train[train.index('<details id="advanced-details">'):train.index("</details>")]
    for element in ("criterion-select", "hparams-advanced", "seed-input"):
        assert f'id="{element}"' in details, element
    assert 'id="hparams-list"' in train  # the primary knobs stay in view


def test_every_page_uses_the_shared_controls(client):
    """User decision 2026-09-12: one stepper / checkbox / file-button look on
    every page, served from static/controls.css+js rather than restyled per
    template."""
    for path in PAGES:
        html = client.get(path).get_data(as_text=True)
        assert '/static/controls.css' in html and '/static/controls.js' in html, path
        assert "one checkbox look" not in html, f"{path} restyles checkboxes locally"
    assert client.get("/static/controls.css").status_code == 200
    assert client.get("/static/controls.js").status_code == 200
    # every native number box sits inside a stepper (dynamic ones are built
    # by stepperHTML / horosControls.stepper, which emit the same markup)
    for path in ("/", "/lab", "/train", "/annotate"):
        html = client.get(path).get_data(as_text=True)
        for match in re.finditer(r'<input type="number"[^>]*id="([^"]+)"', html):
            before = html[max(0, match.start() - 60):match.start()]
            assert 'class="stepper' in before, f"{path}: #{match.group(1)} is not a stepper"
    lab = client.get("/lab").get_data(as_text=True)
    assert 'class="file-btn"' in lab and 'id="serve-file"' in lab
