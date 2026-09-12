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
