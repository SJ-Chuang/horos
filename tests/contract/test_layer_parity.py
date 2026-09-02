"""E9-T2: the three interface layers stay in sync with the capability manifest.

Every capability that declares a web route must have a live Flask route; every
/api/v1 route must trace back to a capability (or be on the explicit extras
list below); every declared CLI subcommand must exist in the parser.
"""

from __future__ import annotations

import argparse

import pytest

from horos.api.manifest import list_capabilities
from horos.cli import build_parser
from horos.web.app import create_app

# Routes that are deliberately manifest-free, with the reason on record:
EXTRA_ROUTES = {
    "/api/v1/project": "read-only summary of the bound project; UI convenience",
    "/api/v1/meta/manifest": "serves the manifest itself",
    "/api/v1/openapi.json": "serves the spec itself (E9-T4)",
    "/api/v1/train/runs/<run_id>/media/<media_id>/frames/<frame_name>":
        "static frame-file serving for the evaluate gallery (data, not an action)",
}


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def routes(app):
    out: dict[str, set[str]] = {}
    for rule in app.url_map.iter_rules():
        if rule.rule.startswith("/api/"):
            methods = {m for m in (rule.methods or set()) if m not in {"HEAD", "OPTIONS"}}
            out.setdefault(rule.rule, set()).update(methods)
    return out


def test_every_declared_web_capability_has_a_route(routes):
    missing = []
    for cap in list_capabilities():
        if not cap.web_route:
            continue
        live = routes.get(cap.web_route)
        if live is None:
            missing.append(f"{cap.name}: no route {cap.web_route}")
        elif not set(cap.web_methods) <= live:
            missing.append(
                f"{cap.name}: {cap.web_route} lacks methods "
                f"{set(cap.web_methods) - live}"
            )
    assert missing == [], "Web API is behind the manifest (E9-S1):\n" + "\n".join(missing)


def test_every_route_traces_back_to_a_capability(routes):
    declared = {c.web_route for c in list_capabilities() if c.web_route}
    orphans = [
        rule
        for rule in routes
        if rule not in declared and rule not in EXTRA_ROUTES
    ]
    assert orphans == [], (
        "Routes with no capability registration (register them or add to "
        f"EXTRA_ROUTES with a reason): {orphans}"
    )


def _cli_subcommands() -> set[str]:
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def test_every_declared_cli_capability_has_a_subcommand():
    subcommands = _cli_subcommands()
    missing = [
        f"{cap.name} -> horos {cap.cli}"
        for cap in list_capabilities()
        if cap.cli and cap.cli not in subcommands
    ]
    assert missing == [], "CLI is behind the manifest:\n" + "\n".join(missing)


def test_parity_gaps_are_all_intentional():
    # The acceptance rule for E9: gaps allowed, silent gaps not.
    for cap in list_capabilities():
        if not cap.web_route:
            assert cap.not_web_because, f"{cap.name} lacks a web-parity reason"
        if not cap.cli:
            assert cap.not_cli_because, f"{cap.name} lacks a cli-parity reason"
