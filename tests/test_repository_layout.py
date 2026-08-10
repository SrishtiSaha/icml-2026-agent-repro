from __future__ import annotations

import json
from pathlib import Path

from agent_harness.workspaces import ROOT, discover_workspaces


def test_paper_workspaces_are_not_at_repository_root() -> None:
    misplaced = [
        path.name for path in ROOT.iterdir()
        if path.is_dir() and path.name.startswith(("repro_", "repro-"))
    ]
    assert misplaced == []
    assert len(discover_workspaces()) >= 1


def test_core_routes_contain_no_named_papers() -> None:
    core = json.loads((ROOT / "configs" / "claim_routes.json").read_text(encoding="utf-8"))
    assert core["papers"] == {}
