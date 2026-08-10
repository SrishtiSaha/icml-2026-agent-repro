from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from agent_harness.cli import _validation_matches_publication, cmd_ingest_judge
from agent_harness.io import atomic_write_json


def test_judge_ingestion_rejects_stale_revision_and_writes_bounded_tickets(tmp_path: Path) -> None:
    workspace = tmp_path / "repro_feedback_workspace"
    control = workspace / ".repro"
    atomic_write_json(
        control / "publish_receipt.json",
        {
            "public_revision": "current-sha",
            "report_hash": "report-hash",
            "public_verified": True,
        },
    )
    response = tmp_path / "judge.json"
    response.write_text(
        json.dumps({
            "public_revision": "old-sha",
            "claims": [{"id": "claim-1", "verdict": "toy", "feedback": "Use the real model."}],
        }),
        encoding="utf-8",
    )
    args = argparse.Namespace(workspace=str(workspace), file=str(response), max_repairs=2)
    with pytest.raises(SystemExit, match="Stale or ambiguous"):
        cmd_ingest_judge(args)

    response.write_text(
        json.dumps({
            "public_revision": "current-sha",
            "claims": [{"id": "claim-1", "verdict": "toy", "feedback": "Use the real model."}],
        }),
        encoding="utf-8",
    )
    assert cmd_ingest_judge(args) == 0
    tickets = json.loads((control / "fix_tickets.json").read_text(encoding="utf-8"))
    ticket = tickets["tickets"][0]
    assert ticket["attempt"] == 1
    assert ticket["maximum_attempts"] == 2
    assert ticket["required_outcome"] == "Produce decisive verified or falsified evidence."


def test_public_validation_must_match_published_space() -> None:
    assert _validation_matches_publication(
        "https://huggingface.co/spaces/Owner/My-Space",
        "https://owner-my-space.static.hf.space/",
    )
    assert not _validation_matches_publication(
        "https://huggingface.co/spaces/Owner/My-Space",
        "https://owner-other-space.static.hf.space/",
    )
