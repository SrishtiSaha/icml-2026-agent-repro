from __future__ import annotations

from pathlib import Path

from agent_harness.io import atomic_write_json, hash_file, hash_json, hash_paths
from agent_harness.state import derive_state


def write(path: Path, value: dict) -> None:
    atomic_write_json(path, value)


def test_content_addressed_stage_progression_and_invalidation(tmp_path: Path) -> None:
    workspace = tmp_path / "repro_example_workspace"
    control = workspace / ".repro"
    output = workspace / "outputs" / "smoke.json"
    output.parent.mkdir(parents=True)

    assert derive_state(workspace).stage == "new"
    source = {
        "schema_version": 1,
        "paper_id": "paper-1",
        "title": "Paper",
        "sources": [{"kind": "paper", "locator": "https://example.test/paper", "revision": "v1"}],
    }
    write(control / "source_manifest.json", source)
    assert derive_state(workspace).stage == "source_pinned"

    claims = {
        "schema_version": 1,
        "paper_id": "paper-1",
        "source_manifest_hash": hash_json(source),
        "claims": [{
            "id": "claim-1",
            "text": "A beats B.",
            "source_anchor": "Table 1",
            "evidence_type": "ranking_comparison",
            "required_metric": "paired score difference",
            "verification_condition": "A > B",
            "falsification_condition": "A <= B",
            "route": "local_command",
            "paper_scale": {"target": "full"},
            "fidelity": {"algorithm": "real"},
            "data_integrity": {"split": "official"},
            "controls": {"baseline": "B"},
            "statistics": {"seeds": 3},
            "counterexample_search": ["every reported dataset"],
        }],
    }
    write(control / "claims.json", claims)
    assert derive_state(workspace).stage == "claims_locked"

    plan = {
        "schema_version": 1,
        "paper_id": "paper-1",
        "source_manifest_hash": hash_json(source),
        "claims_hash": hash_json(claims),
        "budget": {"currency": "USD", "maximum": 0, "expected": 0},
        "units": [{
            "id": "smoke",
            "kind": "smoke",
            "claim_ids": ["claim-1"],
            "route": "local_command",
            "command": "python verify.py --smoke",
            "cost_tier": 0,
            "expected_artifacts": ["outputs/smoke.json"],
        }],
    }
    write(control / "plan.json", plan)
    assert derive_state(workspace).stage == "plan_gated"

    output.write_text('{"ok": true}\n', encoding="utf-8")
    run = {
        "schema_version": 1,
        "unit_id": "smoke",
        "kind": "smoke",
        "status": "success",
        "plan_hash": hash_json(plan),
        "command": "python verify.py --smoke",
        "recorded_at": "2026-01-01T00:00:00Z",
        "artifacts": [{"path": "outputs/smoke.json", "sha256": hash_file(output)}],
    }
    write(control / "runs" / "smoke.json", run)
    assert derive_state(workspace).stage == "smoke_green"

    evidence = {
        "schema_version": 1,
        "paper_id": "paper-1",
        "claims_hash": hash_json(claims),
        "plan_hash": hash_json(plan),
        "claims": [{
            "id": "claim-1",
            "verdict": "falsified",
            "metric": "A-B=-0.1",
            "scope": "full test set, three seeds",
            "artifacts": [{"path": "outputs/smoke.json", "sha256": hash_file(output)}],
            "fidelity": {"passed": True},
            "data_integrity": {"passed": True},
            "controls": {"passed": True},
            "statistics": {"passed": True},
        }],
    }
    write(control / "evidence.json", evidence)
    assert derive_state(workspace).stage == "evidence_complete"

    report = {
        "schema_version": 1,
        "evidence_hash": hash_json(evidence),
        "content_hash": hash_paths([output], workspace),
        "artifacts": [{"path": "outputs/smoke.json", "sha256": hash_file(output)}],
        "precheck_passed": True,
        "recorded_at": "2026-01-01T00:00:00Z",
    }
    write(control / "report_manifest.json", report)
    assert derive_state(workspace).stage == "report_green"

    receipt = {
        "schema_version": 1,
        "report_hash": hash_json(report),
        "url": "https://example.test/space",
        "public_revision": "sha",
        "public_content_sha256": "content-sha",
        "validation_report_hash": "validation-sha",
        "public_verified": True,
        "published_at": "2026-01-01T00:00:00Z",
    }
    write(control / "publish_receipt.json", receipt)
    assert derive_state(workspace).stage == "published_verified"

    judge = {
        "schema_version": 1,
        "publish_hash": hash_json(receipt),
        "judged_at": "2026-01-02T00:00:00Z",
        "claims": [{"id": "claim-1", "verdict": "falsified"}],
    }
    write(control / "judge_snapshot.json", judge)
    assert derive_state(workspace).stage == "done"

    output.write_text('{"ok": false}\n', encoding="utf-8")
    assert derive_state(workspace).stage == "plan_gated"
