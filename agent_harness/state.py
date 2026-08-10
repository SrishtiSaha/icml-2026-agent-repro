from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, require_valid
from .gates import blocking_failures, claim_gates, evidence_gates, fresh_successful_units, plan_gates
from .io import hash_file, hash_json, hash_paths, load_json


@dataclass(frozen=True)
class WorkspaceState:
    workspace: str
    stage: str
    next_action: str
    reason: str
    stale_artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stale_artifacts"] = list(self.stale_artifacts)
        return value


def _load_valid(path: Path, kind: str) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing {path.name}"
    try:
        document = load_json(path)
        require_valid(kind, document)
        return document, None
    except (OSError, ValueError, ContractError) as exc:
        return None, f"invalid {path.name}: {exc}"


def _state(workspace: Path, stage: str, action: str, reason: str, *stale: str) -> WorkspaceState:
    return WorkspaceState(str(workspace), stage, action, reason, tuple(stale))


def derive_state(workspace: Path) -> WorkspaceState:
    control = workspace / ".repro"
    source, error = _load_valid(control / "source_manifest.json", "source")
    if error:
        return _state(workspace, "new", "initialize source manifest", error)

    claims, error = _load_valid(control / "claims.json", "claims")
    if error:
        return _state(workspace, "source_pinned", "complete claim contract", error)
    claim_failures = blocking_failures(claim_gates(source, claims, workspace))
    if claim_failures:
        return _state(workspace, "source_pinned", "repair claim contract", claim_failures[0].message, "claims.json")

    plan, error = _load_valid(control / "plan.json", "plan")
    if error:
        return _state(workspace, "claims_locked", "write and precheck execution plan", error)
    plan_failures = blocking_failures(plan_gates(claims, plan))
    if plan_failures:
        return _state(workspace, "claims_locked", "repair execution plan", plan_failures[0].message, "plan.json")

    successful_units = fresh_successful_units(workspace, plan)
    units = {str(unit.get("id")): unit for unit in plan.get("units", [])}
    smoke_ok = any(unit_id in successful_units and unit.get("kind") == "smoke" for unit_id, unit in units.items())
    if not smoke_ok:
        stale = ()
        if any((control / "runs").glob("*.json")):
            stale = ("runs",)
        return _state(workspace, "plan_gated", "run smoke unit", "No successful smoke manifest for the current plan.", *stale)

    evidence, error = _load_valid(control / "evidence.json", "evidence")
    if error:
        has_scaled = any(unit_id in successful_units and unit.get("kind") != "smoke" for unit_id, unit in units.items())
        stage = "running" if has_scaled else "smoke_green"
        action = "compile evidence" if has_scaled else "run planned evidence units"
        return _state(workspace, stage, action, error)

    evidence_results = evidence_gates(workspace, claims, plan, evidence)
    evidence_failures = blocking_failures(evidence_results)
    if evidence_failures:
        return _state(
            workspace,
            "evidence_compiled",
            "repair evidence gaps",
            evidence_failures[0].message,
            "evidence.json",
        )

    report, error = _load_valid(control / "report_manifest.json", "report")
    if error:
        return _state(workspace, "evidence_complete", "render or repair report manifest", error, "report_manifest.json")
    if report.get("evidence_hash") != hash_json(evidence) or report.get("precheck_passed") is not True:
        return _state(workspace, "evidence_complete", "rerun report precheck", "Report is stale or did not pass.", "report_manifest.json")
    report_paths: list[Path] = []
    for artifact in report.get("artifacts", []):
        path = workspace / str(artifact.get("path", ""))
        if not path.is_file() or artifact.get("sha256") != hash_file(path):
            return _state(workspace, "evidence_complete", "rerun report precheck", "A report artifact changed or is missing.", "report_manifest.json")
        report_paths.append(path)
    if report.get("content_hash") != hash_paths(report_paths, workspace):
        return _state(workspace, "evidence_complete", "rerun report precheck", "Report content hash is stale.", "report_manifest.json")

    receipt, error = _load_valid(control / "publish_receipt.json", "publish")
    if error:
        return _state(workspace, "report_green", "publish or repair public read-back", error, "publish_receipt.json")
    if receipt.get("report_hash") != hash_json(report) or receipt.get("public_verified") is not True:
        return _state(workspace, "report_green", "republish and verify public content", "Publication is stale or unverified.", "publish_receipt.json")

    judge, error = _load_valid(control / "judge_snapshot.json", "judge")
    if error:
        return _state(workspace, "published_verified", "wait for or repair judge snapshot", error, "judge_snapshot.json")
    if judge.get("publish_hash") != hash_json(receipt):
        return _state(workspace, "published_verified", "wait for current publication verdict", "Judge verdict predates current publication.", "judge_snapshot.json")
    verdicts = {str(item.get("verdict", "")).lower() for item in judge.get("claims", [])}
    expected = {str(item.get("id")) for item in claims.get("claims", [])}
    judged = {str(item.get("id")) for item in judge.get("claims", [])}
    if expected == judged and verdicts <= {"verified", "falsified"}:
        return _state(workspace, "done", "none", "Every claim received a current decisive judge verdict.")
    return _state(workspace, "repair_required", "execute fix tickets without regressing decisive claims", "One or more claims lack a decisive judge verdict.")
