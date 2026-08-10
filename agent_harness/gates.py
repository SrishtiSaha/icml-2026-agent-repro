from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import DECISIVE_VERDICTS
from .io import hash_file, hash_json, load_json


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    severity: str
    message: str
    claim_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _result(gate: str, passed: bool, message: str, claim_id: str | None = None, severity: str = "block") -> GateResult:
    return GateResult(gate, passed, severity, message, claim_id)


def claim_gates(source: dict[str, Any], claims: dict[str, Any], workspace: Path | None = None) -> list[GateResult]:
    results = [
        _result(
            "fresh-source",
            claims.get("source_manifest_hash") == hash_json(source),
            "Claim contract is tied to the current source manifest.",
        )
    ]
    results.append(_result("paper-identity", source.get("paper_id") == claims.get("paper_id"), "Source and claim contracts identify the same paper."))
    if workspace is not None:
        for index, item in enumerate(source.get("sources", [])):
            locator = item.get("locator")
            expected = item.get("sha256")
            if not locator or not expected:
                continue
            source_path = Path(str(locator))
            if not source_path.is_absolute():
                source_path = workspace / source_path
            if source_path.exists():
                results.append(
                    _result(
                        "source-content-hash",
                        source_path.is_file() and hash_file(source_path) == expected,
                        f"Pinned local source {index + 1} matches its declared hash.",
                    )
                )
    for claim in claims.get("claims", []):
        claim_id = claim.get("id")
        results.extend(
            [
                _result("source-anchor", bool(claim.get("source_anchor")), "Exact source anchor is recorded.", claim_id),
                _result("symmetric-verdict", bool(claim.get("verification_condition") and claim.get("falsification_condition")), "Both verification and falsification conditions are explicit.", claim_id),
                _result("counterexample", bool(claim.get("counterexample_search")), "At least one counterexample or failure search is planned.", claim_id),
                _result("fidelity-contract", bool(claim.get("paper_scale") and claim.get("fidelity")), "Paper scale and implementation fidelity are declared.", claim_id),
                _result("integrity-contract", bool(claim.get("data_integrity")), "Data provenance, splits, joins, or theorem assumptions are declared.", claim_id),
                _result("controls-contract", bool(claim.get("controls")), "Baseline and negative-control requirements are declared.", claim_id),
                _result("statistics-contract", bool(claim.get("statistics")), "Statistical adequacy requirements are declared.", claim_id),
            ]
        )
        if claim.get("evidence_type") == "theorem_audit":
            anti = claim.get("anti_tautology", {})
            results.append(
                _result(
                    "anti-tautology",
                    bool(anti.get("independent_computation") and anti.get("non_vacuity_test")),
                    "The theorem audit independently computes the target and tests non-vacuity.",
                    claim_id,
                )
            )
    return results


def plan_gates(claims: dict[str, Any], plan: dict[str, Any]) -> list[GateResult]:
    results = [
        _result("fresh-claims", plan.get("claims_hash") == hash_json(claims), "Plan is tied to the current claim contract."),
        _result("fresh-plan-source", plan.get("source_manifest_hash") == claims.get("source_manifest_hash"), "Plan is tied to the claim contract's source manifest."),
        _result("plan-paper-identity", plan.get("paper_id") == claims.get("paper_id"), "Plan and claims identify the same paper."),
    ]
    budget = plan.get("budget", {})
    results.append(
        _result(
            "budget",
            isinstance(budget.get("expected"), (int, float))
            and isinstance(budget.get("maximum"), (int, float))
            and 0 <= budget["expected"] <= budget["maximum"],
            "Expected spend is explicit and does not exceed the maximum.",
        )
    )
    required_ids = {str(claim.get("id")) for claim in claims.get("claims", [])}
    mapped = {str(claim_id) for unit in plan.get("units", []) for claim_id in unit.get("claim_ids", [])}
    results.append(_result("claim-coverage", required_ids <= mapped, f"Execution units cover all {len(required_ids)} claims."))

    units = plan.get("units", [])
    smoke_positions = [index for index, unit in enumerate(units) if unit.get("kind") == "smoke"]
    expensive_positions = [index for index, unit in enumerate(units) if int(unit.get("cost_tier", 0)) > 0]
    ordered = bool(smoke_positions) and (not expensive_positions or min(smoke_positions) < min(expensive_positions))
    results.append(_result("smoke-before-spend", ordered, "A smoke unit runs before paid or scaled work."))

    tiers = [int(unit.get("cost_tier", 0)) for unit in units]
    results.append(_result("cheapest-decisive-first", tiers == sorted(tiers), "Execution units are ordered by non-decreasing cost tier."))
    for unit in units:
        unit_id = str(unit.get("id"))
        results.append(_result("restart-unit", bool(unit.get("expected_artifacts")), "Natural unit has explicit exit artifacts.", unit_id))
    return results


def fresh_successful_units(workspace: Path, plan: dict[str, Any]) -> set[str]:
    expected = {str(unit.get("id")): unit for unit in plan.get("units", [])}
    successful: set[str] = set()
    for path in (workspace / ".repro" / "runs").glob("*.json"):
        try:
            run = load_json(path)
        except (OSError, ValueError):
            continue
        unit_id = str(run.get("unit_id"))
        unit = expected.get(unit_id)
        if unit is None or run.get("status") != "success" or run.get("plan_hash") != hash_json(plan):
            continue
        records = {str(item.get("path")): item for item in run.get("artifacts", []) if isinstance(item, dict)}
        valid = True
        for artifact in unit.get("expected_artifacts", []):
            record = records.get(str(artifact))
            artifact_path = workspace / str(artifact)
            if record is None or not artifact_path.is_file() or record.get("sha256") != hash_file(artifact_path):
                valid = False
                break
        if valid:
            successful.add(unit_id)
    return successful


def evidence_gates(workspace: Path, claims: dict[str, Any], plan: dict[str, Any], evidence: dict[str, Any]) -> list[GateResult]:
    results = [
        _result("fresh-evidence-claims", evidence.get("claims_hash") == hash_json(claims), "Evidence is tied to current claims."),
        _result("fresh-evidence-plan", evidence.get("plan_hash") == hash_json(plan), "Evidence is tied to current plan."),
        _result("evidence-paper-identity", evidence.get("paper_id") == claims.get("paper_id"), "Evidence and claims identify the same paper."),
    ]
    planned_units = {str(unit.get("id")) for unit in plan.get("units", [])}
    successful_units = fresh_successful_units(workspace, plan)
    results.append(
        _result(
            "execution-coverage",
            planned_units <= successful_units,
            f"All planned units have current successful manifests ({len(successful_units)}/{len(planned_units)}).",
        )
    )
    expected_ids = {str(claim.get("id")) for claim in claims.get("claims", [])}
    records = {str(record.get("id")): record for record in evidence.get("claims", [])}
    results.append(_result("evidence-coverage", expected_ids == set(records), "Evidence has exactly one record per claim."))
    for claim_id in sorted(expected_ids):
        record = records.get(claim_id, {})
        verdict = str(record.get("verdict", "")).lower()
        results.append(_result("decisive-verdict", verdict in DECISIVE_VERDICTS, "Claim is decisively verified or falsified.", claim_id))
        artifacts = record.get("artifacts", [])
        valid_artifacts = bool(artifacts)
        for artifact in artifacts:
            raw_path = artifact.get("path") if isinstance(artifact, dict) else artifact
            path = workspace / str(raw_path) if raw_path else None
            if not path or not path.is_file():
                valid_artifacts = False
            elif isinstance(artifact, dict) and artifact.get("sha256") != hash_file(path):
                valid_artifacts = False
        results.append(_result("raw-artifacts", valid_artifacts, "Raw evidence artifacts exist locally.", claim_id))
        for gate, key in (
            ("fidelity-executed", "fidelity"),
            ("integrity-executed", "data_integrity"),
            ("controls-executed", "controls"),
            ("statistics-executed", "statistics"),
        ):
            check = record.get(key, {})
            results.append(_result(gate, check.get("passed") is True, f"{key.replace('_', ' ').title()} check passed.", claim_id))
    return results


def blocking_failures(results: list[GateResult]) -> list[GateResult]:
    return [result for result in results if result.severity == "block" and not result.passed]
