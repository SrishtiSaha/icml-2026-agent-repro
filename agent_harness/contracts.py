from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


EVIDENCE_TYPES = {
    "artifact_recomputation",
    "theorem_audit",
    "empirical_magnitude",
    "ranking_comparison",
    "mechanism_intervention",
    "quality_evaluation",
    "identity_or_provenance",
}
ROUTES = {"local_command", "hf_job_command", "matrix_hf_job", "manual_source_audit"}
VERDICTS = {"verified", "falsified", "toy", "inconclusive"}
DECISIVE_VERDICTS = {"verified", "falsified"}


@dataclass(frozen=True)
class ContractIssue:
    path: str
    message: str


class ContractError(ValueError):
    def __init__(self, issues: list[ContractIssue]):
        self.issues = issues
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in issues))


def _object(value: Any, path: str, issues: list[ContractIssue]) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(ContractIssue(path, "must be an object"))
        return {}
    return value


def _list(value: Any, path: str, issues: list[ContractIssue]) -> list[Any]:
    if not isinstance(value, list):
        issues.append(ContractIssue(path, "must be an array"))
        return []
    return value


def _required_text(obj: dict[str, Any], key: str, path: str, issues: list[ContractIssue]) -> None:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        issues.append(ContractIssue(f"{path}.{key}", "must be non-empty text"))


def validate_source(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    _required_text(document, "paper_id", "$", issues)
    _required_text(document, "title", "$", issues)
    sources = _list(document.get("sources"), "$.sources", issues)
    if not sources:
        issues.append(ContractIssue("$.sources", "must contain at least one pinned source"))
    for index, raw in enumerate(sources):
        path = f"$.sources[{index}]"
        source = _object(raw, path, issues)
        _required_text(source, "kind", path, issues)
        _required_text(source, "locator", path, issues)
        if not source.get("revision") and not source.get("sha256"):
            issues.append(ContractIssue(path, "must pin revision or sha256"))
    return issues


def validate_claims(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    _required_text(document, "paper_id", "$", issues)
    _required_text(document, "source_manifest_hash", "$", issues)
    claims = _list(document.get("claims"), "$.claims", issues)
    if not claims:
        issues.append(ContractIssue("$.claims", "must contain at least one claim"))
    seen: set[str] = set()
    for index, raw in enumerate(claims):
        path = f"$.claims[{index}]"
        claim = _object(raw, path, issues)
        for key in (
            "id", "text", "source_anchor", "evidence_type", "required_metric",
            "verification_condition", "falsification_condition", "route",
        ):
            _required_text(claim, key, path, issues)
        claim_id = claim.get("id")
        if isinstance(claim_id, str):
            if claim_id in seen:
                issues.append(ContractIssue(f"{path}.id", "must be unique"))
            seen.add(claim_id)
        if claim.get("evidence_type") not in EVIDENCE_TYPES:
            issues.append(ContractIssue(f"{path}.evidence_type", f"must be one of {sorted(EVIDENCE_TYPES)}"))
        if claim.get("route") not in ROUTES:
            issues.append(ContractIssue(f"{path}.route", f"must be one of {sorted(ROUTES)}"))
        for key in ("paper_scale", "fidelity", "data_integrity", "controls", "statistics"):
            if not isinstance(claim.get(key), dict):
                issues.append(ContractIssue(f"{path}.{key}", "must be an object describing the requirement"))
        searches = claim.get("counterexample_search")
        if not isinstance(searches, list) or not searches:
            issues.append(ContractIssue(f"{path}.counterexample_search", "must contain at least one falsification attempt"))
    return issues


def validate_plan(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for key in ("paper_id", "source_manifest_hash", "claims_hash"):
        _required_text(document, key, "$", issues)
    budget = _object(document.get("budget"), "$.budget", issues)
    maximum = budget.get("maximum")
    expected = budget.get("expected")
    if not isinstance(maximum, (int, float)) or maximum < 0:
        issues.append(ContractIssue("$.budget.maximum", "must be a non-negative number"))
    if not isinstance(expected, (int, float)) or expected < 0:
        issues.append(ContractIssue("$.budget.expected", "must be a non-negative number"))
    if isinstance(maximum, (int, float)) and isinstance(expected, (int, float)) and expected > maximum:
        issues.append(ContractIssue("$.budget.expected", "must not exceed maximum"))
    units = _list(document.get("units"), "$.units", issues)
    if not units:
        issues.append(ContractIssue("$.units", "must contain at least one execution unit"))
    seen: set[str] = set()
    has_smoke = False
    for index, raw in enumerate(units):
        path = f"$.units[{index}]"
        unit = _object(raw, path, issues)
        for key in ("id", "kind", "route", "command"):
            _required_text(unit, key, path, issues)
        unit_id = unit.get("id")
        if isinstance(unit_id, str):
            if unit_id in seen:
                issues.append(ContractIssue(f"{path}.id", "must be unique"))
            seen.add(unit_id)
        if unit.get("kind") == "smoke":
            has_smoke = True
        claim_ids = unit.get("claim_ids")
        if not isinstance(claim_ids, list) or not claim_ids:
            issues.append(ContractIssue(f"{path}.claim_ids", "must map at least one claim"))
        if not isinstance(unit.get("expected_artifacts"), list) or not unit.get("expected_artifacts"):
            issues.append(ContractIssue(f"{path}.expected_artifacts", "must list expected exit artifacts"))
        tier = unit.get("cost_tier")
        if not isinstance(tier, int) or tier < 0:
            issues.append(ContractIssue(f"{path}.cost_tier", "must be a non-negative integer"))
    if not has_smoke:
        issues.append(ContractIssue("$.units", "must include a smoke execution unit"))
    return issues


def validate_run(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for key in ("unit_id", "kind", "status", "plan_hash", "command", "recorded_at"):
        _required_text(document, key, "$", issues)
    if document.get("status") not in {"running", "success", "failed"}:
        issues.append(ContractIssue("$.status", "must be running, success, or failed"))
    if document.get("status") == "success" and not isinstance(document.get("artifacts"), list):
        issues.append(ContractIssue("$.artifacts", "successful runs must record artifact hashes"))
    return issues


def validate_evidence(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for key in ("paper_id", "claims_hash", "plan_hash"):
        _required_text(document, key, "$", issues)
    claims = _list(document.get("claims"), "$.claims", issues)
    if not claims:
        issues.append(ContractIssue("$.claims", "must contain at least one claim record"))
    for index, raw in enumerate(claims):
        path = f"$.claims[{index}]"
        claim = _object(raw, path, issues)
        for key in ("id", "verdict", "metric", "scope"):
            _required_text(claim, key, path, issues)
        if str(claim.get("verdict", "")).lower() not in VERDICTS:
            issues.append(ContractIssue(f"{path}.verdict", f"must be one of {sorted(VERDICTS)}"))
        artifacts = claim.get("artifacts")
        if not isinstance(artifacts, list):
            issues.append(ContractIssue(f"{path}.artifacts", "must be an array of path/hash records"))
        for key in ("fidelity", "data_integrity", "controls", "statistics"):
            if not isinstance(claim.get(key), dict):
                issues.append(ContractIssue(f"{path}.{key}", "must record the executed check"))
    return issues


def validate_report(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for key in ("evidence_hash", "content_hash", "recorded_at"):
        _required_text(document, key, "$", issues)
    if document.get("precheck_passed") is not True:
        issues.append(ContractIssue("$.precheck_passed", "must be true"))
    if not isinstance(document.get("artifacts"), list) or not document.get("artifacts"):
        issues.append(ContractIssue("$.artifacts", "must contain hashed report artifacts"))
    else:
        for index, artifact in enumerate(document["artifacts"]):
            path = f"$.artifacts[{index}]"
            item = _object(artifact, path, issues)
            _required_text(item, "path", path, issues)
            _required_text(item, "sha256", path, issues)
    return issues


def validate_publish(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for key in ("report_hash", "url", "public_revision", "public_content_sha256", "validation_report_hash", "published_at"):
        _required_text(document, key, "$", issues)
    if document.get("public_verified") is not True:
        issues.append(ContractIssue("$.public_verified", "must be true after public read-back"))
    return issues


def validate_judge(document: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for key in ("publish_hash", "judged_at"):
        _required_text(document, key, "$", issues)
    claims = _list(document.get("claims"), "$.claims", issues)
    if not claims:
        issues.append(ContractIssue("$.claims", "must contain judge verdicts"))
    for index, raw in enumerate(claims):
        path = f"$.claims[{index}]"
        claim = _object(raw, path, issues)
        _required_text(claim, "id", path, issues)
        _required_text(claim, "verdict", path, issues)
        if str(claim.get("verdict", "")).lower() not in VERDICTS:
            issues.append(ContractIssue(f"{path}.verdict", f"must be one of {sorted(VERDICTS)}"))
    return issues


VALIDATORS: dict[str, Callable[[dict[str, Any]], list[ContractIssue]]] = {
    "source": validate_source,
    "claims": validate_claims,
    "plan": validate_plan,
    "run": validate_run,
    "evidence": validate_evidence,
    "report": validate_report,
    "publish": validate_publish,
    "judge": validate_judge,
}


def require_valid(kind: str, document: dict[str, Any]) -> None:
    issues = VALIDATORS[kind](document)
    if issues:
        raise ContractError(issues)
