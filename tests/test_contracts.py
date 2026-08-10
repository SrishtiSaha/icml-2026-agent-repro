from __future__ import annotations

from agent_harness.contracts import validate_claims
from agent_harness.gates import blocking_failures, claim_gates
from agent_harness.io import hash_json


def _source() -> dict:
    return {
        "schema_version": 1,
        "paper_id": "paper-1",
        "title": "Paper",
        "sources": [{"kind": "paper", "locator": "paper.pdf", "revision": "v1"}],
    }


def _claim(evidence_type: str = "empirical_magnitude") -> dict:
    claim = {
        "id": "claim-1",
        "text": "The method improves accuracy.",
        "source_anchor": "page 3, Table 1",
        "evidence_type": evidence_type,
        "required_metric": "paired accuracy difference",
        "verification_condition": "difference > 0",
        "falsification_condition": "difference <= 0",
        "route": "local_command",
        "paper_scale": {"target": "full test set"},
        "fidelity": {"algorithm": "released method"},
        "data_integrity": {"split": "official"},
        "controls": {"baseline": "released baseline"},
        "statistics": {"seeds": 3},
        "counterexample_search": ["all reported settings"],
    }
    if evidence_type == "theorem_audit":
        claim["anti_tautology"] = {
            "independent_computation": "independent derivation",
            "non_vacuity_test": "construct active-assumption case",
        }
    return claim


def test_generic_claim_contract_passes() -> None:
    source = _source()
    claims = {
        "schema_version": 1,
        "paper_id": "paper-1",
        "source_manifest_hash": hash_json(source),
        "claims": [_claim()],
    }
    assert validate_claims(claims) == []
    assert blocking_failures(claim_gates(source, claims)) == []


def test_theorem_claim_requires_anti_tautology_gate() -> None:
    source = _source()
    claim = _claim("theorem_audit")
    claim.pop("anti_tautology")
    claims = {
        "schema_version": 1,
        "paper_id": "paper-1",
        "source_manifest_hash": hash_json(source),
        "claims": [claim],
    }
    failures = blocking_failures(claim_gates(source, claims))
    assert any(result.gate == "anti-tautology" for result in failures)
