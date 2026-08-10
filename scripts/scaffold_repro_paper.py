#!/usr/bin/env python3
"""Scaffold a generic paper reproduction workspace.

This creates a paper-neutral verifier skeleton, methodology contract templates,
and a paper-local route configuration. Paper routes never enter core config.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACES_ROOT = ROOT / "paper_workspaces"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "paper"


def verifier_source(title: str) -> str:
    return f'''#!/usr/bin/env python3
"""Claim verifier for {title}.

Fill in CLAIMS with the paper's major claims, then implement one confirmation
test and one counterexample search per claim. Keep outputs deterministic where
possible and write all artifacts under outputs/.

Default workflow:
1. Inventory released artifacts before launching GPU/API work.
2. Prefer exact aggregate recomputation when public labels, scores, logs,
   metadata, or tables identify the claim.
3. Emit a claim ledger, raw per-claim outputs, a summary, and executable checks.
4. Use HF Jobs for independent validation or substantive reruns only when useful.
5. For theoretical claims, prefer a proof audit over a toy simulation: cite
   theorem source anchors, list assumptions, decompose proof obligations, and
   add deterministic algebra/symbolic checks where possible. Keep simulations as
   sanity checks unless they instantiate the theorem's stated scope.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


CLAIMS = [
    {{
        "id": "claim-1",
        "title": "Claim 1: replace with paper claim",
        "claim": "Replace this with a falsifiable statement from the paper.",
        "expected": "Describe the numeric target, invariant, bound, aggregation, or baseline comparison.",
        "source_anchor": "TODO: exact page/table/figure/theorem/source-line anchor.",
        "evidence_type": "TODO: select the evidence modality from schemas/claims.schema.json.",
        "required_metric": "TODO: exact metric, aggregation, denominator, and comparator.",
        "required_evidence": "Describe the evidence needed to decide this claim.",
        "released_artifacts": [
            "TODO: list official CSV/JSONL/log/table/checkpoint/config/metadata artifacts.",
        ],
        "method": "TODO: deterministic aggregate recomputation, released-code run, HF Job, static inspection, or proxy.",
        "pass_condition": "TODO: exact threshold, tolerance, invariant, denominator, or comparison.",
        "falsification_condition": "TODO: exact observation that would falsify the claim.",
        "paper_scale": {{"target": "TODO", "minimum_adequate": "TODO with justification"}},
        "fidelity": {{"algorithm": "TODO", "model": "TODO", "metric": "TODO"}},
        "data_integrity": {{"revisions": [], "split_policy": "TODO", "row_identity": "TODO", "leakage_prevention": "TODO"}},
        "controls": {{"baseline": "TODO", "negative_control": "TODO", "paired_randomness": False, "all_cells_required": True}},
        "statistics": {{"seeds": 1, "replicates": 1, "uncertainty": "TODO", "multiple_comparisons": "TODO"}},
        "theorem_anchors": [
            "TODO if theoretical: theorem/lemma/source line anchors, e.g. paper.tex:123.",
        ],
        "assumptions_checked": [
            "TODO if theoretical: assumptions/preconditions verified from source or artifacts.",
        ],
        "proof_obligations": [
            "TODO if theoretical: proof steps, reductions, bounds, recurrences, or lemmas to verify.",
        ],
        "algebra_verifier_outputs": [
            "TODO if theoretical: CSV/JSON outputs from symbolic/numeric proof-step checks.",
        ],
        "misextraction_audit": "TODO if relevant: source-text check for any claim fragment absent from the paper.",
        "anti_tautology": {{"independent_computation": "TODO if theoretical", "non_vacuity_test": "TODO if theoretical"}},
    }}
]


def write_claim_ledger(claims: list[dict], output_path: Path) -> None:
    fieldnames = [
        "id",
        "title",
        "claim",
        "required_evidence",
        "released_artifacts",
        "method",
        "pass_condition",
        "falsification_condition",
        "evidence_type",
        "required_metric",
        "verdict",
        "metric",
        "scope",
        "outputs",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for claim in claims:
            writer.writerow(
                {{
                    "id": claim.get("id", ""),
                    "title": claim.get("title", ""),
                    "claim": claim.get("claim", ""),
                    "required_evidence": claim.get("required_evidence", ""),
                    "released_artifacts": "; ".join(claim.get("released_artifacts", [])),
                    "method": claim.get("method", ""),
                    "pass_condition": claim.get("pass_condition", ""),
                    "falsification_condition": claim.get("falsification_condition", ""),
                    "evidence_type": claim.get("evidence_type", ""),
                    "required_metric": claim.get("required_metric", ""),
                    "verdict": claim.get("verdict", ""),
                    "metric": claim.get("metric", ""),
                    "scope": claim.get("scope", ""),
                    "outputs": "; ".join(claim.get("artifacts", [])),
                }}
            )


def verify_claims(args: argparse.Namespace) -> list[dict]:
    claims = []
    for claim in CLAIMS:
        claims.append(
            {{
                "id": claim["id"],
                "title": claim["title"],
                "claim": claim["claim"],
                "verdict": "INCONCLUSIVE",
                "metric": "Verifier scaffold has not been filled in yet.",
                "command": "python scripts/verify_claims.py",
                "scope": "Scaffold only; no claim evidence generated yet.",
                "required_evidence": claim["required_evidence"],
                "released_artifacts": claim["released_artifacts"],
                "method": claim["method"],
                "pass_condition": claim["pass_condition"],
                "falsification_condition": claim["falsification_condition"],
                "source_anchor": claim["source_anchor"],
                "evidence_type": claim["evidence_type"],
                "required_metric": claim["required_metric"],
                "paper_scale": claim["paper_scale"],
                "fidelity": claim["fidelity"],
                "data_integrity": claim["data_integrity"],
                "controls": claim["controls"],
                "statistics": claim["statistics"],
                "evidence_level": "none-yet",
                "artifacts": [
                    "outputs/claim_ledger.csv",
                    "outputs/claim_verification.json",
                    "outputs/summary.json",
                    "outputs/test_results.csv",
                ],
                "confirmation_tests": [
                    "TODO: implement the paper-faithful check for this claim."
                ],
                "counterexample_search": [
                    "TODO: sweep seeds/settings/edge cases/null baselines that could break this claim."
                ],
                "limitations": [
                    "Replace this with real blockers, or remove it once evidence is complete."
                ],
                "theorem_anchors": claim.get("theorem_anchors", []),
                "assumptions_checked": claim.get("assumptions_checked", []),
                "proof_obligations": claim.get("proof_obligations", []),
                "algebra_verifier_outputs": claim.get("algebra_verifier_outputs", []),
                "misextraction_audit": claim.get("misextraction_audit", ""),
                "anti_tautology": claim.get("anti_tautology", {{}}),
            }}
        )
    return claims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--setting", default="default")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    claims = verify_claims(args)
    write_claim_ledger(claims, OUTPUTS / "claim_ledger.csv")
    (OUTPUTS / "claim_verification.json").write_text(
        json.dumps({{"claims": claims}}, indent=2),
        encoding="utf-8",
    )
    (OUTPUTS / "summary.json").write_text(
        json.dumps(
            {{
                "claims_total": len(claims),
                "verdict_counts": {{
                    verdict: sum(1 for claim in claims if claim["verdict"] == verdict)
                    for verdict in ["VERIFIED", "FALSIFIED", "TOY", "INCONCLUSIVE"]
                }},
                "tests_passed": 0,
                "tests_total": 0,
                "all_tests_passed": False,
            }},
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUTS / "test_results.csv").write_text(
        "test_id,claim_id,passed,metric,details\\n"
        "scaffold_filled,claim-1,false,,Replace scaffold with executable checks.\\n",
        encoding="utf-8",
    )
    print(json.dumps({{"claims": [claim["verdict"] for claim in claims], "tests": "0/0"}}, indent=2))
    print("Wrote outputs/claim_ledger.csv")
    print("Wrote outputs/claim_verification.json")
    print("Wrote outputs/summary.json")
    print("Wrote outputs/test_results.csv")


if __name__ == "__main__":
    main()
'''


def route_snippet(slug: str) -> dict:
    return {
        "schema_version": 1,
        "model_ladders": {},
        "papers": {
            slug: {
                "workspace": ".",
                "claims": {
                    "all": {
                        "title": "Run all claim checks",
                        "route": "local_command",
                        "command": "{python} scripts/verify_claims.py",
                        "postprocess_command": "{python} {root}/scripts/write_verdict_logbook.py --workspace {workspace} --claims-json {workspace}/outputs/claim_verification.json",
                        "confirmation_tests": [
                            "Run source and artifact checks before spending API or GPU budget."
                        ],
                        "counterexample_search": [
                            "Sweep alternate aggregations, seeds, edge cases, null baselines, destructive controls, and adversarial regions."
                        ],
                    }
                }
            }
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--slug", help="Paper key; defaults to slugified title")
    parser.add_argument("--workspace", help="Workspace name under paper_workspaces; defaults to repro_<slug>_workspace")
    args = parser.parse_args()

    slug = slugify(args.slug or args.title)
    workspace = args.workspace or f"repro_{slug}_workspace"
    workspace_path = WORKSPACES_ROOT / workspace

    scripts_dir = workspace_path / "scripts"
    outputs_dir = workspace_path / "outputs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    verifier = scripts_dir / "verify_claims.py"
    if not verifier.exists():
        verifier.write_text(verifier_source(args.title), encoding="utf-8")
        verifier.chmod(0o755)

    snippet_path = workspace_path / "claim_routes.json"
    snippet_path.write_text(
        json.dumps(route_snippet(slug), indent=2),
        encoding="utf-8",
    )

    contract_dir = workspace_path / ".repro" / "templates"
    contract_dir.mkdir(parents=True, exist_ok=True)
    for name in ("source_manifest.json", "claims.json", "plan.json", "evidence.json"):
        source = ROOT / "templates" / name
        target = contract_dir / name
        if not target.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    readme = workspace_path / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {args.title}\\n\\n"
            "Fill in `scripts/verify_claims.py`, the `.repro/templates/` contracts, and this workspace's `claim_routes.json`.\\n",
            encoding="utf-8",
        )

    print(f"Created/updated {workspace_path}")
    print(f"Verifier: {verifier}")
    print(f"Route snippet: {snippet_path}")


if __name__ == "__main__":
    main()
