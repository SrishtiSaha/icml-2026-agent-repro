#!/usr/bin/env python3
"""Write verdict-first Trackio logbook pages from structured claim evidence.

The script accepts either the preferred `{"claims": [...]}` schema or the
older JSON shapes emitted by the current reproduction verifier scripts. It
keeps Trackio's local static logbook files in sync enough for `trackio logbook
read` / `trackio logbook sync` to pick up the new pages.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JUDGE_VERDICTS = ("VERIFIED", "FALSIFIED", "TOY", "INCONCLUSIVE")
PROOF_AUDIT_FIELDS = (
    "theorem_anchors",
    "assumptions_checked",
    "proof_obligations",
    "algebra_verifier_outputs",
    "misextraction_audit",
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "claim"


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    return [str(value)]


def infer_verdict(status: str) -> str:
    text = status.lower()
    if any(word in text for word in ("toy", "proxy")):
        return "TOY"
    if any(
        phrase in text
        for phrase in (
            "partial",
            "partially",
            "paper-supported only",
            "not independently",
            "not rerun",
            "not re-run",
            "closed-model",
            "closed model",
            "unavailable",
            "blocked",
        )
    ):
        return "INCONCLUSIVE"
    if any(word in text for word in ("falsified", "contradicted", "not reproduced", "failed")):
        return "FALSIFIED"
    if any(word in text for word in ("inconclusive",)):
        return "INCONCLUSIVE"
    if any(word in text for word in ("verified", "supported", "passes", "pass")):
        return "VERIFIED"
    return "INCONCLUSIVE"


def claim_title(raw_title: Any, fallback: str) -> str:
    title = str(raw_title or fallback)
    if title.isdigit():
        return f"Claim {title}"
    return title


def normalize_claims(raw: Any, default_command: str | None = None) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("claims"), list):
        claims = raw["claims"]
    elif isinstance(raw, list):
        claims = raw
    elif isinstance(raw, dict) and isinstance(raw.get("verdicts"), dict):
        claims = []
        for idx, (key, value) in enumerate(raw["verdicts"].items(), start=1):
            evidence = value.get("evidence") if isinstance(value, dict) else None
            status = value.get("status", "") if isinstance(value, dict) else str(value)
            claims.append(
                {
                    "id": key,
                    "claim": key.replace("_", " ").title(),
                    "status": status,
                    "metric": status,
                    "scope": "Verifier-emitted evidence; inspect limitations in the source JSON.",
                    "artifacts": ["outputs/claim_verification.json"],
                    "confirmation_tests": as_list(evidence),
                    "counterexample_search": [
                        "Not yet encoded in this legacy verifier output; add seeds, edge cases, alternate aggregations, or ablations for this claim."
                    ],
                }
            )
    else:
        raise SystemExit("Unsupported claim JSON shape. Expected {'claims': [...]}, a list, or {'verdicts': {...}}.")

    normalized: list[dict[str, Any]] = []
    for idx, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            claim = {"claim": str(claim)}
        claim_id = str(claim.get("id") or claim.get("claim_id") or claim.get("claim") or f"claim-{idx}")
        status = str(claim.get("verdict") or claim.get("status") or "INCONCLUSIVE")
        verdict = str(claim.get("verdict") or infer_verdict(status)).upper()
        if verdict not in JUDGE_VERDICTS:
            verdict = infer_verdict(verdict)
        result = claim.get("metric") or claim.get("result") or claim.get("summary") or status
        title = claim_title(claim.get("title") or claim.get("claim"), claim_id)
        claim_text = claim.get("claim") or claim.get("title") or title
        if str(claim_text).isdigit():
            claim_text = title
        normalized.append(
            {
                "id": claim_id,
                "slug": slugify(claim.get("slug") or f"verdict-{claim_id}"),
                "title": title,
                "claim": str(claim_text),
                "required_evidence": str(claim.get("required_evidence") or "Not recorded."),
                "released_artifacts": as_list(claim.get("released_artifacts") or claim.get("available_artifacts")),
                "method": str(claim.get("method") or claim.get("reproduction_method") or "Not recorded."),
                "pass_condition": str(claim.get("pass_condition") or "Not recorded."),
                "evidence_level": str(claim.get("evidence_level") or claim.get("scope_type") or "Not recorded."),
                "verdict": verdict,
                "metric": str(result),
                "command": str(claim.get("command") or claim.get("exact_command") or default_command or "Not recorded; rerun through trackio logbook run -- ... if possible."),
                "scope": str(claim.get("scope") or "Scope not recorded."),
                "artifacts": as_list(claim.get("artifacts") or claim.get("artifact_bundle") or claim.get("evidence")),
                "confirmation_tests": as_list(claim.get("confirmation_tests") or claim.get("confirmation_test")),
                "counterexample_search": as_list(claim.get("counterexample_search") or claim.get("counterexamples")),
                "limitations": as_list(claim.get("limitations") or claim.get("blockers")),
                "theorem_anchors": as_list(claim.get("theorem_anchors") or claim.get("source_anchors")),
                "assumptions_checked": as_list(claim.get("assumptions_checked") or claim.get("assumption_checks")),
                "proof_obligations": as_list(claim.get("proof_obligations") or claim.get("proof_checks")),
                "algebra_verifier_outputs": as_list(
                    claim.get("algebra_verifier_outputs")
                    or claim.get("proof_audit_outputs")
                    or claim.get("symbolic_check_outputs")
                ),
                "misextraction_audit": as_list(
                    claim.get("misextraction_audit")
                    or claim.get("claim_extraction_audit")
                    or claim.get("absent_claim_fragments")
                ),
            }
        )
    return normalized


def bullet_list(items: list[str]) -> str:
    if not items:
        return "- None recorded.\n"
    return "".join(f"- {item}\n" for item in items)


def has_proof_audit(claim: dict[str, Any]) -> bool:
    return any(claim.get(field) for field in PROOF_AUDIT_FIELDS)


def proof_audit_markdown(claim: dict[str, Any]) -> str:
    if not has_proof_audit(claim):
        return ""
    return f"""
Proof/theory audit:

Theorem/source anchors:
{bullet_list(claim['theorem_anchors'])}
Assumptions checked:
{bullet_list(claim['assumptions_checked'])}
Proof obligations:
{bullet_list(claim['proof_obligations'])}
Algebra/symbolic verifier outputs:
{bullet_list(claim['algebra_verifier_outputs'])}
Claim-extraction or misextraction audit:
{bullet_list(claim['misextraction_audit'])}
"""


def write_claim_page(logbook: Path, claim: dict[str, Any]) -> None:
    page_dir = logbook / "pages" / claim["slug"]
    page_dir.mkdir(parents=True, exist_ok=True)
    md = f"""# {claim['verdict']}: {claim['title']}

Verdict: **{claim['verdict']}**. Metric/result: {claim['metric']}

Claim: {claim['claim']}

Exact command:

```bash
{claim['command']}
```

Required evidence: {claim['required_evidence']}

Released artifacts:
{bullet_list(claim['released_artifacts'])}
Reproduction method: {claim['method']}

Pass condition: {claim['pass_condition']}

Evidence level: {claim['evidence_level']}

Exact scope: {claim['scope']}

Artifact bundle:
{bullet_list(claim['artifacts'])}
Confirmation tests:
{bullet_list(claim['confirmation_tests'])}
Counterexample search:
{bullet_list(claim['counterexample_search'])}
Limitations/blockers:
{bullet_list(claim['limitations'])}
{proof_audit_markdown(claim)}
"""
    (page_dir / "page.md").write_text(md, encoding="utf-8")


def write_summary_page(logbook: Path, claims: list[dict[str, Any]], summary_slug: str) -> None:
    page_dir = logbook / "pages" / summary_slug
    page_dir.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = [
        "| Claim | Verdict | Exact metric/result | Method | Pass condition | Scope | Artifacts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for claim in claims:
        artifacts = "<br>".join(claim["artifacts"]) if claim["artifacts"] else "None recorded"
        rows.append(
            "| {claim} | **{verdict}** | {metric} | {method} | {pass_condition} | {scope} | {artifacts} |".format(
                claim=claim["title"].replace("|", "\\|"),
                verdict=claim["verdict"],
                metric=claim["metric"].replace("|", "\\|"),
                method=(claim["method"] + ("; proof audit" if has_proof_audit(claim) else "")).replace("|", "\\|"),
                pass_condition=claim["pass_condition"].replace("|", "\\|"),
                scope=claim["scope"].replace("|", "\\|"),
                artifacts=artifacts.replace("|", "\\|"),
            )
        )
    counts = {verdict: sum(1 for claim in claims if claim["verdict"] == verdict) for verdict in JUDGE_VERDICTS}
    md = f"""# Verdict-First Evidence Summary

Updated: {updated}

Counts: VERIFIED={counts['VERIFIED']}, FALSIFIED={counts['FALSIFIED']}, TOY={counts['TOY']}, INCONCLUSIVE={counts['INCONCLUSIVE']}.

This page is intentionally verdict-first for the ICML 2026 reproduction judge. Each claim is mapped to required evidence, released artifacts, a checkable method, a pass condition, a counterexample search, exact scope, exact command, and artifacts where available.

{chr(10).join(rows)}
"""
    (page_dir / "page.md").write_text(md, encoding="utf-8")


def load_logbook_json(logbook: Path) -> dict[str, Any]:
    path = logbook / "logbook.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "title": "Reproduction Logbook",
        "root": {"slug": "index", "title": "Reproduction Logbook", "file": "pages/index.md", "children": []},
    }


def update_logbook_index(logbook: Path, claims: list[dict[str, Any]], summary_slug: str) -> None:
    data = load_logbook_json(logbook)
    root = data.setdefault("root", {})
    root.setdefault("slug", "index")
    root.setdefault("title", data.get("title", "Reproduction Logbook"))
    root.setdefault("file", "pages/index.md")
    existing = root.setdefault("children", [])
    managed_slugs = {summary_slug, *(claim["slug"] for claim in claims)}
    remaining = [
        child
        for child in existing
        if child.get("slug") not in managed_slugs and not str(child.get("slug", "")).startswith("verdict-")
    ]
    managed = [
        {
            "slug": summary_slug,
            "title": "00 - Verdict-first evidence summary",
            "file": f"pages/{summary_slug}/page.md",
            "children": [],
        }
    ] + [
        {
            "slug": claim["slug"],
            "title": f"{claim['verdict']}: {claim['title']}",
            "file": f"pages/{claim['slug']}/page.md",
            "children": [],
        }
        for claim in claims
    ]
    root["children"] = managed + remaining
    data["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    (logbook / "logbook.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    title = root.get("title", "Reproduction Logbook")
    lines = [f"# {title}", "", "## Pages", "", "| Page |", "| --- |"]
    for child in root["children"]:
        lines.append(f"| [{child['title']}](#/{child['slug']}) |")
    lines.append("")
    pages_dir = logbook / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".", help="Paper workspace containing .trackio/logbook")
    parser.add_argument("--claims-json", required=True, help="Structured claim evidence JSON")
    parser.add_argument("--summary-slug", default="00-verdict-first-evidence-summary")
    parser.add_argument("--default-command", help="Command to use for claims whose JSON did not record one")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    logbook = workspace / ".trackio" / "logbook"
    if not logbook.exists():
        raise SystemExit(f"No Trackio logbook found at {logbook}")

    raw = json.loads(Path(args.claims_json).read_text(encoding="utf-8"))
    claims = normalize_claims(raw, default_command=args.default_command)
    if args.dry_run:
        print(json.dumps({"claims": claims}, indent=2))
        return

    write_summary_page(logbook, claims, args.summary_slug)
    for claim in claims:
        write_claim_page(logbook, claim)
    update_logbook_index(logbook, claims, args.summary_slug)
    print(f"Wrote {len(claims)} verdict pages plus summary under {logbook}")


if __name__ == "__main__":
    main()
