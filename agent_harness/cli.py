from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .contracts import ContractError, VALIDATORS, require_valid
from .events import record_event
from .gates import blocking_failures, claim_gates, evidence_gates, plan_gates
from .io import atomic_write_json, hash_file, hash_json, hash_paths, load_json, utc_now
from .state import derive_state
from .workspaces import control_dir, discover_workspaces, resolve_workspace


def _write(path: Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)
    print(path)


def _artifact_records(workspace: Path, values: list[str]) -> list[dict[str, str]]:
    records = []
    for value in values:
        path = (workspace / value).resolve()
        if not path.is_file():
            raise SystemExit(f"Artifact does not exist: {value}")
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise SystemExit(f"Artifact must be inside the workspace: {value}") from exc
        records.append({"path": relative, "sha256": hash_file(path)})
    return records


def _validation_matches_publication(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    base = urlparse(base_url)
    if parsed.hostname and parsed.hostname.endswith(".static.hf.space"):
        return parsed.hostname.lower() == (base.hostname or "").lower()
    marker = "/spaces/"
    if parsed.hostname == "huggingface.co" and marker in parsed.path:
        owner_and_name = parsed.path.split(marker, 1)[1].strip("/").split("/")
        if len(owner_and_name) >= 2:
            expected = f"{owner_and_name[0]}-{owner_and_name[1]}.static.hf.space".lower()
            return expected == (base.hostname or "").lower()
    return False


def cmd_init(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args.workspace)
    control = control_dir(workspace)
    source_path = control / "source_manifest.json"
    claims_path = control / "claims.json"
    if source_path.exists() or claims_path.exists():
        raise SystemExit("Refusing to overwrite existing source/claim contracts.")
    source = {
        "schema_version": 1,
        "paper_id": args.paper_id,
        "title": args.title,
        "sources": [{"kind": args.source_kind, "locator": args.source, "revision": args.revision}],
        "created_at": utc_now(),
    }
    claims = {
        "schema_version": 1,
        "paper_id": args.paper_id,
        "source_manifest_hash": hash_json(source),
        "claims": [],
    }
    _write(source_path, source)
    _write(claims_path, claims)
    record_event(workspace, "workspace_initialized", paper_id=args.paper_id)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    document = load_json(Path(args.file))
    issues = VALIDATORS[args.kind](document)
    for issue in issues:
        print(f"FAIL {issue.path}: {issue.message}")
    if not issues:
        print(f"PASS {args.kind}: {args.file}")
    return 1 if issues else 0


def cmd_hash(args: argparse.Namespace) -> int:
    path = Path(args.file)
    print(hash_json(load_json(path)) if args.json else hash_file(path))
    return 0


def _gate_results(workspace: Path, stage: str):
    control = control_dir(workspace)
    source = load_json(control / "source_manifest.json")
    claims = load_json(control / "claims.json")
    require_valid("source", source)
    require_valid("claims", claims)
    results = claim_gates(source, claims, workspace)
    if stage in {"plan", "evidence"}:
        plan = load_json(control / "plan.json")
        require_valid("plan", plan)
        results.extend(plan_gates(claims, plan))
    if stage == "evidence":
        evidence = load_json(control / "evidence.json")
        require_valid("evidence", evidence)
        results.extend(evidence_gates(workspace, claims, plan, evidence))
    return results


def cmd_precheck(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args.workspace)
    try:
        results = _gate_results(workspace, args.stage)
    except (OSError, ValueError, ContractError) as exc:
        print(f"FAIL contract: {exc}")
        return 1
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        subject = f" [{result.claim_id}]" if result.claim_id else ""
        print(f"{marker} {result.gate}{subject}: {result.message}")
    failures = blocking_failures(results)
    record_event(workspace, "precheck", stage=args.stage, passed=not failures, failures=len(failures))
    return 1 if failures else 0


def cmd_status(args: argparse.Namespace) -> int:
    workspaces = discover_workspaces() if args.all else [resolve_workspace(args.workspace)]
    states = [derive_state(workspace).to_dict() for workspace in workspaces]
    if args.json:
        print(json.dumps(states if args.all else states[0], indent=2))
    else:
        for state in states:
            print(f"{Path(state['workspace']).name}: {state['stage']} -> {state['next_action']} ({state['reason']})")
    return 0


def cmd_record_run(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args.workspace)
    plan = load_json(control_dir(workspace) / "plan.json")
    require_valid("plan", plan)
    document = {
        "schema_version": 1,
        "unit_id": args.unit_id,
        "kind": args.kind,
        "status": args.status,
        "plan_hash": hash_json(plan),
        "command": args.command,
        "recorded_at": utc_now(),
        "artifacts": _artifact_records(workspace, args.artifact),
    }
    require_valid("run", document)
    path = control_dir(workspace) / "runs" / f"{args.unit_id}.json"
    _write(path, document)
    record_event(workspace, "run_recorded", unit_id=args.unit_id, status=args.status, kind=args.kind)
    return 0


def cmd_record_report(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args.workspace)
    control = control_dir(workspace)
    evidence = load_json(control / "evidence.json")
    require_valid("evidence", evidence)
    tickets_path = control / "fix_tickets.json"
    if tickets_path.exists():
        tickets = load_json(tickets_path).get("tickets", [])
        unchanged = [
            str(ticket.get("claim_id")) for ticket in tickets
            if ticket.get("status") == "open" and ticket.get("prior_evidence_hash") == hash_json(evidence)
        ]
        if unchanged:
            raise SystemExit(
                "Refusing unchanged repair report; evidence hash did not change for: " + ", ".join(unchanged)
            )
    results = _gate_results(workspace, "evidence")
    failures = blocking_failures(results)
    if failures:
        for failure in failures:
            print(f"FAIL {failure.gate}: {failure.message}")
        return 1
    paths = [(workspace / value).resolve() for value in args.artifact]
    report = {
        "schema_version": 1,
        "evidence_hash": hash_json(evidence),
        "content_hash": hash_paths(paths, workspace),
        "artifacts": _artifact_records(workspace, args.artifact),
        "precheck_passed": True,
        "recorded_at": utc_now(),
    }
    require_valid("report", report)
    _write(control / "report_manifest.json", report)
    record_event(workspace, "report_recorded", content_hash=report["content_hash"])
    return 0


def cmd_record_publish(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args.workspace)
    control = control_dir(workspace)
    report = load_json(control / "report_manifest.json")
    if report.get("precheck_passed") is not True:
        raise SystemExit("Report precheck has not passed.")
    validation_path = Path(args.validation_report)
    validation = load_json(validation_path)
    if validation.get("passed") is not True or not validation.get("public_content_sha256"):
        raise SystemExit("Publication validation report did not pass public read-back.")
    if not _validation_matches_publication(args.url, str(validation.get("base_url", ""))):
        raise SystemExit("Publication URL does not match the public validation report.")
    receipt = {
        "schema_version": 1,
        "report_hash": hash_json(report),
        "url": args.url,
        "public_revision": args.public_revision,
        "public_content_sha256": validation["public_content_sha256"],
        "validation_report_hash": hash_json(validation),
        "public_verified": True,
        "published_at": utc_now(),
    }
    require_valid("publish", receipt)
    _write(control / "publish_receipt.json", receipt)
    record_event(workspace, "publication_verified", url=args.url, public_revision=args.public_revision)
    return 0


def _normalize_judge_claim(raw: dict[str, Any]) -> dict[str, Any]:
    claim_id = raw.get("id") or raw.get("claim_id")
    verdict = str(raw.get("verdict", "inconclusive")).lower()
    return {
        "id": str(claim_id),
        "verdict": verdict,
        "feedback": str(raw.get("feedback") or raw.get("evidence") or raw.get("reason") or ""),
    }


def cmd_ingest_judge(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args.workspace)
    control = control_dir(workspace)
    receipt = load_json(control / "publish_receipt.json")
    raw = load_json(Path(args.file))
    if raw.get("public_revision") != receipt.get("public_revision"):
        raise SystemExit(
            "Stale or ambiguous judge response: public_revision does not match the current publication receipt."
        )
    claims = [_normalize_judge_claim(item) for item in raw.get("claims", [])]
    snapshot = {
        "schema_version": 1,
        "publish_hash": hash_json(receipt),
        "judged_at": raw.get("judged_at", utc_now()),
        "claims": claims,
    }
    require_valid("judge", snapshot)
    previous_path = control / "judge_snapshot.json"
    previous = load_json(previous_path) if previous_path.exists() else {"claims": []}
    previous_decisive = {
        str(item.get("id")) for item in previous.get("claims", [])
        if str(item.get("verdict", "")).lower() in {"verified", "falsified"}
    }
    current_decisive = {
        str(item.get("id")) for item in claims
        if str(item.get("verdict", "")).lower() in {"verified", "falsified"}
    }
    regressed = sorted(previous_decisive - current_decisive)
    if regressed:
        raise SystemExit(f"Per-claim non-regression failed for: {', '.join(regressed)}")
    old_tickets_path = control / "fix_tickets.json"
    old_tickets = load_json(old_tickets_path) if old_tickets_path.exists() else {"tickets": []}
    attempts = {str(item.get("claim_id")): int(item.get("attempt", 0)) for item in old_tickets.get("tickets", [])}
    evidence_path = control / "evidence.json"
    evidence_hash = hash_json(load_json(evidence_path)) if evidence_path.exists() else None
    tickets = {
        "schema_version": 1,
        "publish_hash": snapshot["publish_hash"],
        "tickets": [
            {
                "claim_id": item["id"],
                "current_verdict": item["verdict"],
                "evidence_gap": item["feedback"],
                "required_outcome": "Produce decisive verified or falsified evidence.",
                "required_experiment_delta": {
                    "evidence_modality": "fill from claim contract and feedback",
                    "algorithm_or_model": "fill only if fidelity is the gap",
                    "data_or_scale": "fill only if coverage is the gap",
                    "metric_or_control": "fill only if decision validity is the gap"
                },
                "prior_evidence_hash": evidence_hash,
                "attempt": attempts.get(item["id"], 0) + 1,
                "maximum_attempts": args.max_repairs,
                "status": "open" if attempts.get(item["id"], 0) < args.max_repairs else "blocked-budget",
            }
            for item in claims if item["verdict"] not in {"verified", "falsified"}
        ],
    }
    _write(previous_path, snapshot)
    _write(control / "fix_tickets.json", tickets)
    record_event(workspace, "judge_ingested", tickets=len(tickets["tickets"]), decisive=len(current_decisive))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repro-agent", description="Deterministic paper-reproduction harness")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("--workspace", required=True)
    init.add_argument("--paper-id", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--source", required=True)
    init.add_argument("--source-kind", default="paper")
    init.add_argument("--revision", required=True)
    init.set_defaults(func=cmd_init)

    validate = commands.add_parser("validate")
    validate.add_argument("kind", choices=sorted(VALIDATORS))
    validate.add_argument("file")
    validate.set_defaults(func=cmd_validate)

    hash_command = commands.add_parser("hash")
    hash_command.add_argument("file")
    hash_command.add_argument("--json", action="store_true", help="Hash canonical JSON rather than file bytes")
    hash_command.set_defaults(func=cmd_hash)

    precheck = commands.add_parser("precheck")
    precheck.add_argument("--workspace", required=True)
    precheck.add_argument("--stage", choices=("claims", "plan", "evidence"), default="evidence")
    precheck.set_defaults(func=cmd_precheck)

    status = commands.add_parser("status")
    selection = status.add_mutually_exclusive_group(required=True)
    selection.add_argument("--workspace")
    selection.add_argument("--all", action="store_true")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    run = commands.add_parser("record-run")
    run.add_argument("--workspace", required=True)
    run.add_argument("--unit-id", required=True)
    run.add_argument("--kind", choices=("smoke", "scaled", "audit"), required=True)
    run.add_argument("--status", choices=("running", "success", "failed"), required=True)
    run.add_argument("--command", required=True)
    run.add_argument("--artifact", action="append", default=[])
    run.set_defaults(func=cmd_record_run)

    report = commands.add_parser("record-report")
    report.add_argument("--workspace", required=True)
    report.add_argument("--artifact", action="append", required=True)
    report.set_defaults(func=cmd_record_report)

    publish = commands.add_parser("record-publish")
    publish.add_argument("--workspace", required=True)
    publish.add_argument("--url", required=True)
    publish.add_argument("--public-revision", required=True)
    publish.add_argument("--validation-report", required=True)
    publish.set_defaults(func=cmd_record_publish)

    judge = commands.add_parser("ingest-judge")
    judge.add_argument("--workspace", required=True)
    judge.add_argument("--file", required=True)
    judge.add_argument("--max-repairs", type=int, default=2)
    judge.set_defaults(func=cmd_ingest_judge)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (ContractError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
