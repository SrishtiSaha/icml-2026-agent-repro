#!/usr/bin/env python3
"""Route paper claims to the right reproduction command.

Default behavior is dry-run: print exact commands without submitting HF Jobs or
spending API budget. Pass --execute when you intentionally want to run them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACES_ROOT = ROOT / "paper_workspaces"
CORE_CONFIG = ROOT / "configs" / "claim_routes.json"
DEFAULT_SECRETS = ROOT / ".hf-job-secrets"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_harness.state import derive_state


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    for paper_cfg in config["papers"].values():
        paper_cfg["_config_dir"] = str(path.parent.resolve())
    return config


def discover_configs() -> list[Path]:
    paths = [CORE_CONFIG] if CORE_CONFIG.exists() else []
    if WORKSPACES_ROOT.exists():
        paths.extend(sorted(WORKSPACES_ROOT.glob("*/claim_routes.json")))
    return paths


def load_configs(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {"schema_version": 1, "route_types": {}, "model_ladders": {}, "papers": {}}
    for path in paths:
        config = load_config(path.resolve())
        merged["route_types"].update(config.get("route_types", {}))
        merged["model_ladders"].update(config.get("model_ladders", {}))
        for paper, paper_cfg in config.get("papers", {}).items():
            if paper in merged["papers"]:
                raise SystemExit(f"Duplicate paper route {paper!r} in {path}")
            merged["papers"][paper] = paper_cfg
    return merged


def require_fields(obj: dict[str, Any], fields: list[str], label: str) -> None:
    missing = [field for field in fields if field not in obj]
    if missing:
        raise SystemExit(f"{label} is missing required field(s): {', '.join(missing)}")


def validate_config(config: dict[str, Any]) -> None:
    require_fields(config, ["papers"], "claim route config")
    for paper, paper_cfg in config["papers"].items():
        require_fields(paper_cfg, ["workspace", "claims"], f"paper {paper!r}")
        for claim_id, claim in paper_cfg["claims"].items():
            label = f"paper {paper!r} claim {claim_id!r}"
            require_fields(claim, ["title", "route"], label)
            route = claim["route"]
            if route == "local_command":
                require_fields(claim, ["command"], label)
            elif route == "hf_job_command":
                require_fields(claim, ["image", "remote_command"], label)
            elif route == "matrix_hf_job":
                require_fields(claim, ["image", "remote_command", "matrix"], label)
            else:
                raise SystemExit(f"{label} has unsupported route {route!r}")


def list_routes(config: dict[str, Any]) -> None:
    if config.get("route_types"):
        print("Route types:")
        for route_name, route_info in config["route_types"].items():
            print(f"  {route_name}: {route_info.get('description', '')}")
        print()
    if config.get("model_ladders"):
        print("Model ladders:")
        for ladder_name, ladder in config["model_ladders"].items():
            models = " ".join(ladder.get("models", []))
            print(f"  {ladder_name}: {models}")
        print()
    print("Papers:")
    for paper, paper_cfg in config["papers"].items():
        print(f"  {paper} ({paper_cfg.get('workspace', '.')})")
        for claim_id, claim in paper_cfg.get("claims", {}).items():
            print(f"    {claim_id}: {claim.get('title', claim_id)} [{claim.get('route')}]")


def read_secret_names(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if value.strip():
            names.add(key.strip())
    return names


def require_secrets(secret_names: set[str], required: list[str], secrets_file: Path, execute: bool) -> None:
    missing = [name for name in required if name not in secret_names and not os.environ.get(name)]
    if missing:
        if not execute:
            print(
                "[dry-run warning] Missing secret names "
                f"{', '.join(missing)} in env or {secrets_file}. Execution would fail until they are set."
            )
            return
        raise SystemExit(
            "Missing required secret names "
            f"{', '.join(missing)} in env or {secrets_file}. Values were not inspected or printed."
        )


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_command(command: str, cwd: Path, execute: bool) -> int:
    print(f"[cwd] {cwd}")
    print(command)
    if not execute:
        return 0
    return subprocess.run(command, cwd=cwd, shell=True, check=False).returncode


def enforce_execution_gate(workspace: Path, route: str, execute: bool) -> None:
    if not execute:
        return
    state = derive_state(workspace)
    local_allowed = {"plan_gated", "smoke_green", "running", "evidence_compiled"}
    remote_allowed = {"smoke_green", "running", "evidence_compiled"}
    allowed = local_allowed if route == "local_command" else remote_allowed
    if state.stage not in allowed:
        raise SystemExit(
            f"Execution gate blocked {route} at stage {state.stage!r}: "
            f"{state.reason} Next action: {state.next_action}."
        )


def run_postprocess(claim: dict[str, Any], cwd: Path, execute: bool) -> int:
    command = claim.get("postprocess_command")
    if not command:
        return 0
    print("[postprocess]")
    return run_command(command, cwd, execute)


def prefix_env_command(command: str, env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items() if value)
    return f"{prefix} {command}" if prefix else command


def resolve_workspace(paper_cfg: dict[str, Any]) -> Path:
    raw = Path(str(paper_cfg.get("workspace", ".")))
    if raw.is_absolute():
        return raw.resolve()
    config_dir = Path(str(paper_cfg.get("_config_dir", ROOT)))
    local = (config_dir / raw).resolve()
    if local.exists() or raw == Path("."):
        return local
    return (ROOT / raw).resolve()


def expand_placeholders(value: str, workspace: Path, secrets_file: Path) -> str:
    return (
        value.replace("{root}", str(ROOT))
        .replace("{workspace}", str(workspace))
        .replace("{secrets_file}", str(secrets_file))
        .replace("{python}", shlex.quote(sys.executable))
    )


def resolve_env(value: Any, workspace: Path, secrets_file: Path) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    text = str(value)
    return (
        expand_placeholders(text, workspace, secrets_file)
    )


def claim_env(claim: dict[str, Any], workspace: Path, secrets_file: Path) -> dict[str, str]:
    return {key: resolve_env(value, workspace, secrets_file) for key, value in claim.get("env", {}).items()}


def hf_job_command(claim: dict[str, Any], workspace: Path, secrets_file: Path, extra_env: dict[str, str] | None = None) -> str:
    hf_bin = claim.get("hf_bin", str(ROOT / ".venv" / "bin" / "hf"))
    parts = [
        hf_bin,
        "jobs",
        "run",
    ]
    if claim.get("detach", True):
        parts.append("--detach")
    parts.extend(["--flavor", str(claim.get("flavor", "cpu-basic"))])
    parts.extend(["--timeout", str(claim.get("timeout", "2h"))])
    if claim.get("secrets_file", True):
        parts.extend(["--secrets-file", str(secrets_file)])

    env = claim_env(claim, workspace, secrets_file)
    env.update(extra_env or {})
    for key, value in env.items():
        parts.extend(["-e", f"{key}={value}"])

    for mount in claim.get("mounts", []):
        host = expand_placeholders(str(mount["host"]), workspace, secrets_file)
        container = str(mount["container"])
        mode = mount.get("mode")
        spec = f"{host}:{container}"
        if mode:
            spec = f"{spec}:{mode}"
        parts.extend(["-v", spec])

    parts.append(str(claim.get("image", "python:3.12")))
    remote_command = expand_placeholders(claim["remote_command"], workspace, secrets_file)
    parts.extend(["bash", "-lc", remote_command])
    return shell_join(parts)


def matrix_values(claim: dict[str, Any], models: list[str] | None) -> dict[str, list[str]]:
    matrix = {key: [str(v) for v in value] for key, value in claim.get("matrix", {}).items()}
    if models:
        matrix["MODEL"] = [str(model) for model in models]
    return matrix


def matrix_env_batches(claim: dict[str, Any], models: list[str] | None) -> list[dict[str, str]]:
    matrix = matrix_values(claim, models)
    if not matrix:
        return [{}]
    keys = list(matrix)
    values = [matrix[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def route_models(
    config: dict[str, Any],
    claim: dict[str, Any],
    model_ladder_override: str | None,
    model_override: list[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    ladder_name = model_ladder_override or claim.get("model_ladder")
    ladder = config.get("model_ladders", {}).get(ladder_name, {}) if ladder_name else {}
    models = model_override or ladder.get("models", [])
    return [str(model) for model in models], ladder


def run_matrix_hf_job(claim: dict[str, Any], workspace: Path, secrets_file: Path, models: list[str] | None, execute: bool) -> int:
    batches = matrix_env_batches(claim, models)
    rc = 0
    for idx, env in enumerate(batches, start=1):
        print(f"[matrix {idx}/{len(batches)}] {env}")
        command = hf_job_command(claim, workspace, secrets_file, extra_env=env)
        rc = run_command(command, workspace, execute)
        if rc != 0 and execute:
            return rc
    return rc


def select_claims(config: dict[str, Any], paper: str, claim_id: str | None, all_claims: bool) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if paper not in config["papers"]:
        raise SystemExit(f"Unknown paper {paper!r}. Use --list to see available routes.")
    paper_cfg = config["papers"][paper]
    claims = paper_cfg.get("claims", {})
    if all_claims:
        return [(cid, claim, paper_cfg) for cid, claim in claims.items()]
    selected = claim_id or "all"
    if selected not in claims:
        raise SystemExit(f"Unknown claim {selected!r} for paper {paper!r}. Use --list to see available routes.")
    return [(selected, claims[selected], paper_cfg)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", help="Route config; defaults to core plus paper-local configs")
    parser.add_argument("--paper", help="Paper key from a paper-local claim_routes.json")
    parser.add_argument("--claim", help="Claim key; defaults to 'all' when present")
    parser.add_argument("--all", action="store_true", help="Run all configured claims for --paper")
    parser.add_argument("--list", action="store_true", help="List available routes")
    parser.add_argument("--execute", action="store_true", help="Actually run commands / submit HF jobs")
    parser.add_argument("--secrets-file", default=str(DEFAULT_SECRETS))
    parser.add_argument("--model-ladder", help="Override the claim's configured model ladder")
    parser.add_argument("--models", nargs="+", help="Override models for an LLM matrix route")
    args = parser.parse_args()

    config_paths = [Path(value) for value in args.config] if args.config else discover_configs()
    config = load_configs(config_paths)
    if args.list:
        list_routes(config)
        return 0
    if not args.paper:
        raise SystemExit("Pass --paper or --list.")

    secrets_file = Path(args.secrets_file).resolve()
    secret_names = read_secret_names(secrets_file)
    selected = select_claims(config, args.paper, args.claim, args.all)
    exit_code = 0

    for claim_id, claim, paper_cfg in selected:
        print(f"\n=== {args.paper}/{claim_id}: {claim.get('title', claim_id)} ===")
        print(f"Route: {claim.get('route')}")
        if claim.get("confirmation_tests"):
            print("Confirmation tests:")
            for item in claim["confirmation_tests"]:
                print(f"  - {item}")
        if claim.get("counterexample_search"):
            print("Counterexample search:")
            for item in claim["counterexample_search"]:
                print(f"  - {item}")

        route = claim.get("route")
        workspace = resolve_workspace(paper_cfg)
        enforce_execution_gate(workspace, str(route), args.execute)
        if route == "local_command":
            command = expand_placeholders(claim["command"], workspace, secrets_file)
            rc = run_command(command, workspace, args.execute)
            if rc == 0:
                postprocess = claim.get("postprocess_command")
                rc = run_command(expand_placeholders(postprocess, workspace, secrets_file), workspace, args.execute) if postprocess else 0
        elif route == "hf_job_command":
            models, ladder = route_models(config, claim, args.model_ladder, args.models)
            require_secrets(
                secret_names,
                claim.get("secret_requirements", []) + ladder.get("secret_requirements", []),
                secrets_file,
                args.execute,
            )
            extra_env = {}
            if models:
                extra_env[claim.get("model_env", "MODEL_LIST")] = " ".join(models)
            command = hf_job_command(claim, workspace, secrets_file, extra_env=extra_env)
            rc = run_command(command, workspace, args.execute)
        elif route == "matrix_hf_job":
            models, ladder = route_models(config, claim, args.model_ladder, args.models)
            require_secrets(
                secret_names,
                claim.get("secret_requirements", []) + ladder.get("secret_requirements", []),
                secrets_file,
                args.execute,
            )
            rc = run_matrix_hf_job(claim, workspace, secrets_file, models, args.execute)
        else:
            raise SystemExit(f"Unsupported route {route!r} for {args.paper}/{claim_id}.")

        if rc != 0:
            exit_code = rc
            if args.execute:
                break

    if not args.execute:
        print("\nDry run only. Re-run with --execute to launch commands/jobs.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
