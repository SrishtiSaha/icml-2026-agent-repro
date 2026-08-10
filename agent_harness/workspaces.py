from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_WORKSPACES = ROOT / "paper_workspaces"
WORKSPACE_PREFIXES = ("repro_", "repro-")


def discover_workspaces() -> list[Path]:
    if not PAPER_WORKSPACES.exists():
        return []
    return sorted(
        path for path in PAPER_WORKSPACES.iterdir()
        if path.is_dir() and path.name.startswith(WORKSPACE_PREFIXES)
    )


def resolve_workspace(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    nested = PAPER_WORKSPACES / candidate
    if nested.exists():
        return nested.resolve()
    raise FileNotFoundError(f"Paper workspace not found: {value}")


def control_dir(workspace: Path) -> Path:
    return workspace / ".repro"
