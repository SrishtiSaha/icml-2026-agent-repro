from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import append_jsonl, utc_now


def record_event(workspace: Path, kind: str, **payload: Any) -> None:
    append_jsonl(
        workspace / ".repro" / "events.jsonl",
        {"timestamp": utc_now(), "kind": kind, **payload},
    )
