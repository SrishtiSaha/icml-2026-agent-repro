#!/usr/bin/env python3
"""Validate that a published Trackio static logbook contains fresh evidence.

This is a public read-back check: it does not inspect local files. Use it after
`trackio logbook publish`, `trackio logbook sync`, or a direct Space upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_FORBIDDEN = [
    "trackio-local-path://",
    "trackio-artifact://",
]


@dataclass
class FetchResult:
    url: str
    text: str


def static_base_from_space(space: str) -> str:
    if "/" not in space:
        raise SystemExit("--space must look like owner/name")
    owner, name = space.split("/", 1)
    return f"https://{owner.lower()}-{name.lower()}.static.hf.space/"


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "icml-2026-agent-repro-validator/1.0"})
    try:
        with urlopen(req, timeout=30) as response:
            body = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            return body.decode(charset, errors="replace")
    except HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} while fetching {url}") from exc
    except URLError as exc:
        raise SystemExit(f"Could not fetch {url}: {exc.reason}") from exc


def walk_pages(node: dict[str, Any]) -> list[str]:
    files: list[str] = []
    file_name = node.get("file")
    if isinstance(file_name, str):
        files.append(file_name)
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            files.extend(walk_pages(child))
    return files


def load_logbook(base_url: str) -> tuple[dict[str, Any], list[FetchResult]]:
    base = base_url.rstrip("/") + "/"
    logbook_url = urljoin(base, "logbook.json")
    logbook_text = fetch_text(logbook_url)
    try:
        data = json.loads(logbook_text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{logbook_url} is not valid JSON: {exc}") from exc

    results = [FetchResult(logbook_url, logbook_text)]
    page_files = sorted(set(walk_pages(data.get("root", {}))))
    for page_file in page_files:
        results.append(FetchResult(urljoin(base, page_file), fetch_text(urljoin(base, page_file))))
    return data, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--space", help="Published Space id, e.g. owner/repro-paper")
    source.add_argument("--url", help="Static Space URL, e.g. https://owner-name.static.hf.space/")
    parser.add_argument("--must-contain", action="append", default=[], help="String that must appear in the public logbook. Repeatable.")
    parser.add_argument("--must-not-contain", action="append", default=[], help="String that must not appear. Repeatable.")
    parser.add_argument("--allow-local-placeholders", action="store_true", help="Do not forbid Trackio local placeholder artifact URLs.")
    parser.add_argument("--print-pages", action="store_true", help="Print fetched page URLs.")
    parser.add_argument("--report-json", help="Write the validation receipt used by repro-agent record-publish.")
    args = parser.parse_args()

    base_url = static_base_from_space(args.space) if args.space else args.url
    data, fetched = load_logbook(base_url)
    haystack = "\n".join(result.text for result in fetched)

    forbidden = list(args.must_not_contain)
    if not args.allow_local_placeholders:
        forbidden.extend(DEFAULT_FORBIDDEN)

    missing = [needle for needle in args.must_contain if needle not in haystack]
    present_forbidden = [needle for needle in forbidden if needle in haystack]

    if args.print_pages:
        for result in fetched:
            print(result.url)

    report = {
        "title": data.get("title") or data.get("root", {}).get("title"),
        "base_url": base_url,
        "fetched_files": len(fetched),
        "required_strings": len(args.must_contain),
        "missing_required": missing,
        "forbidden_strings": len(forbidden),
        "present_forbidden": present_forbidden,
        "public_content_sha256": hashlib.sha256(haystack.encode()).hexdigest(),
        "validated_at": datetime.now(UTC).isoformat(),
        "passed": not missing and not present_forbidden,
    }
    print(json.dumps(report, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
