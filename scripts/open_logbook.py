from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a Trackio logbook and write ICML challenge metadata."
    )
    parser.add_argument("--title", required=True, help="Paper title")
    parser.add_argument("--openreview-id", required=True, help="OpenReview paper id")
    parser.add_argument("--arxiv-id", default="", help="arXiv id if available")
    args = parser.parse_args()

    subprocess.run(
        ["trackio", "logbook", "open", "--title", f"Repro - {args.title}"],
        check=True,
    )

    metadata_path = Path(".trackio/metadata.json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    paper = {"openreview_id": args.openreview_id}
    if args.arxiv_id:
        paper["arxiv_id"] = args.arxiv_id

    metadata["paper"] = paper
    metadata["tags"] = [
        "icml2026-repro",
        f"paper-{args.openreview_id}",
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()

