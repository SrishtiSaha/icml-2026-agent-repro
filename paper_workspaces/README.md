# Paper workspaces

Each `repro_*` or `repro-*` directory is an independent paper reproduction. Paper-specific code, routes, model ladders, launchers, data, evidence, Trackio pages, posters, and outputs stay inside that directory.

New workspaces are created from the repository root with:

```bash
uv run python scripts/scaffold_repro_paper.py --title "Paper title" --slug paper_slug
```

The core harness discovers paper-local `claim_routes.json` files automatically. Do not add named papers to `../configs/claim_routes.json` or special-case them in `../agent_harness/`.

## Archived workspaces

Only the 13 active reference workspaces are kept expanded in this directory. The other 248 historical workspaces and three former loose files are stored losslessly in `archived_paper_workspaces.zip`. The archive is intentionally ignored by Git because it is approximately 58 GiB.

Restore one archived workspace in place with:

```bash
unzip archived_paper_workspaces.zip 'repro_example_workspace/*' -d .
```

The archive passed a complete ZIP CRC test before the expanded sources were removed.
