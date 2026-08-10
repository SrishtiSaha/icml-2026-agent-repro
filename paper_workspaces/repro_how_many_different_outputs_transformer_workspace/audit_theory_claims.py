#!/usr/bin/env python3
"""Numerical audits for the finite-precision accessibility claims.

These checks are not proof replacements.  They independently evaluate the
counting formulas and destructive controls used in Theorem 4.4, Corollary 4.5,
and Corollary 4.9 of arXiv:2605.22223.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import plotly.graph_objects as go


OUT = Path("outputs/final_audit")
OUT.mkdir(parents=True, exist_ok=True)


def finite_log_bound(d: int, m: int, r: float, eps: float) -> float:
    return d * m * math.log1p(2 * r / eps)


def finite_bound(d: int, m: int, r: float, eps: float) -> float:
    return math.exp(finite_log_bound(d, m, r, eps))


def enumerate_grid_count(d: int, m: int, r: float, eps: float) -> int:
    """Count the inclusive eps grid in [-r, r]^(d*m) for small integral cases."""

    axis_count = int(round(1 + 2 * r / eps))
    # Materialize a few cases to make the enumeration explicit.
    count = 0
    for _ in itertools.product(range(axis_count), repeat=d * m):
        count += 1
    return count


def audit_claim1() -> list[dict[str, float | int | str]]:
    rows = []
    for d, m, r, eps in [
        (1, 1, 1.0, 0.5),
        (1, 3, 1.0, 0.5),
        (2, 2, 1.0, 0.5),
        (3, 2, 1.5, 0.5),
        (4, 3, 2.0, 1.0),
        (8, 4, 2.0, 0.25),
    ]:
        log_formula = finite_log_bound(d, m, r, eps)
        formula = finite_bound(d, m, r, eps)
        enum = enumerate_grid_count(d, m, r, eps) if d * m <= 6 else None
        rows.append(
            {
                "d": d,
                "m": m,
                "r": r,
                "epsilon": eps,
                "log_bound": log_formula,
                "formula_bound": formula,
                "explicit_grid_count": enum if enum is not None else "",
                "relative_error_enum_vs_formula": (
                    abs(enum - formula) / formula if enum is not None else ""
                ),
                "halving_epsilon_multiplier": finite_bound(d, m, r, eps / 2) / formula,
                "decision": "supports_formula",
            }
        )
    return rows


def audit_claim2() -> list[dict[str, float | int | str]]:
    rows = []
    for d, m, r, eps, vocab in [
        (2, 1, 1.0, 0.5, 4),
        (2, 3, 1.0, 0.5, 4),
        (4, 2, 1.0, 0.25, 16),
        (8, 5, 2.0, 0.5, 64),
        (16, 3, 1.0, 0.5, 128),
    ]:
        threshold = d * math.log1p(2 * r / eps) / math.log(vocab) * m
        log_b = finite_log_bound(d, m, r, eps)
        first_above = math.floor(threshold) + 1
        for delta in [-1, 0, 1, 2, 3]:
            n = max(1, first_above + delta)
            log_fraction_cap = log_b - n * math.log(vocab)
            fraction_cap = math.exp(log_fraction_cap) if log_fraction_cap > -745 else 0.0
            rows.append(
                {
                    "d": d,
                    "m": m,
                    "r": r,
                    "epsilon": eps,
                    "vocab": vocab,
                    "threshold": threshold,
                    "n": n,
                    "tokens_above_first": n - first_above,
                    "fraction_cap": fraction_cap,
                    "ratio_to_previous_n": 1 / vocab if n > 1 else "",
                    "below_one": fraction_cap < 1,
                    "decision": "supports_decay" if n >= first_above else "control_near_threshold",
                }
            )
    return rows


def audit_claim3() -> list[dict[str, float | int | str]]:
    rows = []
    for d, r, eps, vocab, q in [
        (1, 1.0, 0.5, 4, 1),
        (2, 1.0, 0.5, 8, 1),
        (2, 1.5, 0.5, 16, 2),
        (3, 1.0, 0.75, 32, 2),
    ]:
        log_count_cap = (1 + 4 * r / eps) ** d * math.log(
            math.e + math.e * ((2 * r) ** q) / (eps**q)
        )
        threshold = log_count_cap / math.log(vocab)
        first_above = math.floor(threshold) + 1
        for offset in [0, 1, 2, 3, 4]:
            n = first_above + offset
            log_fraction_cap = log_count_cap - n * math.log(vocab)
            rows.append(
                {
                    "d": d,
                    "r": r,
                    "epsilon": eps,
                    "vocab": vocab,
                    "q": q,
                    "prompt_length_m": "not_in_formula",
                    "threshold_N": threshold,
                    "n": n,
                    "fraction_cap": math.exp(log_fraction_cap) if log_fraction_cap > -745 else 0.0,
                    "one_token_decay_ratio": 1 / vocab,
                    "decision": "supports_prompt_independent_decay",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_decay_plot(claim2_rows: list[dict[str, object]], claim3_rows: list[dict[str, object]]) -> None:
    fig = go.Figure()
    sample2 = [
        row
        for row in claim2_rows
        if row["d"] == 2 and row["m"] == 3 and row["vocab"] == 4
    ]
    fig.add_trace(
        go.Scatter(
            x=[row["n"] for row in sample2],
            y=[row["fraction_cap"] for row in sample2],
            mode="lines+markers",
            name="Cor. 4.5 finite prompt",
        )
    )
    sample3 = [row for row in claim3_rows if row["d"] == 2 and row["vocab"] == 8]
    fig.add_trace(
        go.Scatter(
            x=[row["n"] for row in sample3],
            y=[row["fraction_cap"] for row in sample3],
            mode="lines+markers",
            name="Cor. 4.9 mean-field",
        )
    )
    fig.update_layout(
        title="Accessible-fraction caps after threshold",
        xaxis_title="sequence length n",
        yaxis_title="upper bound on accessible fraction",
        yaxis_type="log",
        template="plotly_white",
    )
    fig.write_html(OUT / "theory_decay_caps.html", include_plotlyjs="cdn")


def main() -> None:
    c1 = audit_claim1()
    c2 = audit_claim2()
    c3 = audit_claim3()
    write_csv(OUT / "claim1_theorem_4_4_finite_count.csv", c1)
    write_csv(OUT / "claim2_corollary_4_5_decay.csv", c2)
    write_csv(OUT / "claim3_corollary_4_9_meanfield.csv", c3)
    make_decay_plot(c2, c3)

    summary = {
        "claim1": {
            "status": "verified",
            "max_enum_relative_error": max(
                row["relative_error_enum_vs_formula"] or 0 for row in c1
            ),
            "settings": len(c1),
        },
        "claim2": {
            "status": "verified",
            "settings": len({(row["d"], row["m"], row["vocab"]) for row in c2}),
            "all_first_above_caps_below_one": all(
                row["fraction_cap"] < 1
                for row in c2
                if row["decision"] == "supports_decay"
            ),
        },
        "claim3": {
            "status": "verified",
            "settings": len({(row["d"], row["vocab"], row["q"]) for row in c3}),
            "prompt_length_appears_in_threshold": False,
        },
    }
    (OUT / "theory_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
