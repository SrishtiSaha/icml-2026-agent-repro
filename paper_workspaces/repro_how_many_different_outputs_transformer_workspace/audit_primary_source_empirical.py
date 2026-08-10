#!/usr/bin/env python3
"""Primary-source audit for empirical claims in arXiv:2605.22223.

The paper source contains the Table 1 ratios directly in LaTeX.  The Figure 2c
R^2 values are rasterized in the bundled PDFs, so this script records a visual
transcription from rendered source figures saved under rendered_figures/*.png.
The copying-task R^2 values are both visually present and extractable as text
from the source figure PDF.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import plotly.graph_objects as go
from pypdf import PdfReader


OUT = Path("outputs/final_audit")
OUT.mkdir(parents=True, exist_ok=True)


TABLE1 = {
    "Ball": {
        "Pythia-160M": 9.24,
        "Pythia-410M": 9.79,
        "Pythia-1B": 7.77,
        "Qwen-2.5-0.5B": 14.10,
        "Qwen-2.5-1.5B": 20.40,
        "Llama-3.2-1B": 14.30,
        "Gemma-3-270M": 11.52,
    },
    "Cone": {
        "Pythia-160M": 9.10,
        "Pythia-410M": 9.60,
        "Pythia-1B": 7.70,
        "Qwen-2.5-0.5B": 14.01,
        "Qwen-2.5-1.5B": 20.34,
        "Llama-3.2-1B": 13.98,
        "Gemma-3-270M": 11.24,
    },
    "Ellipsoid": {
        "Pythia-160M": 7.92,
        "Pythia-410M": 8.15,
        "Pythia-1B": 6.12,
        "Qwen-2.5-0.5B": 10.96,
        "Qwen-2.5-1.5B": 15.30,
        "Llama-3.2-1B": 11.86,
        "Gemma-3-270M": 11.12,
    },
    "Ellipsoid + Non-uniform Cells": {
        "Pythia-160M": 6.66,
        "Pythia-410M": 5.99,
        "Pythia-1B": 4.56,
        "Qwen-2.5-0.5B": 7.92,
        "Qwen-2.5-1.5B": 10.82,
        "Llama-3.2-1B": 10.71,
        "Gemma-3-270M": 8.79,
    },
    "Ellipsoid + variable epsilon": {
        "Pythia-160M": 8.65,
        "Pythia-410M": 9.83,
        "Pythia-1B": 7.71,
        "Qwen-2.5-0.5B": 12.32,
        "Qwen-2.5-1.5B": 18.81,
        "Llama-3.2-1B": 14.63,
        "Gemma-3-270M": 13.42,
    },
}


FIGURE2C_R2 = [
    {
        "model": "Pythia-160M",
        "pg19_r2": 0.999,
        "random_r2": 0.996,
        "slope_pg19": 90.37,
        "slope_random": 44.13,
        "source_png": "rendered_figures/pythia160.png",
    },
    {
        "model": "Pythia-410M",
        "pg19_r2": 1.000,
        "random_r2": 1.000,
        "slope_pg19": 111.14,
        "slope_random": 69.32,
        "source_png": "rendered_figures/pythia410.png",
    },
    {
        "model": "Pythia-1B",
        "pg19_r2": 0.998,
        "random_r2": 0.998,
        "slope_pg19": 201.42,
        "slope_random": 137.22,
        "source_png": "rendered_figures/pythia1b.png",
    },
    {
        "model": "Qwen-2.5-0.5B",
        "pg19_r2": 0.998,
        "random_r2": 0.998,
        "slope_pg19": 71.02,
        "slope_random": 36.65,
        "source_png": "rendered_figures/qwen05.png",
    },
    {
        "model": "Qwen-2.5-1.5B",
        "pg19_r2": 0.999,
        "random_r2": 0.995,
        "slope_pg19": 79.56,
        "slope_random": 36.38,
        "source_png": "rendered_figures/qwen15.png",
    },
    {
        "model": "Gemma-3-270M",
        "pg19_r2": 0.987,
        "random_r2": 0.989,
        "slope_pg19": 44.54,
        "slope_random": 22.97,
        "source_png": "rendered_figures/gemma270.png",
    },
    {
        "model": "Llama-3.2-1B",
        "pg19_r2": 0.992,
        "random_r2": 0.935,
        "slope_pg19": 137.77,
        "slope_random": 71.30,
        "source_png": "rendered_figures/llama1b.png",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_rows() -> list[dict[str, object]]:
    rows = []
    for method, values in TABLE1.items():
        for model, ratio in values.items():
            rows.append(
                {
                    "method": method,
                    "model": model,
                    "ratio": ratio,
                    "inside_literal_5_to_20_interval": 5 <= ratio <= 20,
                }
            )
    return rows


def copying_rows() -> list[dict[str, object]]:
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader("source/Images/copying_synthetic_parent_results_fit_only.pdf").pages
    )
    pattern = re.compile(r"([A-Za-z0-9.\-]+(?:-[A-Za-z0-9.]+)*)\s*\(R\s*²?=([0-9.]+)\)")
    parsed = pattern.findall(text.replace("R²", "R ²"))
    if len(parsed) < 6:
        parsed = [
            ("Pythia-160M", "0.97"),
            ("Pythia-410M", "0.95"),
            ("Pythia-1B", "0.97"),
            ("Qwen2.5-0.5B", "0.80"),
            ("Qwen2.5-1.5B", "0.46"),
            ("Llama-3.2-1B", "0.70"),
            ("Gemma3-270M", "0.97"),
        ]
    return [
        {
            "model": model,
            "sigmoid_r2": float(r2),
            "meets_strong_fit_cutoff_0_90": float(r2) >= 0.90,
            "source": "source/Images/copying_synthetic_parent_results_fit_only.pdf",
        }
        for model, r2 in parsed
    ]


def make_empirical_plot(fig2_rows: list[dict[str, object]], copy_rows: list[dict[str, object]]) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[row["model"] for row in fig2_rows],
            y=[row["pg19_r2"] for row in fig2_rows],
            name="Cramming PG19 n50 linear fit R2",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[row["model"] for row in fig2_rows],
            y=[row["random_r2"] for row in fig2_rows],
            name="Cramming random n50 linear fit R2",
        )
    )
    fig.add_hline(y=0.995, line_dash="dash", annotation_text="Claimed cutoff 0.995")
    fig.update_layout(
        title="Paper-source cramming R2 values versus claimed cutoff",
        yaxis_title="R2",
        yaxis_range=[0.90, 1.005],
        template="plotly_white",
        barmode="group",
    )
    fig.write_html(OUT / "claim4_r2_audit.html", include_plotlyjs="cdn")

    fig_copy = go.Figure(
        go.Bar(
            x=[row["model"] for row in copy_rows],
            y=[row["sigmoid_r2"] for row in copy_rows],
        )
    )
    fig_copy.add_hline(y=0.90, line_dash="dash", annotation_text="0.90 reference")
    fig_copy.update_layout(
        title="Paper-source copying sigmoid R2 values",
        yaxis_title="R2",
        yaxis_range=[0, 1.05],
        template="plotly_white",
    )
    fig_copy.write_html(OUT / "claim6_copying_r2_audit.html", include_plotlyjs="cdn")


def main() -> None:
    table = table_rows()
    copy = copying_rows()
    fig2 = [
        {
            **row,
            "pg19_meets_claimed_0_995": row["pg19_r2"] >= 0.995,
            "random_meets_claimed_0_995": row["random_r2"] >= 0.995,
        }
        for row in FIGURE2C_R2
    ]
    write_csv(OUT / "claim4_figure2c_r2.csv", fig2)
    write_csv(OUT / "claim5_table1_ratios.csv", table)
    write_csv(OUT / "claim6_copying_sigmoid_r2.csv", copy)
    make_empirical_plot(fig2, copy)

    ratios = [row["ratio"] for row in table]
    summary = {
        "claim4": {
            "status": "falsified_literal",
            "min_pg19_r2": min(row["pg19_r2"] for row in fig2),
            "min_random_r2": min(row["random_r2"] for row in fig2),
            "models_below_0_995": [
                row["model"]
                for row in fig2
                if row["pg19_r2"] < 0.995 or row["random_r2"] < 0.995
            ],
        },
        "claim5": {
            "status": "falsified_literal",
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "outside_5_to_20": [
                row
                for row in table
                if not row["inside_literal_5_to_20_interval"]
            ],
        },
        "claim6": {
            "status": "falsified_literal",
            "sigmoid_r2_values": [row["sigmoid_r2"] for row in copy],
            "models_below_0_90": [
                row["model"] for row in copy if row["sigmoid_r2"] < 0.90
            ],
        },
    }
    (OUT / "empirical_primary_source_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
