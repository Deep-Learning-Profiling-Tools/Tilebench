#!/usr/bin/env python3
"""Plot per-operator TileLang comparison figures from dashboard CSV data."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DASHBOARD = Path(__file__).resolve().parents[1]
INPUT = DASHBOARD / "data" / "operators.csv"
TILEBENCH = DASHBOARD.parent
if not (TILEBENCH / "results").is_dir():
    TILEBENCH = DASHBOARD.parent / "Tilebench"
OUTPUT = TILEBENCH / "results" / "figures" / "comparison"


COMPARISONS = (
    (
        "tilelang_vs_torch",
        "default_tilelang",
        "autotune_tilelang",
        "TileLang speedup relative to Torch (x)",
    ),
    (
        "tilelang_vs_triton",
        "default_tl_over_triton",
        "autotune_tl_over_triton",
        "TileLang speedup relative to Triton (x)",
    ),
    (
        "tilelang_vs_cutile",
        "default_tl_over_cutile",
        "autotune_tl_over_cutile",
        "TileLang speedup relative to CuTile (x)",
    ),
)


def parse_ratio(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value:
        return None
    return float(value)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_comparison(
    rows: list[dict[str, str]],
    stem: str,
    default_key: str,
    autotune_key: str,
    xlabel: str,
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: parse_ratio(row, autotune_key) or parse_ratio(row, default_key) or 0.0,
    )

    labels = [row["op"] for row in ordered]
    default = [parse_ratio(row, default_key) for row in ordered]
    autotune = [parse_ratio(row, autotune_key) for row in ordered]
    y = list(range(len(ordered)))

    fig, ax = plt.subplots(figsize=(7.2, 10.0))
    bar_height = 0.34

    default_pairs = [(value, index) for index, value in enumerate(default) if value is not None]
    tuned_pairs = [(value, index) for index, value in enumerate(autotune) if value is not None]
    ax.barh(
        [index - bar_height / 2 for _, index in default_pairs],
        [value for value, _ in default_pairs],
        height=bar_height,
        color="#0072B2",
        label="Default",
    )
    ax.barh(
        [index + bar_height / 2 for _, index in tuned_pairs],
        [value for value, _ in tuned_pairs],
        height=bar_height,
        color="#D55E00",
        label="Autotune",
    )

    values = [value for value in (*default, *autotune) if value is not None]
    max_value = max(values)
    ax.axvline(1.0, color="#222222", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlim(0, max(1.05, max_value * 1.08))
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#D9DDE3", linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    for comparison in COMPARISONS:
        plot_comparison(rows, *comparison)
    print(f"wrote {len(COMPARISONS)} comparison figures for {len(rows)} operators to {OUTPUT}")


if __name__ == "__main__":
    main()
