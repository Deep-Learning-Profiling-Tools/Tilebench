#!/usr/bin/env python3
"""Plot per-operator TileLang speedups from result CSVs."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "results" / "csv"
OUTPUT = ROOT / "results" / "figures" / "comparison"
MODES = ("default", "autotune")

COMPARISONS = (
    ("tilelang_vs_torch", "torch_ms", "TileLang speedup relative to Torch (x)"),
    ("tilelang_vs_triton", "triton_ms", "TileLang speedup relative to Triton (x)"),
    ("tilelang_vs_cutile", "cutile_ms", "TileLang speedup relative to cuTile (x)"),
)


def parse_positive(value: str | None) -> float | None:
    if value is None or not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def operator_names() -> list[str]:
    return sorted(path.name.removesuffix("_default.csv") for path in CSV_DIR.glob("*_default.csv"))


def load_ratio(op: str, mode: str, baseline_key: str) -> float | None:
    path = CSV_DIR / f"{op}_{mode}.csv"
    if not path.exists():
        return None
    ratios = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            baseline = parse_positive(row.get(baseline_key))
            tilelang = parse_positive(row.get("tilelang_ms"))
            if baseline is not None and tilelang is not None:
                ratios.append(baseline / tilelang)
    return geometric_mean(ratios)


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


def plot_comparison(ops: list[str], stem: str, baseline_key: str, xlabel: str) -> None:
    rows = [
        {
            "op": op,
            "default": load_ratio(op, "default", baseline_key),
            "autotune": load_ratio(op, "autotune", baseline_key),
        }
        for op in ops
    ]
    rows.sort(key=lambda row: row["autotune"] or row["default"] or 0.0)

    labels = [row["op"] for row in rows]
    y = list(range(len(rows)))
    default_pairs = [(row["default"], index) for index, row in enumerate(rows) if row["default"] is not None]
    autotune_pairs = [(row["autotune"], index) for index, row in enumerate(rows) if row["autotune"] is not None]
    values = [value for value, _ in (*default_pairs, *autotune_pairs)]
    if not values:
        raise RuntimeError(f"no plottable values for {stem}")

    fig, ax = plt.subplots(figsize=(7.2, 10.0))
    bar_height = 0.34
    ax.barh(
        [index - bar_height / 2 for _, index in default_pairs],
        [value for value, _ in default_pairs],
        height=bar_height,
        color="#0072B2",
        label="Default",
    )
    ax.barh(
        [index + bar_height / 2 for _, index in autotune_pairs],
        [value for value, _ in autotune_pairs],
        height=bar_height,
        color="#D55E00",
        label="Autotune",
    )
    ax.axvline(1.0, color="#222222", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlim(0, max(1.05, max(values) * 1.08))
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#D9DDE3", linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ops = operator_names()
    if len(ops) != 45:
        raise RuntimeError(f"expected 45 operators, found {len(ops)}")
    configure_matplotlib()
    for comparison in COMPARISONS:
        plot_comparison(ops, *comparison)
    print(f"wrote {len(COMPARISONS)} comparison figures for {len(ops)} operators to {OUTPUT}")


if __name__ == "__main__":
    main()
