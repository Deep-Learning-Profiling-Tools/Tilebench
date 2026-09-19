"""
README figure: PyTorch / Triton / cuTile latency at each operator's sweep-max case.

For every operator this takes the largest swept case, the one recorded in
tilebench/profiling/ncu_catalogue.json (`default_params_per_dtype`), so the figure
uses the same inputs as the NCU profiles. The dtype is fp16 when the operator
sweeps it, otherwise the operator's first dtype (shown after the name).
Latencies are the autotuned Proton means from results/<gpu>/csv/<op>_autotune.csv.

Usage:
    python scripts/plot_sweep_max.py --gpu B200    # -> results/B200/figures/sweep_max_latency.png
    python scripts/plot_sweep_max.py --gpu B200 --output assets/sweep_max_latency.png   # README figure

The default output is inside the GPU's own namespace, so plotting another GPU
can never overwrite the README figure, which shows B200.
"""

from __future__ import annotations

import argparse
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without requiring PYTHONPATH=.
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.paths import (NCU_CATALOGUE, hardware_label,  # noqa: E402
                             results_csv_dir, results_figures_dir)

CATALOGUE = NCU_CATALOGUE

# label, CSV column, color
SERIES = [
    ("PyTorch", "torch_ms",  "#33B39F"),
    ("Triton",  "triton_ms", "#6376A0"),
    ("cuTile",  "cutile_ms", "#EB6F5D"),
]

# Journal-style figure: sans-serif 7 pt text at print size (180 mm wide),
# thin axes, no top/right spines, frameless legend.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})


def sweep_max_rows(gpu):
    """[(label, {column: latency_ms})], one per operator, from results/<gpu>/csv/."""
    csv_dir = results_csv_dir(gpu)
    out = []
    for entry in json.load(open(CATALOGUE)):
        op = entry["op"]
        dtype = "fp16" if "fp16" in entry["dtypes"] else entry["dtypes"][0]
        want = {k: str(v) for k, v in entry["default_params_per_dtype"][dtype].items()}
        with open(csv_dir / f"{op}_autotune.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["dtype"].strip() == dtype and all(
                want.get(k.strip()) == v.strip()
                for k, v in (kv.split("=", 1) for kv in r["params"].split(",")))]
        assert len(rows) == 1, f"{op}/{dtype}: {len(rows)} CSV rows match the catalogue case"
        label = op if dtype == "fp16" else f"{op} ({dtype})"
        out.append((label, {col: float(rows[0][col]) for _, col, _ in SERIES}))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--gpu", type=hardware_label, required=True, metavar="LABEL",
                        help="Hardware label (e.g. B200): reads results/<gpu>/csv/")
    parser.add_argument("--output", default=None,
                        help="PNG to write (default: results/<gpu>/figures/sweep_max_latency.png)")
    args = parser.parse_args(argv)
    out = args.output or str(results_figures_dir(args.gpu) / "sweep_max_latency.png")

    data = sorted(sweep_max_rows(args.gpu), key=lambda r: -r[1]["torch_ms"])
    half = math.ceil(len(data) / 2)
    vals = [v for _, d in data for v in d.values() if math.isfinite(v)]
    ylim = (10 ** math.floor(math.log10(min(vals))), 10 ** math.ceil(math.log10(max(vals))))

    fig, axes = plt.subplots(2, 1, figsize=(7.09, 5.0), facecolor="white")
    width = 0.27
    for ax, chunk in zip(axes, (data[:half], data[half:])):
        for si, (label, col, color) in enumerate(SERIES):
            xs = [i + (si - 1) * width for i in range(len(chunk))]
            ax.bar(xs, [d[col] for _, d in chunk], width=width, color=color,
                   edgecolor="white", linewidth=0.4, label=label)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.6, half - 0.4)
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([name for name, _ in chunk], rotation=40, ha="right")
        ax.set_ylabel("latency (ms, log scale)")
        ax.yaxis.grid(True, which="major", color="#E5E5E5", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].set_title(f"Latency at the sweep-max case on {args.gpu} (autotuned; lower is better)",
                      weight="bold", loc="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"wrote {out}  ({len(data)} operators)")


if __name__ == "__main__":
    main()
