"""Visualize TileBench benchmark results with derived performance metrics.

Typical usage (run from Tilebench/):
    # Run benchmark first:
    PYTHONPATH=. python scripts/run_bench.py --operator mul2

    # Then visualize (paths are inferred from --operator automatically):
    PYTHONPATH=. python scripts/visualize.py --operator mul2

    # Override paths or metrics explicitly:
    PYTHONPATH=. python scripts/visualize.py --operator mul2 \\
        --input  results/logs/time_measurement_logs/mul2_results.json \\
        --output-dir results/figures/mul2/ \\
        --metrics latency_ms bandwidth_GBs speedup pct_peak_bw

Default paths (derived from --operator):
    --input      results/logs/time_measurement_logs/<operator>_results.json
    --output-dir results/figures/<operator>/

Available derived metrics (from core/metrics.py):
    latency_ms          raw mean latency (always available)
    bandwidth_GBs       memory bandwidth (needs bytes_expr)
    pct_peak_bw         % of peak HBM bandwidth (needs bytes_expr + peak_bw_GBs)
    tflops              arithmetic throughput (needs flops_expr)
    pct_peak_tflops     % of peak TFLOPS (needs flops_expr + peak_tflops map)
    arithmetic_intensity FLOP/Byte ratio (needs both expressions)
    speedup             vs PyTorch baseline (non-torch backends only)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import yaml

# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without requiring PYTHONPATH=.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np                      # noqa: E402
import matplotlib                       # noqa: E402
matplotlib.use("Agg")                   # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.ticker as ticker      # noqa: E402

from tilebench.core.metrics import applicable_backends, compute_derived, load_peak_config  # noqa: E402
from tilebench.paths import operator_config  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

BACKENDS = ["torch", "triton", "cutile", "tilelang", "nki"]

_STYLE: dict[str, dict] = {
    "torch":  {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "PyTorch"},
    "triton": {"color": "#2ca02c", "marker": "s", "linestyle": "--", "label": "Triton"},
    "cutile": {"color": "#ff7f0e", "marker": "^", "linestyle": ":",  "label": "cuTile"},
    "tilelang": {"color": "#9467bd", "marker": "D", "linestyle": "-.", "label": "TileLang"},
    "nki":    {"color": "#d62728", "marker": "v", "linestyle": (0, (3, 1, 1, 1)), "label": "NKI (Trainium)"},
}

_METRIC_LABEL: dict[str, str] = {
    "latency_ms":           "Latency (ms)",
    "bandwidth_GBs":        "Memory Bandwidth (GB/s)",
    "tflops":               "Throughput (TFLOPS)",
    "speedup":              "Speedup over PyTorch (×)",
    "pct_peak_bw":          "% of Peak Bandwidth",
    "pct_peak_tflops":      "% of Peak TFLOPS",
    "arithmetic_intensity": "Arithmetic Intensity (FLOP/Byte)",
}

_DEFAULT_METRICS = [
    "latency_ms", "bandwidth_GBs", "tflops", "speedup",
    "pct_peak_bw", "pct_peak_tflops", "roofline",
]
_GPU_PEAK_METRICS = ["pct_peak_bw", "pct_peak_tflops", "roofline"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_operator_config(operator: str) -> dict:
    path = operator_config(operator)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dedupe_preserve_order(metrics: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for metric in metrics:
        if metric in seen:
            continue
        seen.add(metric)
        deduped.append(metric)
    return deduped


def _group_by_dtype(
    results_with_derived: list[tuple[dict, dict]]
) -> dict[str, list[tuple[float, dict, dict]]]:
    """Return {dtype: [(n_elements, raw_result, derived_metrics), ...]} sorted by n."""
    grouped: dict[str, list] = {}
    for raw, derived in results_with_derived:
        dtype = raw.get("dtype", "unknown")
        n = int(raw.get("params", {}).get("n", raw.get("problem_size", 0)))
        grouped.setdefault(dtype, []).append((n, raw, derived))
    for dtype in grouped:
        grouped[dtype].sort(key=lambda t: t[0])
    return grouped


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def _plot_one_metric(
    grouped: dict[str, list[tuple[float, dict, dict]]],
    metric: str,
    operator: str,
    output_dir: str,
) -> None:
    dtypes = sorted(grouped.keys())
    if not dtypes:
        return

    ncols = min(3, len(dtypes))
    nrows = math.ceil(len(dtypes) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
    )
    ylabel = _METRIC_LABEL.get(metric, metric)
    fig.suptitle(
        f"{operator}  —  {ylabel}  vs  Problem Size",
        fontsize=14, fontweight="bold", y=1.01,
    )

    any_data_in_figure = False

    for ax_idx, dtype in enumerate(dtypes):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        rows = grouped[dtype]

        x_vals = np.array([n / 1e6 for n, _, _ in rows], dtype=np.float64)

        any_plotted = False
        for backend in BACKENDS:
            y_vals = np.array(
                [r_derived.get(backend, {}).get(metric, float("nan")) for _, _, r_derived in rows],
                dtype=np.float64,
            )
            if np.all(np.isnan(y_vals)):
                continue
            st = _STYLE[backend]
            ax.plot(
                x_vals, y_vals,
                marker=st["marker"], linestyle=st["linestyle"],
                color=st["color"], label=st["label"],
                linewidth=2, markersize=6,
            )
            any_plotted = True

        ax.set_title(f"dtype = {dtype}", fontsize=12)
        ax.set_xlabel("Problem size (M elements)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        if any_plotted:
            ax.legend(fontsize=10)
            any_data_in_figure = True
        # Use log scale for latency (spans orders of magnitude)
        if metric == "latency_ms":
            ax.set_yscale("log")
        # Add reference line at 100 % for pct metrics
        if metric in ("pct_peak_bw", "pct_peak_tflops"):
            ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="100 % peak")
        # Speedup: add 1× reference line
        if metric == "speedup":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="1× (parity)")

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))

    # Hide unused subplots
    for ax_idx in range(len(dtypes), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    if not any_data_in_figure:
        plt.close(fig)
        print(f"  [skip] No data available for metric '{metric}'")
        return

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{operator}_{metric}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# roofline plot
# ---------------------------------------------------------------------------

# Jitter scale for X axis (separates backend points at the same AI value)
_JITTER_FACTORS = {"torch": 0.76, "triton": 0.88, "cutile": 1.0, "tilelang": 1.12, "nki": 1.24}


def _plot_roofline(
    grouped: dict[str, list[tuple[float, dict, dict]]],
    metrics_cfg: dict,
    operator: str,
    output_dir: str,
) -> None:
    """Classic Roofline plot: attained TFLOPS vs Arithmetic Intensity.

    X axis: Arithmetic Intensity (FLOP/Byte, log scale)
    Y axis: Attained TFLOPS (log scale)
    Roof lines: memory bandwidth slope + per-dtype compute ceiling
    Data points: one per (backend, n) per dtype subplot
    """
    peak_bw_GBs = metrics_cfg.get("peak_bw_GBs")
    peak_tflops_map = metrics_cfg.get("peak_tflops", {})

    if not peak_bw_GBs:
        print("  [skip roofline] peak_bw_GBs not available (pass --gpu to load peak data)")
        return

    peak_bw_TBps = float(peak_bw_GBs) / 1000.0  # GB/s → TB/s

    dtypes = sorted(grouped.keys())
    if not dtypes:
        return

    ncols = min(3, len(dtypes))
    nrows = math.ceil(len(dtypes) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"{operator}  —  Roofline Model",
        fontsize=14, fontweight="bold", y=1.01,
    )

    any_data_in_figure = False

    # Only plot backends that actually ran on the hardware metrics_cfg's
    # peak_* values describe -- otherwise this would be a cross-hardware
    # comparison (see PR #102 review). metrics_cfg here is the peak_cfg merged
    # over the operator's metrics config, so a peak file's declared "backends"
    # (e.g. Trainium2.json -> ["torch", "nki"]) takes effect automatically.
    roofline_backends = [b for b in BACKENDS if b in applicable_backends(metrics_cfg)]

    for ax_idx, dtype in enumerate(dtypes):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        rows = grouped[dtype]

        # Collect (AI, TFLOPS) for each backend
        points: dict[str, tuple[list, list]] = {b: ([], []) for b in roofline_backends}
        ai_vals = []
        for _, _, derived in rows:
            for backend in roofline_backends:
                ai = derived.get(backend, {}).get("arithmetic_intensity")
                tf = derived.get(backend, {}).get("tflops")
                if ai and tf and not math.isnan(tf):
                    points[backend][0].append(ai)
                    points[backend][1].append(tf)
                    ai_vals.append(ai)

        if not ai_vals:
            ax.set_visible(False)
            continue

        ai_center = float(np.median(ai_vals))

        # --- Roof lines ---
        # X range: cover at least 3 decades around ai_center
        peak_tf = float(peak_tflops_map.get(dtype, 0)) if peak_tflops_map else 0
        x_ridge = (peak_tf / peak_bw_TBps) if (peak_tf and peak_bw_TBps) else ai_center * 100
        x_min = min(ai_center, x_ridge) * 0.05
        x_max = x_ridge * 20
        x_roof = np.logspace(np.log10(x_min), np.log10(x_max), 300)

        # Memory bandwidth roof: y = peak_bw_TBps × x
        y_mem_roof = peak_bw_TBps * x_roof
        if peak_tf:
            y_roof = np.minimum(y_mem_roof, peak_tf)
            ax.plot(x_roof, y_roof, color="black", linewidth=2,
                    linestyle="-", label="Roofline", zorder=1)
            # Label the two segments
            ax.annotate(
                f"Mem BW ({peak_bw_GBs:.0f} GB/s)",
                xy=(ai_center * 0.3, peak_bw_TBps * ai_center * 0.3),
                fontsize=8, color="gray",
                rotation=np.degrees(np.arctan(np.log10(peak_bw_TBps))),
            )
            ax.axhline(peak_tf, color="red", linewidth=2.5, linestyle="--",
                       alpha=0.85, label=f"Peak Performance ({peak_tf:.0f} TFLOPS)")
            # Ridge point
            ax.axvline(x_ridge, color="gray", linewidth=0.8, linestyle=":",
                       alpha=0.5, label=f"Ridge ({x_ridge:.1f} FLOP/B)")
        else:
            ax.plot(x_roof, y_mem_roof, color="black", linewidth=2,
                    linestyle="-", label=f"Mem BW ({peak_bw_GBs:.0f} GB/s)", zorder=1)

        # --- Data points (jitter X so backends don't overlap) ---
        for backend in roofline_backends:
            xs, ys = points[backend]
            if not xs:
                continue
            jitter = _JITTER_FACTORS.get(backend, 1.0)
            xs_j = [x * jitter for x in xs]
            st = _STYLE[backend]
            ax.scatter(xs_j, ys,
                       marker=st["marker"], color=st["color"],
                       label=st["label"], s=50, zorder=5, alpha=0.85)
            any_data_in_figure = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=11)
        ax.set_ylabel("Attained Performance (TFLOPS)", fontsize=11)
        ax.set_title(f"dtype = {dtype}", fontsize=12)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)

    # Hide unused subplots
    for ax_idx in range(len(dtypes), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    if not any_data_in_figure:
        plt.close(fig)
        print("  [skip roofline] No TFLOPS/AI data available")
        return

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{operator}_roofline.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize TileBench results with derived metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--operator", type=str, required=True,
                        help="Operator name (e.g. mul2) — used to locate config.yaml "
                             "and derive default input/output paths")
    parser.add_argument("--input", type=str, default=None,
                        help="Timing results JSON produced by run_bench.py "
                             "(default: results/logs/time_measurement_logs/<operator>_results.json)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Metrics to plot (default: from config.yaml metrics.plots)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write PNG files "
                             "(default: results/figures/<operator>/)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU short name (e.g. B200). Loads peak performance from "
                             "tilebench/data/peak_performance/<GPU>.json for roofline and pct_peak metrics.")
    args = parser.parse_args()

    # Resolve operator-bound default paths
    input_path = args.input or (
        f"results/logs/time_measurement_logs/{args.operator}_results.json"
    )
    output_dir = args.output_dir or f"results/figures/{args.operator}"

    # Load data
    with open(input_path) as f:
        results: list[dict] = json.load(f)
    print(f"Loaded {len(results)} result(s) from {input_path}")

    # Load operator config for metric expressions
    config = _load_operator_config(args.operator)
    metrics_cfg: dict = config.get("metrics", {})

    # Load GPU-specific peak performance data
    peak_cfg: dict = {}
    if args.gpu:
        peak_cfg = load_peak_config(args.gpu)
        if not peak_cfg:
            print(f"Warning: no peak data found for GPU '{args.gpu}' at "
                  f"tilebench/data/peak_performance/{args.gpu}.json — "
                  f"roofline/pct_peak will be skipped")

    # Determine which metrics to plot
    if args.metrics:
        metrics_to_plot = args.metrics
    elif metrics_cfg.get("plots"):
        metrics_to_plot = list(metrics_cfg["plots"])
        if peak_cfg:
            metrics_to_plot.extend(_GPU_PEAK_METRICS)
        metrics_to_plot = _dedupe_preserve_order(metrics_to_plot)
    else:
        metrics_to_plot = _DEFAULT_METRICS
    print(f"Metrics to plot: {metrics_to_plot}")

    # Compute derived metrics for every result entry
    results_with_derived = [
        (r, compute_derived(r, metrics_cfg, peak_cfg=peak_cfg)) for r in results
    ]

    # Group by dtype once; reuse for all metrics
    grouped = _group_by_dtype(results_with_derived)
    dtypes_found = sorted(grouped.keys())
    print(f"dtypes found in results: {dtypes_found}")

    # One figure per metric
    for metric in metrics_to_plot:
        if metric == "roofline":
            roofline_cfg = {**metrics_cfg, **peak_cfg}
            _plot_roofline(grouped, roofline_cfg, args.operator, output_dir)
        else:
            _plot_one_metric(grouped, metric, args.operator, output_dir)

    print(f"\nAll plots written to: {output_dir}")


if __name__ == "__main__":
    main()
