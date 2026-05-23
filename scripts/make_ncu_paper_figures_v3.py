#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    sns.set_theme(
        context="paper",
        style="white",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        },
    )


BACKEND_COLORS = {
    "triton": "#0072B2",
    "cutile": "#D55E00",
    "unknown": "#999999",
}

GROUP_ORDER = [
    "Severe/Moderate",
    "Mild",
    "Likely",
    "Confounded",
    "No direct conflict",
    "Other",
]

GROUP_COLORS = {
    "Severe/Moderate": "#7A0177",
    "Mild": "#C51B8A",
    "Likely": "#F768A1",
    "Confounded": "#FBB4B9",
    "No direct conflict": "#74A9CF",
    "Other": "#BDBDBD",
}


AUX_PATTERNS = [
    r"void\s+(at::)?vectorized_elementwise_kernel",
    r"void\s+unrolled_elementwise_kernel",
    r"copy_kernel_cuda",
    r"direct_copy_kernel_cuda",
    r"cudaMemcpy",
    r"cudaMemset",
    r"at::native",
    r"DeviceRadixSort",
    r"cub::",
    r"thrust::",
]


def infer_aux(kernel_name: str) -> bool:
    s = str(kernel_name)
    for pat in AUX_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def diag_group(diag: str) -> str:
    diag = str(diag)
    if diag in {"BANK_CONFLICT_CONFIRMED_SEVERE", "BANK_CONFLICT_CONFIRMED_MODERATE"}:
        return "Severe/Moderate"
    if diag == "BANK_CONFLICT_CONFIRMED_MILD":
        return "Mild"
    if "LIKELY" in diag:
        return "Likely"
    if "DIVERGENCE_CONFOUNDED" in diag:
        return "Confounded"
    if "NO_BANK_CONFLICT" in diag or "NO_SHARED" in diag or "NO_LSU" in diag:
        return "No direct conflict"
    return "Other"


def prepare(df: pd.DataFrame, include_aux: bool) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "bank_conflicts",
        "shared_wavefronts",
        "conflict_per_wavefront",
        "branch_efficiency_pct",
        "ipc_gap_ratio",
        "tma_inst_total",
        "tc_shared_wavefronts_total",
        "gpu_time_us",
        "branch_metric_valid",
        "profile_valid_for_tilebench_kernel",
        "selected_is_auxiliary",
    ]
    df = to_numeric(df, numeric_cols)

    if "selected_is_auxiliary" not in df.columns:
        df["selected_is_auxiliary"] = df["kernel_name"].map(infer_aux)

    if "profile_valid_for_tilebench_kernel" not in df.columns:
        df["profile_valid_for_tilebench_kernel"] = ~df["selected_is_auxiliary"]

    if not include_aux:
        df = df[df["profile_valid_for_tilebench_kernel"].astype(bool)].copy()

    df["conflict_pct"] = 100.0 * df["conflict_per_wavefront"]
    df["conflict_pct_plot"] = df["conflict_pct"].clip(lower=1e-3)
    df["diag_group"] = df["diagnosis"].map(diag_group)
    df["case"] = df["operator"].astype(str) + "/" + df["backend"].astype(str) + "/" + df["dtype"].astype(str)
    df["row_label"] = df["operator"].astype(str) + " / " + df["backend"].astype(str)

    # Branch validity fallback.
    if "branch_metric_valid" not in df.columns:
        df["branch_metric_valid"] = df["branch_efficiency_pct"].notna() & (df["branch_efficiency_pct"] > 0)
    else:
        df["branch_metric_valid"] = df["branch_metric_valid"].astype(bool)

    return df


def savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def plot_diagnosis_stacked(df, out_dir):
    counts = df.groupby(["backend", "diag_group"]).size().reset_index(name="count")
    pivot = counts.pivot(index="backend", columns="diag_group", values="count").fillna(0)
    cols = [c for c in GROUP_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(4.7, 1.65))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))

    for c in cols:
        vals = pivot[c].values
        ax.barh(
            y,
            vals,
            left=left,
            color=GROUP_COLORS[c],
            edgecolor="white",
            linewidth=0.5,
            label=c,
        )
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("# reports")
    ax.set_title("Automated NCU bank-conflict diagnosis")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        columnspacing=0.9,
        handlelength=1.0,
    )
    savefig(fig, out_dir, "fig_ncu_diagnosis_stacked")


def plot_top_bar(df, out_dir, top_k=20):
    d = df[np.isfinite(df["conflict_pct"])].sort_values("conflict_pct", ascending=False).head(top_k)
    d = d.sort_values("conflict_pct", ascending=True)

    fig_h = max(2.5, 0.20 * len(d) + 0.45)
    fig, ax = plt.subplots(figsize=(5.2, fig_h))

    colors = d["backend"].map(BACKEND_COLORS).fillna("#999999")

    ax.barh(
        np.arange(len(d)),
        d["conflict_pct"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.95,
    )

    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["case"])
    ax.set_xlabel("Bank conflicts / shared wavefront (%)")
    ax.set_title(f"Top {top_k} bank-conflict cases")
    ax.set_xscale("log")
    ax.grid(axis="x", which="both", alpha=0.25)

    handles = [
        mpl.patches.Patch(color=BACKEND_COLORS[b], label=b)
        for b in ["triton", "cutile"]
        if b in set(d["backend"])
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    savefig(fig, out_dir, "fig_ncu_top_conflict_bar")


def plot_top_heatmap(df, out_dir, top_k=25):
    d = df[np.isfinite(df["conflict_pct"])].copy()
    rank = d.groupby("row_label")["conflict_pct"].max().sort_values(ascending=False)
    top_rows = rank.head(top_k).index
    d = d[d["row_label"].isin(top_rows)]

    pivot = d.pivot_table(index="row_label", columns="dtype", values="conflict_pct", aggfunc="median")
    pivot["__rank__"] = pivot.max(axis=1)
    pivot = pivot.sort_values("__rank__", ascending=False).drop(columns="__rank__")

    dtype_order = [x for x in ["bf16", "fp16", "fp32", "fp8_e4m3fn", "fp8_e5m2", "int32", "int8", "unknown"] if x in pivot.columns]
    pivot = pivot[dtype_order]

    vals = pivot.values[np.isfinite(pivot.values)]
    vmax = max(1.0, np.percentile(vals, 95)) if len(vals) else 1.0

    fig_h = max(2.8, 0.20 * len(pivot) + 0.5)
    fig_w = max(4.2, 0.45 * len(pivot.columns) + 2.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap.set_bad("#F2F2F2")

    sns.heatmap(
        pivot,
        mask=pivot.isna(),
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "conflict score (%)"},
    )

    ax.set_xlabel("dtype")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_k} bank-conflict operator/backend rows")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", labelsize=6.5)

    savefig(fig, out_dir, "fig_ncu_top_conflict_heatmap")


def plot_backend_comparison(df, out_dir):
    d = df[np.isfinite(df["conflict_pct"]) & df["backend"].isin(["triton", "cutile"])].copy()
    pivot = d.pivot_table(
        index=["operator", "dtype"],
        columns="backend",
        values="conflict_pct",
        aggfunc="median",
    ).reset_index()

    if not {"triton", "cutile"}.issubset(set(pivot.columns)):
        return

    pivot = pivot[np.isfinite(pivot["triton"]) & np.isfinite(pivot["cutile"])]
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    sns.scatterplot(
        data=pivot,
        x="triton",
        y="cutile",
        hue="dtype",
        s=28,
        alpha=0.85,
        linewidth=0.3,
        edgecolor="white",
        ax=ax,
    )

    eps = 1e-3
    hi = max(pivot["triton"].max(), pivot["cutile"].max()) * 1.25
    lo = max(eps, min(pivot["triton"].min(), pivot["cutile"].min()) * 0.8)
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Triton conflict score (%)")
    ax.set_ylabel("cuTile conflict score (%)")
    ax.set_title("Backend comparison")
    ax.legend(frameon=False, fontsize=5.5, title="dtype", loc="best")
    ax.grid(True, which="both", alpha=0.18)

    savefig(fig, out_dir, "fig_ncu_backend_comparison")


def plot_branch_clean(df, out_dir):
    d = df[np.isfinite(df["conflict_pct"]) & np.isfinite(df["branch_efficiency_pct"])].copy()
    d = d[d["branch_metric_valid"].astype(bool)]

    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    sns.scatterplot(
        data=d,
        x="branch_efficiency_pct",
        y="conflict_pct_plot",
        hue="backend",
        style="diag_group",
        palette=BACKEND_COLORS,
        s=30,
        alpha=0.85,
        linewidth=0.25,
        edgecolor="white",
        ax=ax,
    )

    ax.axvline(99, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(95, color="gray", linestyle=":", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Branch efficiency (%)")
    ax.set_ylabel("Conflict score (%)")
    ax.set_title("Branch-divergence confounder")
    ax.legend(frameon=False, fontsize=5.2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, which="both", alpha=0.18)
    savefig(fig, out_dir, "fig_ncu_branch_confounder")


def plot_ipc_clean(df, out_dir):
    d = df[np.isfinite(df["conflict_pct"]) & np.isfinite(df["ipc_gap_ratio"])].copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    sns.scatterplot(
        data=d,
        x="ipc_gap_ratio",
        y="conflict_pct_plot",
        hue="backend",
        style="diag_group",
        palette=BACKEND_COLORS,
        s=30,
        alpha=0.85,
        linewidth=0.25,
        edgecolor="white",
        ax=ax,
    )

    ax.axvline(0.10, color="black", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Issued/executed IPC gap")
    ax.set_ylabel("Conflict score (%)")
    ax.set_title("IPC gap is not a conflict proxy")

    xmax = np.nanpercentile(d["ipc_gap_ratio"], 99) * 1.2
    ax.set_xlim(-0.001, min(max(xmax, 0.01), 0.12))
    ax.legend(frameon=False, fontsize=5.2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, which="both", alpha=0.18)
    savefig(fig, out_dir, "fig_ncu_ipc_gap")


def write_tables(df, out_dir, top_k=15):
    d = df[np.isfinite(df["conflict_pct"])].sort_values("conflict_pct", ascending=False).head(top_k)
    cols = [
        "operator", "backend", "dtype", "diagnosis",
        "conflict_pct", "branch_efficiency_pct", "ipc_gap_ratio", "kernel_name"
    ]
    cols = [c for c in cols if c in d.columns]
    t = d[cols].copy()
    t["conflict_pct"] = t["conflict_pct"].map(lambda x: f"{x:.2f}")
    if "branch_efficiency_pct" in t:
        t["branch_efficiency_pct"] = t["branch_efficiency_pct"].map(lambda x: f"{x:.2f}")
    if "ipc_gap_ratio" in t:
        t["ipc_gap_ratio"] = t["ipc_gap_ratio"].map(lambda x: f"{x:.4f}")

    latex = t.to_latex(
        index=False,
        escape=True,
        caption="Top NCU profiles ranked by normalized shared-memory bank-conflict severity.",
        label="tab:ncu_top_bank_conflicts",
    )
    (out_dir / "table_top_conflict_cases.tex").write_text(latex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--include-aux", action="store_true")
    args = parser.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    df = prepare(df, include_aux=args.include_aux)
    df.to_csv(args.out_dir / "ncu_bank_conflict_summary_for_paper.csv", index=False)

    print("Rows used for figures:", len(df))
    if "profile_valid_for_tilebench_kernel" in df:
        print("Valid profiles:", df["profile_valid_for_tilebench_kernel"].sum())
    if "selected_is_auxiliary" in df:
        print("Auxiliary selected:", df["selected_is_auxiliary"].sum())
    print(df["diag_group"].value_counts())

    plot_diagnosis_stacked(df, args.out_dir)
    plot_top_bar(df, args.out_dir, top_k=20)
    plot_top_heatmap(df, args.out_dir, top_k=args.top_k)
    plot_backend_comparison(df, args.out_dir)
    plot_branch_clean(df, args.out_dir)
    plot_ipc_clean(df, args.out_dir)
    write_tables(df, args.out_dir)

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()