#!/usr/bin/env python3
"""
Batch analyze Nsight Compute raw CSV files for shared-memory bank conflicts.

Supports two NCU CSV formats:

1. Long format:
   Metric Name, Metric Unit, Metric Value

2. Wide format:
   ID, Process ID, Kernel Name, ..., <metric columns...>
   second row is often a unit row
   subsequent rows are kernel launches

Expected input:
  ncu --import report.ncu-rep --page raw --csv > report.raw.csv

Typical file name:
  1d_conv__cutile_fp16.raw.csv
  streamk_matmul__triton_fp32.raw.csv

Outputs:
  ncu_bank_conflict_summary.csv
  fig_bank_conflict_heatmap.pdf
  fig_conflict_vs_branch_efficiency.pdf
  fig_conflict_vs_ipc_gap.pdf
  fig_diagnosis_counts.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

def setup_style():
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        },
    )


# ---------------------------------------------------------------------
# Metric aliases
# ---------------------------------------------------------------------

METRICS = {
    # Direct shared-memory bank-conflict counters.
    "bank_conflicts_total": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "l1tex__data_bank_conflicts_pipe_lsu.sum",
    ],
    "bank_conflicts_ld": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    ],
    "bank_conflicts_st": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    ],
    "bank_conflicts_ldgsts": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum",
    ],
    "bank_conflicts_atom": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum",
    ],

    # Shared-memory wavefront counters, LSU path.
    "shared_wavefronts_total": [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    ],
    "shared_wavefronts_ld": [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    ],
    "shared_wavefronts_st": [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    ],
    "shared_wavefronts_ldgsts": [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    ],
    "shared_wavefronts_atom": [
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
    ],

    # Tensor-Core / TC path shared wavefronts.
    # These are NOT bank-conflict counters, but useful to tell whether shared-like
    # tensor traffic exists even when LSU bank conflict counters are zero.
    "tc_shared_wavefronts_total": [
        "l1tex__data_pipe_tc_wavefronts_mem_shared.sum",
    ],
    "tc_shared_wavefronts_utccp": [
        "l1tex__data_pipe_tc_wavefronts_mem_shared_op_utccp.sum",
    ],

    # Source-derived metrics, if present.
    "source_shared_conflict_nway": [
        "derived__memory_l1_conflicts_shared_nway",
    ],
    "source_shared_excessive_wavefronts": [
        "derived__memory_l1_wavefronts_shared_excessive",
    ],
    "source_shared_wavefronts": [
        "memory_l1_wavefronts_shared",
    ],
    "source_shared_wavefronts_ideal": [
        "memory_l1_wavefronts_shared_ideal",
    ],

    # Branch / divergence.
    "branch_efficiency_pct": [
        "smsp__sass_average_branch_targets_threads_uniform.pct",
    ],
    "divergent_branch_targets": [
        "smsp__branch_targets_threads_divergent",
        "smsp__branch_targets_threads_divergent.sum",
        "smsp__sass_branch_targets_threads_divergent.sum",
    ],
    "branch_instructions": [
        "smsp__inst_executed_op_branch.sum",
        "smsp__inst_executed_op_branch.avg",
    ],
    "branch_instruction_pct": [
        "derived__smsp__inst_executed_op_branch_pct",
    ],
    # IPC.
    "executed_ipc_active": [
        "sm__inst_executed.avg.per_cycle_active",
        "TPC.TriageCompute.sm__inst_executed_realtime.avg.per_cycle_active",
        "smsp__inst_executed.avg.per_cycle_active",
    ],
    "issued_ipc_active": [
        "sm__inst_issued.avg.per_cycle_active",
        "smsp__inst_issued.avg.per_cycle_active",
    ],

    # Scheduling and stall signals.
    "eligible_warps_per_cycle": [
        "smsp__warps_eligible.avg.per_cycle_active",
    ],
    "stall_mio_throttle": [
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
        "smsp__pcsamp_warps_issue_stalled_mio_throttle",
        "warpsampling:smsp__pcsamp_warps_issue_stalled_mio_throttle",
    ],
    "stall_long_scoreboard": [
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        "warpsampling:smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    ],
    "stall_short_scoreboard": [
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
        "warpsampling:smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    ],
    "stall_barrier": [
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__pcsamp_warps_issue_stalled_barrier",
        "warpsampling:smsp__pcsamp_warps_issue_stalled_barrier",
    ],

    # Launch / timing.
    "gpu_time_us": [
        "gpu__time_duration.sum",
        "gpu__time_duration.avg",
    ],
    "inst_executed": [
        "inst_executed",
        "smsp__inst_executed.sum",
        "sm__inst_executed.sum.per_cycle_active",
    ],
    "launch_registers_per_thread": [
        "launch__registers_per_thread",
        "launch__registers_per_thread_allocated",
    ],
    "launch_shared_mem_per_block": [
        "launch__shared_mem_per_block",
        "launch__shared_mem_per_block_allocated",
    ],
    "launch_block_size": [
        "launch__block_size",
    ],
    "launch_grid_size": [
        "launch__grid_size",
    ],

    # TMA / tensor-path indicators.
    "tma_inst_ld": [
        "smsp__inst_executed_op_tma_ld.sum",
        "smsp__sass_inst_executed_op_tma_ld.sum",
    ],
    "tma_inst_st": [
        "smsp__inst_executed_op_tma_st.sum",
        "smsp__sass_inst_executed_op_tma_st.sum",
    ],
    "tma_pipe_util": [
        "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_executed_pipe_tma.avg.pct_of_peak_sustained_elapsed",
    ],
    "tc_pipe_util": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
    ],
}


# ---------------------------------------------------------------------
# CSV parsing utilities
# ---------------------------------------------------------------------

DTYPE_ORDER = [
    "fp8_e4m3fn",
    "fp8_e5m2",
    "fp16",
    "bf16",
    "fp32",
    "int8",
    "int32",
]

DTYPE_ALIASES = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "bfloat16": "bf16",
    "half": "fp16",
}


def normalize_col(c: str) -> str:
    return str(c).strip()


def to_float(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "n/a", "na", "none", "--"}:
        return np.nan
    s = s.replace(",", "")
    s = s.replace("%", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_metadata_from_filename(path: Path) -> dict:
    """
    Examples:
      1d_conv__cutile_fp16.raw.csv
      matmul_fp32_fp16_fp8__triton_fp8_e4m3fn.raw.csv
    """
    name = path.name
    stem = name
    for suffix in [".raw.csv", ".csv"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    if "__" in stem:
        operator, rest = stem.split("__", 1)
    else:
        operator, rest = stem, stem

    backend = "unknown"
    if rest.startswith("triton"):
        backend = "triton"
    elif rest.startswith("cutile"):
        backend = "cutile"
    elif "triton" in rest:
        backend = "triton"
    elif "cutile" in rest:
        backend = "cutile"

    dtype = "unknown"
    # Prefer matching dtype from rest, not operator name.
    for d in DTYPE_ORDER:
        if d in rest:
            dtype = d
            break
    if dtype == "unknown":
        for alias, canonical in DTYPE_ALIASES.items():
            if alias in rest:
                dtype = canonical
                break

    mode = "unknown"
    if "autotune" in rest:
        mode = "autotune"
    elif "default" in rest:
        mode = "default"

    return {
        "file": path.name,
        "operator": operator,
        "backend": backend,
        "dtype": dtype,
        "mode": mode,
    }


def is_long_format(df: pd.DataFrame) -> bool:
    cols_lower = {c.lower().replace(" ", "_") for c in df.columns}
    has_metric = any(c in cols_lower for c in ["metric_name", "metric", "name"])
    has_value = any(c in cols_lower for c in ["metric_value", "value", "avg", "aggregate_value"])
    return has_metric and has_value


def read_long_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path, comment="#", engine="python")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    metric_col = None
    for c in ["metric_name", "metric", "name"]:
        if c in df.columns:
            metric_col = c
            break

    value_col = None
    for c in ["metric_value", "value", "avg", "aggregate_value"]:
        if c in df.columns:
            value_col = c
            break

    if metric_col is None or value_col is None:
        raise ValueError(f"Cannot identify long-format metric/value columns in {path}")

    kernel_col = None
    for c in ["kernel_name", "kernel", "function_name", "name_demangled"]:
        if c in df.columns:
            kernel_col = c
            break

    metric_map = {}
    for _, r in df.iterrows():
        m = str(r[metric_col]).strip()
        v = to_float(r[value_col])
        if m:
            metric_map[m] = v

    kernel_name = ""
    if kernel_col is not None and len(df) > 0:
        kernel_name = str(df[kernel_col].dropna().iloc[0])

    return [{"kernel_name": kernel_name, "metrics": metric_map, "raw_row_index": 0}]


def read_wide_csv(path: Path) -> list[dict]:
    """
    Wide format:
      header row: metric names
      row 0: units, ID is usually blank
      row 1+: kernel launches
    """
    df = pd.read_csv(path, dtype=str, engine="python")
    df.columns = [normalize_col(c) for c in df.columns]

    # Detect ID column.
    id_col = None
    for cand in ["ID", "Id", "id"]:
        if cand in df.columns:
            id_col = cand
            break

    if id_col is not None:
        numeric_id = pd.to_numeric(df[id_col], errors="coerce")
        df_data = df[numeric_id.notna()].copy()
    else:
        # Fallback: remove rows where Kernel Name is empty.
        if "Kernel Name" in df.columns:
            df_data = df[df["Kernel Name"].notna() & (df["Kernel Name"].astype(str).str.len() > 0)].copy()
        else:
            df_data = df.copy()

    rows = []
    for idx, r in df_data.iterrows():
        metric_map = {}
        for c in df.columns:
            metric_map[c] = to_float(r[c])

        kernel_name = ""
        for kc in ["Kernel Name", "kernel_name", "launch__kernel_name"]:
            if kc in df.columns:
                val = str(r.get(kc, "")).strip()
                if val and val.lower() != "nan":
                    kernel_name = val
                    break

        rows.append(
            {
                "kernel_name": kernel_name,
                "metrics": metric_map,
                "raw_row_index": int(idx),
            }
        )

    return rows


def read_ncu_csv_any(path: Path) -> list[dict]:
    # First read a small header to identify format.
    df_head = pd.read_csv(path, nrows=3, dtype=str, engine="python")
    if is_long_format(df_head):
        return read_long_csv(path)
    return read_wide_csv(path)


# ---------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------

def get_metric(metric_map: dict, aliases: list[str]) -> float:
    # Exact match first.
    for a in aliases:
        if a in metric_map:
            return to_float(metric_map[a])

    # Case-insensitive exact match.
    lower_map = {str(k).lower(): v for k, v in metric_map.items()}
    for a in aliases:
        al = a.lower()
        if al in lower_map:
            return to_float(lower_map[al])

    # Some Nsight pages prefix section names, e.g.
    # TPC.TriageCompute.sm__inst_executed_realtime.avg.per_cycle_active
    # Exact aliases above should cover important ones, but we also allow suffix match.
    for a in aliases:
        al = a.lower()
        candidates = []
        for k, v in lower_map.items():
            if k.endswith(al):
                candidates.append(v)
        if candidates:
            vals = [to_float(v) for v in candidates]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                # Most metrics appear once; if multiple section-prefixed copies exist,
                # use max to avoid blank/zero prefixed duplicates suppressing the raw value.
                return float(np.nanmax(vals))

    return np.nan


def summarize_kernel(path: Path, kernel_entry: dict) -> dict:
    meta = parse_metadata_from_filename(path)
    metrics = kernel_entry["metrics"]

    row = dict(meta)
    row["kernel_name"] = kernel_entry.get("kernel_name", "")
    row["raw_row_index"] = kernel_entry.get("raw_row_index", -1)

    for out_name, aliases in METRICS.items():
        row[out_name] = get_metric(metrics, aliases)

    # Bank conflicts.
    total = row["bank_conflicts_total"]
    parts = [
        row["bank_conflicts_ld"],
        row["bank_conflicts_st"],
        row["bank_conflicts_ldgsts"],
        row["bank_conflicts_atom"],
    ]
    parts_sum = np.nansum(parts)

    if np.isfinite(total):
        bank_conflicts = total
    elif parts_sum > 0:
        bank_conflicts = parts_sum
    else:
        bank_conflicts = np.nan

    # LSU shared wavefronts.
    wf_total = row["shared_wavefronts_total"]
    wf_parts = [
        row["shared_wavefronts_ld"],
        row["shared_wavefronts_st"],
        row["shared_wavefronts_ldgsts"],
        row["shared_wavefronts_atom"],
    ]
    wf_parts_sum = np.nansum(wf_parts)

    if np.isfinite(wf_total):
        shared_wavefronts = wf_total
    elif wf_parts_sum > 0:
        shared_wavefronts = wf_parts_sum
    else:
        shared_wavefronts = np.nan

    row["bank_conflicts"] = bank_conflicts
    row["shared_wavefronts"] = shared_wavefronts

    if np.isfinite(bank_conflicts) and np.isfinite(shared_wavefronts) and shared_wavefronts > 0:
        row["conflict_per_wavefront"] = bank_conflicts / shared_wavefronts
    else:
        row["conflict_per_wavefront"] = np.nan

    # Source-level overhead if available.
    sw = row["source_shared_wavefronts"]
    si = row["source_shared_wavefronts_ideal"]
    if np.isfinite(sw) and np.isfinite(si) and si > 0:
        row["source_shared_wavefront_overhead"] = max(sw - si, 0) / si
    else:
        row["source_shared_wavefront_overhead"] = np.nan

    # IPC gap.
    exec_ipc = row["executed_ipc_active"]
    issue_ipc = row["issued_ipc_active"]
    if np.isfinite(exec_ipc) and np.isfinite(issue_ipc) and max(abs(exec_ipc), abs(issue_ipc)) > 0:
        row["ipc_gap_ratio"] = abs(issue_ipc - exec_ipc) / max(abs(exec_ipc), abs(issue_ipc))
    else:
        row["ipc_gap_ratio"] = np.nan

    # TMA indicators.
    tma_ld = row["tma_inst_ld"]
    tma_st = row["tma_inst_st"]
    row["tma_inst_total"] = np.nansum([tma_ld, tma_st])
    row["uses_tma_counter"] = bool(np.isfinite(row["tma_inst_total"]) and row["tma_inst_total"] > 0)

    # Branch-efficiency validity: only treat branch efficiency as meaningful when
    # there is evidence of branch activity in the profile.
    branch = row.get("branch_efficiency_pct", np.nan)
    branch_inst = row.get("branch_instructions", np.nan)
    branch_pct = row.get("branch_instruction_pct", np.nan)
    div_targets = row.get("divergent_branch_targets", np.nan)

    branch_activity = False
    if np.isfinite(branch_inst) and branch_inst > 0:
        branch_activity = True
    if np.isfinite(branch_pct) and branch_pct > 0.1:
        branch_activity = True
    if np.isfinite(div_targets) and div_targets > 0:
        branch_activity = True

    branch_metric_valid = (
        np.isfinite(branch)
        and branch > 0
        and branch_activity
    )

    row["branch_activity"] = branch_activity
    row["branch_metric_valid"] = branch_metric_valid

    row["bank_conflict_severity"] = severity_from_ratio(row["conflict_per_wavefront"])
    row["diagnosis"] = diagnose(row)

    return row


# ---------------------------------------------------------------------
# Kernel selection
# ---------------------------------------------------------------------

AUXILIARY_KERNEL_PATTERNS = [
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

def is_auxiliary_kernel(kernel_name: str) -> bool:
    s = str(kernel_name)
    for pat in AUXILIARY_KERNEL_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def select_primary_kernel(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    valid = [r for r in rows if not is_auxiliary_kernel(r.get("kernel_name", ""))]
    aux = [r for r in rows if is_auxiliary_kernel(r.get("kernel_name", ""))]

    def key_fn(r):
        t = r.get("gpu_time_us", np.nan)
        inst = r.get("inst_executed", np.nan)
        if np.isfinite(t):
            return (2, t)
        if np.isfinite(inst):
            return (1, inst)
        return (0, 0)

    if valid:
        best = max(valid, key=key_fn)
        best = dict(best)
        best["kernel_policy_selected"] = "primary"
        best["selected_is_auxiliary"] = False
        best["profile_valid_for_tilebench_kernel"] = True
    else:
        # Keep the largest aux row for auditing, but mark invalid.
        best = max(aux, key=key_fn)
        best = dict(best)
        best["kernel_policy_selected"] = "auxiliary_only"
        best["selected_is_auxiliary"] = True
        best["profile_valid_for_tilebench_kernel"] = False

    best["num_kernels_in_report"] = len(rows)
    best["num_non_aux_kernels_in_report"] = len(valid)
    return [best]


# ---------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------

def severity_from_ratio(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x < 0.01:
        return "none"
    if x < 0.10:
        return "mild"
    if x < 0.30:
        return "moderate"
    return "severe"


def diagnose(row: dict) -> str:
    c = row.get("conflict_per_wavefront", np.nan)
    branch = row.get("branch_efficiency_pct", np.nan)
    shared_wf = row.get("shared_wavefronts", np.nan)
    bank_conflicts = row.get("bank_conflicts", np.nan)
    source_nway = row.get("source_shared_conflict_nway", np.nan)
    source_overhead = row.get("source_shared_wavefront_overhead", np.nan)
    tc_wf = row.get("tc_shared_wavefronts_total", np.nan)

    branch_valid = bool(row.get("branch_metric_valid", False))

    branch_confounded = branch_valid and branch < 95.0
    branch_mild = branch_valid and 95.0 <= branch < 99.0

    # No LSU shared wavefronts but TC/TMA path may exist.
    if (not np.isfinite(shared_wf) or shared_wf <= 0):
        if np.isfinite(tc_wf) and tc_wf > 0:
            return "NO_LSU_SHARED_CONFLICT_EVIDENCE_TC_SHARED_ACTIVE"
        return "NO_SHARED_MEMORY_EVIDENCE"

    if not np.isfinite(bank_conflicts):
        if np.isfinite(source_nway) and source_nway > 1:
            return "BANK_CONFLICT_LIKELY_SOURCE_ONLY"
        return "INSUFFICIENT_METRICS"

    if bank_conflicts <= 0 or not np.isfinite(c) or c < 0.01:
        if branch_confounded:
            return "DIVERGENCE_CONFOUNDED_NO_CONFLICT"
        if np.isfinite(source_nway) and source_nway > 1:
            return "BANK_CONFLICT_LIKELY_SOURCE_ONLY"
        if np.isfinite(source_overhead) and source_overhead > 0.10:
            return "SHARED_WAVEFRONT_OVERHEAD_NO_DIRECT_CONFLICT"
        return "NO_BANK_CONFLICT_EVIDENCE"

    if branch_confounded:
        return "DIVERGENCE_CONFOUNDED"

    if c >= 0.30 and not branch_mild:
        return "BANK_CONFLICT_CONFIRMED_SEVERE"
    if c >= 0.10:
        return "BANK_CONFLICT_CONFIRMED_MODERATE"
    if c >= 0.01:
        return "BANK_CONFLICT_CONFIRMED_MILD"

    return "INSUFFICIENT_METRICS"


# ---------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------

def summarize_one_file(path: Path, kernel_policy: str) -> list[dict]:
    entries = read_ncu_csv_any(path)
    rows = [summarize_kernel(path, e) for e in entries]

    if kernel_policy == "all":
        for r in rows:
            r["kernel_policy_selected"] = "all"
            r["num_kernels_in_report"] = len(rows)
            r["num_non_aux_kernels_in_report"] = sum(
                not is_auxiliary_kernel(x.get("kernel_name", "")) for x in rows
            )
        return rows

    if kernel_policy == "primary":
        return select_primary_kernel(rows)

    raise ValueError(f"Unknown kernel policy: {kernel_policy}")


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def savefig(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_dir: Path):
    d = df.copy()
    d = d[np.isfinite(d["conflict_per_wavefront"])]
    if d.empty:
        return

    d["row"] = d["operator"] + " / " + d["backend"]
    pivot = d.pivot_table(
        index="row",
        columns="dtype",
        values="conflict_per_wavefront",
        aggfunc="median",
    )
    if pivot.empty:
        return

    pivot["__max__"] = pivot.max(axis=1)
    pivot = pivot.sort_values("__max__", ascending=False).drop(columns="__max__")

    fig_h = max(3.0, 0.25 * len(pivot))
    fig, ax = plt.subplots(figsize=(5.5, fig_h))
    sns.heatmap(
        100.0 * pivot,
        ax=ax,
        cmap="magma_r",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "bank conflicts / shared wavefront (%)"},
    )
    ax.set_title("Shared-memory bank-conflict severity")
    ax.set_xlabel("dtype")
    ax.set_ylabel("operator / backend")
    savefig(fig, out_dir, "fig_bank_conflict_heatmap")


def plot_branch_scatter(df: pd.DataFrame, out_dir: Path):
    d = df.copy()
    d = d[np.isfinite(d["conflict_per_wavefront"])]
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    sns.scatterplot(
        data=d,
        x="branch_efficiency_pct",
        y=100.0 * d["conflict_per_wavefront"],
        hue="backend",
        style="diagnosis",
        s=50,
        ax=ax,
    )
    ax.axvline(99.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(95.0, color="gray", linestyle=":", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Branch efficiency (%)")
    ax.set_ylabel("Bank conflicts / shared wavefront (%)")
    ax.set_title("Branch divergence as a confounder")
    ax.legend(frameon=False, fontsize=6, loc="best")
    savefig(fig, out_dir, "fig_conflict_vs_branch_efficiency")


def plot_ipc_gap(df: pd.DataFrame, out_dir: Path):
    d = df.copy()
    d = d[np.isfinite(d["conflict_per_wavefront"]) & np.isfinite(d["ipc_gap_ratio"])]
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    sns.scatterplot(
        data=d,
        x="ipc_gap_ratio",
        y=100.0 * d["conflict_per_wavefront"],
        hue="backend",
        style="diagnosis",
        s=50,
        ax=ax,
    )
    ax.axvline(0.10, color="black", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("|issued IPC - executed IPC| / max")
    ax.set_ylabel("Bank conflicts / shared wavefront (%)")
    ax.set_title("IPC imbalance is not a bank-conflict proxy")
    ax.legend(frameon=False, fontsize=6, loc="best")
    savefig(fig, out_dir, "fig_conflict_vs_ipc_gap")


def plot_diagnosis_counts(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return

    order = df["diagnosis"].value_counts().index.tolist()
    fig_h = max(3.0, 0.28 * len(order))
    fig, ax = plt.subplots(figsize=(5.2, fig_h))
    sns.countplot(
        data=df,
        y="diagnosis",
        order=order,
        hue="backend",
        ax=ax,
    )
    ax.set_xlabel("# reports")
    ax.set_ylabel("")
    ax.set_title("Automated NCU diagnosis categories")
    ax.legend(frameon=False, title="")
    savefig(fig, out_dir, "fig_diagnosis_counts")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("ncu_bank_analysis"))
    parser.add_argument("--pattern", type=str, default="*.raw.csv")
    parser.add_argument(
        "--kernel-policy",
        choices=["primary", "all"],
        default="primary",
        help="primary: choose one implementation kernel per report; all: keep every kernel row.",
    )
    args = parser.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    for p in sorted(args.raw_csv_dir.glob(args.pattern)):
        try:
            out_rows = summarize_one_file(p, args.kernel_policy)
            rows.extend(out_rows)
            print(f"[OK] {p.name}: parsed {len(out_rows)} row(s)")
        except Exception as e:
            print(f"[WARN] failed to parse {p}: {e}")
            failures.append({"file": str(p), "error": str(e)})

    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / "parse_failures.csv", index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No reports parsed.")

    summary_path = args.out_dir / "ncu_bank_conflict_summary.csv"
    df.to_csv(summary_path, index=False)

    plot_heatmap(df, args.out_dir)
    plot_branch_scatter(df, args.out_dir)
    plot_ipc_gap(df, args.out_dir)
    plot_diagnosis_counts(df, args.out_dir)

    print()
    print(f"Wrote summary and figures to {args.out_dir}")
    print(f"Summary: {summary_path}")
    print()
    print("Diagnosis counts:")
    print(df["diagnosis"].value_counts())


if __name__ == "__main__":
    main()