"""Load benchmark results from the 8-column summary CSVs on main.

One tidy row per (op, backend, dtype, case, mode) with speedup and roofline
utilization R. Case params are reconstructed by merging each op's
config.yaml case_defaults under the params parsed from the CSV `params`
column, and problem_size comes from data.tensors.infer_problem_size — the
same path the engine uses — so bytes_expr/flops_expr evaluate identically.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.tensors import infer_problem_size            # noqa: E402
from tools.llm_codegen.roofline import load_peak, roofline_pct  # noqa: E402
from scripts.analysis.paper_style import CATEGORY, DIFFICULTY   # noqa: E402

CSV_DIR = REPO / "results" / "csv"


def _parse_params(s: str) -> dict:
    out: dict = {}
    for part in s.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def load_main_table() -> list[dict]:
    peak = load_peak("B200")
    rows: list[dict] = []
    for op, cat in CATEGORY.items():
        cfg_path = REPO / "benchmarks" / "operators" / op / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        case_defaults = cfg.get("case_defaults") or {}
        metrics = cfg.get("metrics") or {}
        flops_expr = metrics.get("flops_expr")
        bytes_expr = metrics.get("bytes_expr")

        for mode in ("default", "autotune"):
            path = CSV_DIR / f"{op}_{mode}.csv"
            if not path.exists():
                continue
            with path.open() as f:
                for rec in csv.DictReader(f):
                    dtype = rec.get("dtype")
                    try:
                        torch_ms = float(rec["torch_ms"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not dtype or torch_ms <= 0:
                        continue
                    params = dict(case_defaults)
                    params.update(_parse_params(rec.get("params", "")))
                    params["problem_size"] = infer_problem_size(op, params)
                    for backend in ("triton", "cutile"):
                        try:
                            ms = float(rec[f"{backend}_ms"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if ms <= 0:
                            continue
                        rp = roofline_pct(
                            flops_expr=flops_expr, bytes_expr=bytes_expr,
                            params=params, dtype_str=dtype,
                            latency_s=ms * 1e-3, peak=peak,
                        )
                        R = rp.get("roofline_pct")
                        rows.append({
                            "op": op, "category": cat,
                            "difficulty": DIFFICULTY.get(op, 0),
                            "backend": backend, "dtype": dtype, "mode": mode,
                            "params": rec.get("params", ""),
                            "torch_ms": torch_ms, "kernel_ms": ms,
                            "speedup": torch_ms / ms,
                            "R": (min(max(R, 0.0), 1.0) if R is not None else None),
                            "bound_by": rp.get("bound_by", "?"),
                        })
    return rows


def geomean(vals) -> float:
    """Equally-weighted geometric mean of positive values."""
    a = np.asarray(list(vals), dtype=float)
    return float(np.exp(np.mean(np.log(a))))


def per_op_mean_speedup(main_rows: list[dict], mode: str = "default") -> dict:
    """Per-op equally-weighted geometric mean of speedup across dtype x case."""
    bucket: dict[str, dict[str, list]] = defaultdict(lambda: {"triton": [], "cutile": []})
    for r in main_rows:
        if r["mode"] != mode or r["speedup"] <= 0:
            continue
        bucket[r["op"]][r["backend"]].append(r["speedup"])
    out: dict = {}
    for op, d in bucket.items():
        if not d["triton"] or not d["cutile"]:
            continue
        out[op] = {"triton": geomean(d["triton"]),
                   "cutile": geomean(d["cutile"]),
                   "category": CATEGORY.get(op, "?"),
                   "difficulty": DIFFICULTY.get(op, 0)}
    return out


def per_op_mean_R(main_rows: list[dict]) -> dict:
    """Per-(op, backend) equally-weighted geometric mean of R per mode."""
    bucket: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {"default": [], "autotune": []})
    for r in main_rows:
        if r["R"] is None or r["R"] <= 0:
            continue
        bucket[(r["op"], r["backend"])][r["mode"]].append(r["R"])
    out = {}
    for (op, b), d in bucket.items():
        if not d["default"] or not d["autotune"]:
            continue
        out[(op, b)] = {
            "default_R": geomean(d["default"]),
            "autotune_R": geomean(d["autotune"]),
            "category": CATEGORY.get(op, "?"),
        }
    return out
