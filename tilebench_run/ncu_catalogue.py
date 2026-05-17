"""
Catalogue all benchmark operators for the NCU sweep:
  - read each operator's config.yaml
  - compute sweep-max case (largest product of varied case_grid dims)
  - list dtypes
  - look up the autotune-winner cfg for that case in the autotune log

Writes tilebench_run/ncu_catalogue.json for the driver to consume.
"""
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
sys.path.insert(0, str(ROOT))

OPS_DIR = ROOT / "benchmarks" / "operators"
LOG_DIR = ROOT / "results" / "logs" / "autotune_logs"


def expand_case_grid(case_grid: dict) -> list[dict]:
    """Expand a case_grid (dict of name -> list-or-expr) into a list of param dicts."""
    if not case_grid:
        return [{}]
    keys = list(case_grid.keys())
    values_per_key = []
    for k in keys:
        v = case_grid[k]
        if isinstance(v, dict) and "expr" in v:
            v = eval(v["expr"], {"range": range, "list": list, "int": int, "float": float, "min": min, "max": max, "abs": abs}, {})
        if not isinstance(v, list):
            v = [v]
        values_per_key.append(v)

    cases = [{}]
    for k, vs in zip(keys, values_per_key):
        new = []
        for c in cases:
            for v in vs:
                cc = dict(c)
                cc[k] = v
                new.append(cc)
        cases = new
    return cases


def case_size(case: dict) -> int:
    """Rough scalar size for ordering — product of all integer-valued dims."""
    size = 1
    for k, v in case.items():
        if isinstance(v, (int, float)):
            size *= int(v)
    return size


def collect_op(op_name: str) -> dict:
    """Build a catalogue entry for one operator."""
    op_dir = OPS_DIR / op_name
    cfg_path = op_dir / "config.yaml"
    if not cfg_path.exists():
        return {"op": op_name, "error": "no config.yaml"}

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    case_defaults = cfg.get("case_defaults", {}) or {}
    case_grid = cfg.get("case_grid", {}) or {}
    grid_cases = expand_case_grid(case_grid)

    dtypes = []
    if "dtype" in case_grid:
        v = case_grid["dtype"]
        if isinstance(v, dict) and "expr" in v:
            v = eval(v["expr"], {"range": range, "list": list, "int": int, "float": float, "min": min, "max": max, "abs": abs}, {})
        if not isinstance(v, list):
            v = [v]
        dtypes = v
    elif "dtype" in case_defaults:
        dtypes = [case_defaults["dtype"]]
    else:
        dtypes = ["fp32"]

    per_dtype = {}
    for dt in dtypes:
        candidates = [c for c in grid_cases if c.get("dtype") == dt or "dtype" not in c]
        if not candidates:
            candidates = grid_cases
        cleaned = []
        for c in candidates:
            cc = {k: v for k, v in c.items() if k != "dtype"}
            cleaned.append(cc)
        sized = [(case_size(c), i, c) for i, c in enumerate(cleaned)]
        sized.sort(key=lambda t: t[0], reverse=True)
        max_case = sized[0][2]
        params = dict(case_defaults)
        params.update(max_case)
        per_dtype[dt] = params

    log_path = LOG_DIR / f"{op_name}_autotune.json"
    autotune_data = None
    if log_path.exists():
        try:
            with open(log_path) as f:
                autotune_data = json.load(f)
        except Exception:
            autotune_data = None

    autotune_by_dtype = {}
    if isinstance(autotune_data, list):
        for dt in dtypes:
            best = None
            best_size = -1
            for entry in autotune_data:
                if entry.get("dtype") != dt:
                    continue
                ps = entry.get("problem_size", 0)
                if ps > best_size:
                    best_size = ps
                    best = entry
            if best is not None:
                autotune_by_dtype[dt] = {
                    "params":  best.get("params", {}),
                    "triton":  best.get("triton_autotune_cfg"),
                    "cutile":  best.get("cutile_autotune_cfg"),
                }

    return {
        "op": op_name,
        "dtypes": dtypes,
        "default_params_per_dtype": per_dtype,
        "autotune_winner_per_dtype": autotune_by_dtype,
        "has_autotune_log": bool(autotune_data),
    }


def main():
    skip = {"_template", "__pycache__"}
    ops = sorted([
        p.name for p in OPS_DIR.iterdir()
        if p.is_dir() and p.name not in skip and (p / "config.yaml").exists()
    ])
    catalogue = [collect_op(op) for op in ops]
    out = ROOT / "tilebench_run" / "ncu_catalogue.json"
    out.write_text(json.dumps(catalogue, indent=2, default=str))
    print(f"wrote {out}  ({len(catalogue)} ops)")
    no_log = [c["op"] for c in catalogue if not c.get("has_autotune_log")]
    if no_log:
        print(f"ops without autotune log: {no_log}")
    for c in catalogue:
        op = c["op"]
        dts = c["dtypes"]
        for dt in dts:
            has_tune = dt in c["autotune_winner_per_dtype"]
            ps = c["default_params_per_dtype"][dt]
            print(f"  {op:30s}  dtype={dt:8s}  "
                  f"autotune={'yes' if has_tune else 'NO'}  "
                  f"params={ps}")


if __name__ == "__main__":
    main()
