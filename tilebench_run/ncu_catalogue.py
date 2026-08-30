"""
Catalogue all benchmark operators for the NCU sweep:
  - read each operator's config.yaml
  - compute sweep-max case (largest product of varied case_grid dims)
  - list dtypes
  - look up the autotune-winner cfg for that case in the autotune log

Writes tilebench_run/ncu_catalogue.json for the driver to consume.
"""
import json
import sys
from pathlib import Path

import yaml

from ncu_common import repo_root

ROOT = repo_root()
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


# Some ops write dtype="float32" / "float16" / "bfloat16" in the autotune
# log even though config.yaml uses the short form. Normalise.
DTYPE_ALIASES = {
    "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
}


def _norm_dtype(s):
    return DTYPE_ALIASES.get(s, s)


def read_log(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, list) else None


def pick_sweep_max_entry(op_name: str, dt: str, entries: list[dict]):
    entries = [
        e for e in entries
        if _norm_dtype(e.get("dtype")) == _norm_dtype(dt) and e.get("params")
    ]
    if not entries:
        return None

    # Sweep-max = the entry whose EVERY int-valued param equals that param's
    # maximum across entries. For cross-product case grids (all current ops)
    # this entry always exists and is unique. Float params (dropout's p, eps,
    # etc.) are per-op constants and are ignored. The engine's scalar
    # problem_size field is NOT used: it records a single dim, which ties
    # across cases in multi-dim sweeps.
    int_keys = sorted({
        k for e in entries
        for k, v in e["params"].items()
        if isinstance(v, int) and not isinstance(v, bool)
    })
    dim_max = {
        k: max(e["params"][k] for e in entries if k in e["params"])
        for k in int_keys
    }
    best = next(
        (e for e in entries
         if all(e["params"].get(k) == dim_max[k] for k in int_keys)),
        None,
    )
    if best is None:
        # Non-cross-product sweep: the all-dims-max combination does not
        # exist in the log. Fall back to max int-product, loudly.
        best = max(entries, key=lambda e: case_size(
            {k: v for k, v in e["params"].items()
             if isinstance(v, int) and not isinstance(v, bool)}))
        print(f"  WARNING {op_name}/{dt}: no all-dims-max case in "
              f"autotune log (per-dim maxes {dim_max}); falling back "
              f"to max int-product params={best['params']}")
    return best


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

    autotune_data = read_log(LOG_DIR / f"{op_name}_autotune.json")
    tilelang_autotune_data = read_log(LOG_DIR / f"{op_name}_tilelang_autotune.json")

    autotune_by_dtype = {}
    for dt in dtypes:
        record = {}
        best = pick_sweep_max_entry(op_name, dt, autotune_data or [])
        if best is not None:
            record.update({
                "params": best.get("params", {}),
                "triton": best.get("triton_autotune_cfg"),
                "cutile": best.get("cutile_autotune_cfg"),
            })

        tl_best = pick_sweep_max_entry(op_name, dt, tilelang_autotune_data or [])
        if tl_best is not None:
            record.setdefault("params", tl_best.get("params", {}))
            record["tilelang"] = tl_best.get("tilelang_autotune_cfg")

        if record:
            autotune_by_dtype[dt] = record

    return {
        "op": op_name,
        "dtypes": dtypes,
        "default_params_per_dtype": per_dtype,
        "autotune_winner_per_dtype": autotune_by_dtype,
        "has_autotune_log": bool(autotune_data),
        "has_tilelang_autotune_log": bool(tilelang_autotune_data),
    }


def main():
    skip = {"_template", "__pycache__"}
    all_ops = sorted([
        p.name for p in OPS_DIR.iterdir()
        if p.is_dir() and p.name not in skip and (p / "config.yaml").exists()
    ])
    out = ROOT / "tilebench_run" / "ncu_catalogue.json"

    requested = sys.argv[1:]
    if requested:
        unknown = [op for op in requested if op not in all_ops]
        if unknown:
            raise SystemExit(f"unknown ops: {unknown}")
        existing = json.loads(out.read_text()) if out.exists() else []
        by_op = {c["op"]: c for c in existing}
        for op in requested:
            by_op[op] = collect_op(op)
        catalogue = [by_op[op] for op in sorted(by_op)]
        out.write_text(json.dumps(catalogue, indent=2, default=str))
        print(f"wrote {out}  (updated {len(requested)} of {len(catalogue)} ops: {requested})")
        catalogue = [by_op[op] for op in requested]
    else:
        catalogue = [collect_op(op) for op in all_ops]
        out.write_text(json.dumps(catalogue, indent=2, default=str))
        print(f"wrote {out}  ({len(catalogue)} ops)")
    no_log = [c["op"] for c in catalogue if not c.get("has_autotune_log")]
    if no_log:
        print(f"ops without autotune log: {no_log}")
    no_tl_log = [c["op"] for c in catalogue if not c.get("has_tilelang_autotune_log")]
    if no_tl_log:
        print(f"ops without tilelang autotune log: {no_tl_log}")
    for c in catalogue:
        op = c["op"]
        dts = c["dtypes"]
        for dt in dts:
            winner = c["autotune_winner_per_dtype"].get(dt, {})
            has_tune = bool(winner.get("triton") or winner.get("cutile"))
            has_tl_tune = bool(winner.get("tilelang"))
            ps = c["default_params_per_dtype"][dt]
            print(f"  {op:30s}  dtype={dt:8s}  "
                  f"autotune={'yes' if has_tune else 'NO'}  "
                  f"tilelang_autotune={'yes' if has_tl_tune else 'NO'}  "
                  f"params={ps}")


if __name__ == "__main__":
    main()
