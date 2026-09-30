"""NCU catalogue: for every operator, the sweep-max case of its config.yaml and the
autotune winners recorded for it on one GPU.

Library only. The command line lives in scripts/profiling/ncu_catalogue.py.

  - sweep_max_cases(op)   dtypes and sweep-max case per dtype, from config.yaml
                          alone (largest product of the varied case_grid dims)
  - collect_op(...)       one catalogue entry: the cases above plus the winners
                          read from ONE autotune log, named explicitly by GPU,
                          mode and backend selection
                          (tilebench.paths.autotune_log_path): never picked by
                          glob or mtime
  - write_catalogue(...)  build or refresh outputs/profiling/<gpu>/ncu_catalogue.json

The catalogue is generated, hardware-specific data (the winners come from that
GPU's autotune runs), so it lives with the other generated outputs, not in the
source package. Each GPU has its own; building one never touches another.
"""
import json
from pathlib import Path

import yaml
from tilebench.paths import OPERATOR_ROOT, autotune_log_path, list_operators, ncu_catalogue_path

OPS_DIR = OPERATOR_ROOT


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


def sweep_max_cases(op_name: str) -> tuple[list, dict]:
    """(dtypes, {dtype: params of the sweep-max case}) of one operator, derived
    from its config.yaml alone: no measurement and no hardware is involved."""
    with open(OPS_DIR / op_name / "config.yaml") as f:
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
    return dtypes, per_dtype


def collect_op(op_name: str, gpu: str, backends: list[str]) -> dict:
    """Build a catalogue entry for one operator."""
    if not (OPS_DIR / op_name / "config.yaml").exists():
        return {"op": op_name, "error": "no config.yaml"}
    dtypes, per_dtype = sweep_max_cases(op_name)

    log_path = autotune_log_path(gpu, op_name, "autotune", backends)
    autotune_data = None
    if log_path.exists():
        try:
            with open(log_path) as f:
                autotune_data = json.load(f)
        except Exception:
            autotune_data = None

    # Some ops write dtype="float32" / "float16" / "bfloat16" in the autotune
    # log even though the config.yaml dtype field uses the short form. Normalise.
    DTYPE_ALIASES = {
        "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
    }
    def _norm(s):
        return DTYPE_ALIASES.get(s, s)

    autotune_by_dtype = {}
    if isinstance(autotune_data, list):
        for dt in dtypes:
            entries = [e for e in autotune_data
                       if _norm(e.get("dtype")) == _norm(dt) and e.get("params")]
            if not entries:
                continue
            # Sweep-max = the entry whose EVERY int-valued param equals that
            # param's maximum across entries. For cross-product case grids
            # (all current ops) this entry always exists and is unique.
            # Float params (dropout's p, eps, ...) are per-op constants and
            # are ignored. The engine's scalar problem_size field is NOT
            # used: it records a single dim, which ties across cases in
            # multi-dim sweeps (top_k's N x k, streamk's m x n, ...) and the
            # old first-tie-wins scan paired the params of one case with the
            # winner of another. params and winner now come from ONE entry.
            int_keys = sorted({k for e in entries
                               for k, v in e["params"].items()
                               if isinstance(v, int) and not isinstance(v, bool)})
            dim_max = {k: max(e["params"][k] for e in entries
                              if k in e["params"]) for k in int_keys}
            best = next((e for e in entries
                         if all(e["params"].get(k) == dim_max[k]
                                for k in int_keys)), None)
            if best is None:
                # Non-cross-product sweep: the all-dims-max combination does
                # not exist in the log. Fall back to max int-product, LOUDLY.
                best = max(entries, key=lambda e: case_size(
                    {k: v for k, v in e["params"].items()
                     if isinstance(v, int) and not isinstance(v, bool)}))
                print(f"  WARNING {op_name}/{dt}: no all-dims-max case in "
                      f"autotune log (per-dim maxes {dim_max}); falling back "
                      f"to max int-product params={best['params']}")
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


def write_catalogue(gpu: str, backends: list[str], ops: list[str] | None = None) -> tuple[Path, list[dict]]:
    """Build the catalogue of `gpu`, or refresh only `ops` inside the existing
    one, and write it. Returns (path, the entries that were (re)built)."""
    all_ops = list_operators()
    out = ncu_catalogue_path(gpu)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ops:
        unknown = [op for op in ops if op not in all_ops]
        if unknown:
            raise ValueError(f"unknown ops: {unknown}")
        existing = json.loads(out.read_text()) if out.exists() else []
        by_op = {c["op"]: c for c in existing}
        for op in ops:
            by_op[op] = collect_op(op, gpu, backends)
        out.write_text(json.dumps([by_op[op] for op in sorted(by_op)], indent=2, default=str))
        return out, [by_op[op] for op in ops]
    catalogue = [collect_op(op, gpu, backends) for op in all_ops]
    out.write_text(json.dumps(catalogue, indent=2, default=str))
    return out, catalogue
