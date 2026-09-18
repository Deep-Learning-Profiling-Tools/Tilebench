"""For each of the 4 problem ops, probe every candidate cfg in its
autotune search space at the sweep-max case. Each cfg gets one subprocess
launch with a 60s timeout. Outcomes: OK / TIMEOUT / ERROR.

Output: outputs/autotune_timing_rerun/hung_cfgs.json
"""
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from tilebench.paths import OUTPUT_ROOT, PROFILING_ROOT, REPO_ROOT

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))

HARNESS = PROFILING_ROOT / "ncu_generic_harness.py"
OUT_DIR = OUTPUT_ROOT / "autotune_timing_rerun"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT_S = 60

# Sweep-max test cases per op-dtype.
TEST_CASES = {
    "softmax": {
        "fp16": dict(n_rows=2048, n_cols=10240),
        "fp32": dict(n_rows=2048, n_cols=10240),
    },
    "matmul_fp32_fp16_fp8": {
        "fp16":       dict(M=4096, N=4096, K=20480),
        "fp32":       dict(M=4096, N=4096, K=20480),
        "fp8_e4m3fn": dict(M=4096, N=4096, K=20480),
        "fp8_e5m2":   dict(M=4096, N=4096, K=20480),
    },
    "kl_divergence": {
        "fp32": dict(rows=4096, cols=16384),
    },
    "histogramming": {
        "int32": dict(N=67108864, num_bins=4096),
    },
}


def enum_cutile_cfgs(op: str) -> list[dict]:
    """Read the operator's impl_cutile module-level search-space lists and
    flatten to a list of cfg dicts."""
    mod = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_cutile")
    for name in ("_SEARCH_SPACE", "_SEARCH_SPACE_BASE"):
        space = getattr(mod, name, None)
        if space:
            return [vars(c) if hasattr(c, "__dict__") else dict(c) for c in space]
    # histogramming has split partial/reduce
    p = getattr(mod, "_PARTIAL_SEARCH_SPACE", None)
    r = getattr(mod, "_REDUCE_SEARCH_SPACE", None)
    if p and r:
        return [{**vars(pp), **{f"reduce_{k}": v for k, v in vars(rr).items()}}
                for pp in p for rr in r]
    return []


def enum_triton_cfgs(op: str) -> list[dict]:
    mod = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_triton")
    for name in dir(mod):
        obj = getattr(mod, name)
        configs = getattr(obj, "configs", None)
        if configs and hasattr(obj, "fn"):
            out = []
            for c in configs:
                d = dict(getattr(c, "kwargs", {}) or {})
                for attr in ("num_warps", "num_stages", "num_ctas"):
                    v = getattr(c, attr, None)
                    if v is not None:
                        d[attr] = v
                out.append(d)
            return out
    return []


def probe_one(op: str, backend: str, dtype: str,
              params: dict, cfg: dict) -> dict:
    env = dict(os.environ)
    env["NCU_OP"] = op
    env["NCU_BACKEND"] = backend
    env["NCU_DTYPE"] = dtype
    env["NCU_PARAMS_JSON"] = json.dumps(params)
    env["NCU_CFG_JSON"] = json.dumps(cfg)
    env["PYTHONPATH"] = str(ROOT)
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(HARNESS)],
            env=env, cwd=str(ROOT),
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        return {
            "ok": r.returncode == 0,
            "rc": r.returncode,
            "elapsed_s": round(time.time() - t0, 2),
            "stderr_tail": r.stderr[-400:] if r.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False, "rc": -1,
            "elapsed_s": TIMEOUT_S,
            "stderr_tail": "TIMEOUT",
        }


def main() -> None:
    results = []
    for op, dt_cases in TEST_CASES.items():
        for dt, params in dt_cases.items():
            for backend in ("triton", "cutile"):
                cfgs = (enum_triton_cfgs(op) if backend == "triton"
                        else enum_cutile_cfgs(op))
                print(f"\n=== {op}/{dt}/{backend}  ({len(cfgs)} cfgs) ===",
                      flush=True)
                ok_cnt = 0
                hang_cnt = 0
                err_cnt = 0
                for cfg in cfgs:
                    r = probe_one(op, backend, dt, dict(params), cfg)
                    status = ("OK" if r["ok"]
                              else "TIMEOUT" if r["stderr_tail"] == "TIMEOUT"
                              else "ERR")
                    if r["ok"]:
                        ok_cnt += 1
                    elif status == "TIMEOUT":
                        hang_cnt += 1
                    else:
                        err_cnt += 1
                    flag = ("✓" if r["ok"]
                            else "✗-hang" if status == "TIMEOUT"
                            else "✗-err")
                    print(f"  {flag:11s} {r['elapsed_s']:>6.1f}s  cfg={cfg}",
                          flush=True)
                    results.append({
                        "op": op, "dtype": dt, "backend": backend,
                        "cfg": cfg, **r, "status": status,
                    })
                print(f"  → ok={ok_cnt}  hang={hang_cnt}  err={err_cnt}",
                      flush=True)
    out = OUT_DIR / "hung_cfgs.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
