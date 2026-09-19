"""Generic NCU harness — parameterized by env vars.

Env:
  NCU_OP          operator name (e.g. "matmul_int8")
  NCU_BACKEND     "triton" or "cutile"
  NCU_PARAMS_JSON JSON dict of params passed to GENERATORS[op](**params)
  NCU_CFG_JSON    JSON dict for `_DEFAULT_CONFIG` override (optional)
  NCU_DTYPE       dtype string (e.g. "fp16"); converted to torch.dtype and
                  injected into params if op's generator takes a `dtype` kwarg

Usage (under ncu):
  NCU_OP=matmul_int8 NCU_BACKEND=cutile \
    NCU_PARAMS_JSON='{"M":2048,"N":2048,"K":20480}' \
    NCU_CFG_JSON='{"tm":256,"tn":64,"tk":32,"group_size_m":8,"occupancy":8}' \
    NCU_DTYPE=int8 \
    ncu --set full --import-source on --launch-skip 3 --launch-count 1 \
        --kernel-name regex:"matmul" -o out python scripts/profiling/ncu_generic_harness.py

This file is the process NCU profiles, not a library: ncu_driver.py and
ncu_one.py, its siblings, start it by path.
"""
import importlib
import json
import os
import re
import sys
from types import SimpleNamespace


import torch
# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.data.tensors import GENERATORS  # noqa: E402


DTYPE_MAP = {
    "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
    "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
    "int8": torch.int8, "int32": torch.int32, "int64": torch.int64,
    "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2":   getattr(torch, "float8_e5m2", None),
}


_PREFIXED_CONFIG_RE = re.compile(r"^_DEFAULT_([A-Z0-9]+)_CONFIG$")


def _inject_prefixed(impl, cfg: dict) -> dict:
    """Route prefixed winner keys into the per-kernel config dicts run() reads.

    Operators with more than one tunable kernel keep their configs in
    per-kernel dicts named `_DEFAULT_<PREFIX>_CONFIG` (linear_self_attention:
    `_DEFAULT_KV_CONFIG` / `_DEFAULT_OUT_CONFIG`), and the catalogue prefixes
    their winner keys to match (`kv_BLOCK_M`, `out_num_warps`, ...). Without
    this routing the override lands on an unused `_DEFAULT_CONFIG` attribute
    and NCU silently profiles the DEFAULT config instead of the winner.

    Mutates the target dicts in place (never rebinds them, so a `run()` that
    captured a reference still sees the update). Returns the unconsumed keys
    for the caller's existing single-config handling.
    """
    targets = {}
    for attr in dir(impl):
        m = _PREFIXED_CONFIG_RE.match(attr)
        if m:
            targets[m.group(1).lower()] = (attr, getattr(impl, attr))
    if not targets:
        return cfg

    leftover = {}
    for key, val in cfg.items():
        placed = False
        for pref, (attr, target) in targets.items():
            if not key.lower().startswith(pref + "_"):
                continue
            base = key[len(pref) + 1:]
            if isinstance(target, dict) and base in target:
                target[base] = val
                placed = True
            elif isinstance(target, SimpleNamespace) and hasattr(target, base):
                setattr(target, base, val)
                placed = True
            if placed:
                print(f"  cfg: {attr}[{base}] = {val}")
                break
        if not placed:
            leftover[key] = val
    return leftover


def main() -> None:
    op       = os.environ["NCU_OP"]
    backend  = os.environ["NCU_BACKEND"]
    params   = json.loads(os.environ.get("NCU_PARAMS_JSON", "{}"))
    cfg_json = os.environ.get("NCU_CFG_JSON")
    dtype    = os.environ.get("NCU_DTYPE")

    if dtype is not None and "dtype" not in params:
        td = DTYPE_MAP.get(dtype)
        if td is None:
            raise RuntimeError(f"unknown dtype {dtype}")
        params["dtype"] = td
    elif isinstance(params.get("dtype"), str):
        td = DTYPE_MAP.get(params["dtype"])
        if td is not None:
            params["dtype"] = td

    impl = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_{backend}")

    if cfg_json:
        cfg = _inject_prefixed(impl, json.loads(cfg_json))

    if cfg_json and cfg:
        existing = getattr(impl, "_DEFAULT_CONFIG", None)
        configs = getattr(impl, "_DEFAULT_CONFIGS", None)  # per-dtype dict, optional

        if existing is None and configs is not None:
            # Op uses per-dtype `_DEFAULT_CONFIGS[dtype]` (e.g. matmul_fp32_fp16_fp8).
            # Override the entry for the dtype we're about to run so the kernel
            # actually picks up `cfg`. Singular `_DEFAULT_CONFIG` is irrelevant
            # to this impl, so don't bother setting it.
            td = params.get("dtype")  # this is the torch.dtype already resolved
            if td in configs:
                cur = configs[td]
                if isinstance(cur, dict):
                    merged = dict(cur); merged.update(cfg)
                    configs[td] = merged
                else:
                    merged = vars(cur).copy(); merged.update(cfg)
                    configs[td] = SimpleNamespace(**merged)
            else:
                # No entry for this dtype — create one wholesale.
                configs[td] = SimpleNamespace(**cfg)
        elif isinstance(existing, dict):
            merged = dict(existing); merged.update(cfg)
            impl._DEFAULT_CONFIG = merged
        elif isinstance(existing, SimpleNamespace):
            merged = vars(existing).copy(); merged.update(cfg)
            impl._DEFAULT_CONFIG = SimpleNamespace(**merged)
        else:
            impl._DEFAULT_CONFIG = SimpleNamespace(**cfg)

    if op not in GENERATORS:
        raise RuntimeError(f"no GENERATORS entry for op={op}")
    inputs = GENERATORS[op](**params)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    torch.cuda.synchronize()  # drain all generator launches before profiling

    # Unified operator-level methodology: warmups (JIT/autotune caches) and a
    # manual 256 MB L2 eviction happen OUTSIDE the profiler range, then exactly
    # ONE complete impl.run() executes inside cudaProfilerStart/Stop. The
    # operator starts entry-cold while intra-operator producer-consumer L2
    # reuse is preserved (NCU runs with `--replay-mode application
    # --cache-control none`, so it never purges caches between the operator's
    # internal kernels). With `--profile-from-start off`, NCU's launch counter
    # sees only the measured run below, so the driver selects kernels with
    # `--launch-skip 0 --launch-count <matched-launches-per-run>`.
    for _ in range(3):
        out = impl.run(*inputs)
        torch.cuda.synchronize()

    from tilebench.core.timer import _flush_l2_cache
    _flush_l2_cache()          # same auto-sized (2x device L2) eviction as Proton
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    out = impl.run(*inputs)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    if isinstance(out, torch.Tensor):
        print(f"{op}/{backend}/{dtype} ok: shape={tuple(out.shape)} dtype={out.dtype}")
    else:
        print(f"{op}/{backend}/{dtype} ok (non-tensor output)")


if __name__ == "__main__":
    main()
