"""Generic NCU harness — parameterized by env vars.

Env:
  NCU_OP          operator name (e.g. "matmul_int8")
  NCU_BACKEND     "triton", "cutile" or "tilelang"
  NCU_PARAMS_JSON JSON dict of params passed to GENERATORS[op](**params)
  NCU_CFG_JSON    JSON dict for `_DEFAULT_CONFIG` override (optional)
  NCU_DTYPE       dtype string (e.g. "fp16"); converted to torch.dtype and
                  injected into params if op's generator takes a `dtype` kwarg

Usage (under ncu):
  NCU_OP=matmul_int8 NCU_BACKEND=cutile \
    NCU_PARAMS_JSON='{"M":2048,"N":2048,"K":20480}' \
    NCU_CFG_JSON='{"tm":256,"tn":64,"tk":32,"group_size_m":8,"occupancy":8}' \
    NCU_DTYPE=int8 \
    ncu --set full --import-source yes --source-folders "$PWD,..." \
        --profile-from-start off \
        --replay-mode application --cache-control none \
        --kernel-name regex:"matmul" --launch-skip 0 --launch-count 1 \
        -o out python ncu_generic_harness.py
"""
import importlib
import json
import os
import sys

from ncu_common import apply_config_override, register_tilelang_source_capture, repo_root

ROOT = repo_root()
sys.path.insert(0, str(ROOT))

import torch
from data.tensors import GENERATORS


DTYPE_MAP = {
    "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
    "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
    "int8": torch.int8, "int32": torch.int32, "int64": torch.int64,
    "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2":   getattr(torch, "float8_e5m2", None),
}

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

    register_tilelang_source_capture(op, backend, dtype or "unknown")
    impl = importlib.import_module(f"benchmarks.operators.{op}.impl_{backend}")

    if cfg_json:
        apply_config_override(impl, json.loads(cfg_json), params.get("dtype"))

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

    from core.timer import _flush_l2_cache
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
