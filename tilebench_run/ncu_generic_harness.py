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
        --kernel-name regex:"matmul" -o out python ncu_generic_harness.py
"""
import importlib
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/projects/kzhou6/bcui2/research/tilebench/Tilebench")

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

    impl = importlib.import_module(f"benchmarks.operators.{op}.impl_{backend}")

    if cfg_json:
        cfg = json.loads(cfg_json)
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

    # Wrap warmups + final launch in cudaProfilerStart/Stop. NCU is invoked
    # with `--profile-from-start off`, so the generator-side kernel launches
    # (torch.randn, * scale, .to(dtype), ...) are hidden from NCU's launch
    # counter. `--launch-skip 3N --launch-count N` then correctly indexes
    # into the impl.run() launches only.
    torch.cuda.profiler.start()
    for _ in range(3):
        out = impl.run(*inputs)
        torch.cuda.synchronize()

    out = impl.run(*inputs)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    if isinstance(out, torch.Tensor):
        print(f"{op}/{backend}/{dtype} ok: shape={tuple(out.shape)} dtype={out.dtype}")
    else:
        print(f"{op}/{backend}/{dtype} ok (non-tensor output)")


if __name__ == "__main__":
    main()
