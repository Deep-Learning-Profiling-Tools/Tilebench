"""Probe how many CUDA kernels each impl.run() launches per single call.

For every (op, backend) we:
  - import the impl
  - generate inputs at the sweep-max case
  - run a warmup call (loads JIT cache, autotune cache, etc.)
  - run a second call inside torch.profiler and count distinct device
    kernels triggered between the two synchronization points

Output: tilebench_run/ncu/kernel_counts.json
        and a short summary table to stdout.
"""
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
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


def count_one(op: str, backend: str, params: dict, cfg: dict | None, dtype: str) -> dict:
    td = DTYPE_MAP.get(dtype)
    if td is not None:
        params = {**params, "dtype": td}

    impl = importlib.import_module(f"benchmarks.operators.{op}.impl_{backend}")
    if cfg:
        existing = getattr(impl, "_DEFAULT_CONFIG", None)
        if isinstance(existing, dict):
            merged = dict(existing); merged.update(cfg)
            impl._DEFAULT_CONFIG = merged
        elif isinstance(existing, SimpleNamespace):
            merged = vars(existing).copy(); merged.update(cfg)
            impl._DEFAULT_CONFIG = SimpleNamespace(**merged)
        else:
            impl._DEFAULT_CONFIG = SimpleNamespace(**cfg)

    inputs = GENERATORS[op](**params)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # warmup
    impl.run(*inputs)
    torch.cuda.synchronize()

    # measured call inside profiler
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        impl.run(*inputs)
        torch.cuda.synchronize()

    kernels = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            name = e.key
            if "memcpy" in name.lower() or "memset" in name.lower():
                continue
            kernels.append(name)
    return {"count": len(kernels), "names": kernels[:10]}


def main() -> None:
    catalogue = json.loads((ROOT / "tilebench_run" / "ncu_catalogue.json").read_text())
    out_path = ROOT / "tilebench_run" / "ncu" / "kernel_counts.json"
    counts: list[dict] = []

    only_op = os.environ.get("ONLY_OP")

    for entry in catalogue:
        op = entry["op"]
        if only_op and op != only_op:
            continue
        for dt in entry["dtypes"]:
            params = entry["default_params_per_dtype"][dt]
            winner = entry["autotune_winner_per_dtype"].get(dt)
            for backend in ("triton", "cutile"):
                cfg = (winner or {}).get(backend)
                try:
                    r = count_one(op, backend, dict(params), cfg, dt)
                    counts.append({"op": op, "dtype": dt, "backend": backend,
                                   "count": r["count"], "names": r["names"]})
                except Exception as e:
                    counts.append({"op": op, "dtype": dt, "backend": backend,
                                   "count": None, "error": f"{type(e).__name__}: {e}"[:200]})

    out_path.write_text(json.dumps(counts, indent=2))

    # Print summary: ops where ANY backend launches >1 kernel
    max_per_op = {}
    for r in counts:
        if r["count"] is None:
            continue
        key = (r["op"], r["dtype"])
        cur = max_per_op.get(key, 0)
        if r["count"] > cur:
            max_per_op[key] = r["count"]

    print(f"\n{'='*80}\nOps where impl.run() launches >1 kernel per call\n{'='*80}")
    print(f"{'op':30s} {'dtype':10s} {'max_kernels_per_call':>22s}")
    multi = sorted(
        ((k[0], k[1], v) for k, v in max_per_op.items() if v > 1),
        key=lambda t: (-t[2], t[0]),
    )
    for op, dt, n in multi:
        print(f"{op:30s} {dt:10s} {n:>22d}")
    print(f"\nTotal ops with multi-kernel launches: {len(multi)}")

    # Errors
    errs = [r for r in counts if r.get("error")]
    if errs:
        print(f"\nErrored probes: {len(errs)}")
        for e in errs[:10]:
            print(f"  {e['op']}/{e['dtype']}/{e['backend']}: {e['error']}")


if __name__ == "__main__":
    main()
