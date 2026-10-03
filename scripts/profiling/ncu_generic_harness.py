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
import sys


import torch
# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.data.tensors import GENERATORS  # noqa: E402
# Shared with the probe and the ROCm Compute Profiler harness, so all of them
# replay the same params and the same winner.
from tilebench.profiling.replay import apply_winner, resolve_params  # noqa: E402


def main() -> None:
    op       = os.environ["NCU_OP"]
    backend  = os.environ["NCU_BACKEND"]
    params   = json.loads(os.environ.get("NCU_PARAMS_JSON", "{}"))
    cfg_json = os.environ.get("NCU_CFG_JSON")
    dtype    = os.environ.get("NCU_DTYPE")

    params = resolve_params(params, dtype)

    impl = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_{backend}")

    if cfg_json:
        apply_winner(impl, json.loads(cfg_json), params.get("dtype"), strict=(backend == "tilelang"))

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
