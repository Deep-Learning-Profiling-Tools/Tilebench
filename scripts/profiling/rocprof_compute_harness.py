"""ROCm Compute Profiler harness — the process rocprof-compute profiles.

The AMD counterpart of ncu_generic_harness.py: same pair, same inputs, same
autotune winner (tilebench.profiling.replay), same cache eviction. Parameterized
by env vars, like the NCU harness:

  PROF_OP           operator name (e.g. "vector_add")
  PROF_BACKEND      "triton"
  PROF_DTYPE        dtype string (e.g. "fp32"); converted to torch.dtype and
                    injected into params if the generator takes a `dtype` kwarg
  PROF_PARAMS_JSON  JSON dict of params passed to GENERATORS[op](**params)
  PROF_CFG_JSON     JSON dict of the autotune winner (optional)
  PROF_MODE         "prime" or "profile" (default "profile")

PROF_MODE=prime (run without a profiler, before profiling): input generation,
then 3 complete impl.run() calls. This is the JIT warmup: it compiles the
winner's kernels into Triton's on-disk cache, so the profiled process only
loads them.

PROF_MODE=profile (run under rocprof-compute): input generation, the
last-level-cache eviction of the benchmark timer (2x the LLC: 512 MiB on
MI300X), then exactly ONE impl.run(). It is the first call of the process.

Why the profiled process has no in-process warmup, unlike the NCU harness:
rocprof-compute has no profiler start/stop range (NCU's --profile-from-start
off). It selects dispatches by kernel-name regex (-k) and by a dispatch range
(-d) that counts iterations PER KERNEL, not across the run (measured on
MI300X, rocprof-compute 3.7.0: with 3 warmups of an A,B,B,C pipeline, -d 4
captures A's, B's and C's 4th dispatch, one of them inside a warmup). When the
kernels of one call launch different numbers of times, no -d range selects one
whole call after warmups. Without warmups in the process, every dispatch that
matches the operator's kernel names is the measured call, so -k alone selects
exactly ONE impl.run(): input generation, the eviction fill, memcpy blits and
ATen helpers never match. probe_kernel_count.py checks for every pair that the
first call launches the same compute sequence as later calls
(first_call_identical), and the driver validates every capture against it.

This file is the process rocprof-compute profiles, not a library: the driver
starts it by path.
"""
import importlib
import json
import os
import sys

import torch
# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.data.tensors import GENERATORS  # noqa: E402
from tilebench.profiling.replay import apply_winner, resolve_params  # noqa: E402

PRIME_CALLS = 3


def main() -> None:
    op       = os.environ["PROF_OP"]
    backend  = os.environ["PROF_BACKEND"]
    params   = json.loads(os.environ.get("PROF_PARAMS_JSON", "{}"))
    cfg_json = os.environ.get("PROF_CFG_JSON")
    dtype    = os.environ.get("PROF_DTYPE")
    mode     = os.environ.get("PROF_MODE", "profile")
    if mode not in ("prime", "profile"):
        raise RuntimeError(f"PROF_MODE must be prime or profile, not {mode!r}")

    params = resolve_params(params, dtype)
    impl = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_{backend}")
    if cfg_json:
        apply_winner(impl, json.loads(cfg_json), params.get("dtype"))

    if op not in GENERATORS:
        raise RuntimeError(f"no GENERATORS entry for op={op}")
    inputs = GENERATORS[op](**params)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    torch.cuda.synchronize()  # drain all generator launches

    if mode == "prime":
        for _ in range(PRIME_CALLS):
            out = impl.run(*inputs)
            torch.cuda.synchronize()
    else:
        from tilebench.core.timer import _flush_l2_cache
        _flush_l2_cache()          # same auto-sized (2x device LLC) eviction as Proton
        torch.cuda.synchronize()

        out = impl.run(*inputs)    # the ONE measured call
        torch.cuda.synchronize()

    if isinstance(out, torch.Tensor):
        print(f"{op}/{backend}/{dtype} {mode} ok: shape={tuple(out.shape)} dtype={out.dtype}")
    else:
        print(f"{op}/{backend}/{dtype} {mode} ok (non-tensor output)")


if __name__ == "__main__":
    main()
