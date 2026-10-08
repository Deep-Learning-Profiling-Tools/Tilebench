"""Probe the kernel launches of ONE impl.run() per (op, dtype, backend).

For every catalogue pair of the selected backends we:
  - import the impl and install the exact autotune winner, the same way the
    profiler harnesses do (tilebench.profiling.replay)
  - generate inputs at the sweep-max case
  - record the first call (JIT/caches cold) and a later call under
    torch.profiler, and keep the device-kernel launch sequence of each

Output: outputs/profiling/<gpu>/kernel_counts.json (generated profiling
metadata, one file per GPU: launch counts and kernel names differ between
GPUs; it is measured data, kept with the other generated outputs). One row per
pair:

    {"op", "dtype", "backend",
     "count":    compute-kernel launches of ONE impl.run(),
     "names":    those launches in launch order, repeats kept (never deduplicated),
     "excluded": the other device launches of the call, in order: memcpy/memset,
                 ATen/library helpers, runtime blit kernels (ncu_kernel_select
                 .is_aux_kernel); not operator compute kernels,
     "first_call_identical": whether the first call launched the same compute
                 sequence as the later call}

A failed probe records count=None and the error; a pair the GPU's autotune run
has no result for (an unsupported dtype) records count=None and "skipped". A full run rewrites the file;
an ONLY_OP run merges that operator's rows into the existing file instead of
truncating it to one operator. Only the file of --gpu is ever read or written.
A short summary table goes to stdout.

Usage:  python scripts/profiling/probe_kernel_count.py --gpu B200
        python scripts/profiling/probe_kernel_count.py --gpu MI300X --tile-language triton
        ONLY_OP=softmax python scripts/profiling/probe_kernel_count.py --gpu B200
"""
import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.backends import parse_backends  # noqa: E402
from tilebench.paths import REPO_ROOT, hardware_label, kernel_counts_path  # noqa: E402
from tilebench.profiling import ncu_kernel_select as ks  # noqa: E402
ROOT = REPO_ROOT

import torch
from tilebench.data.tensors import GENERATORS  # noqa: E402
from tilebench.profiling.replay import apply_winner, resolve_params  # noqa: E402

#: The backends whose kernels the profilers capture (and so the probe counts).
PROFILED_BACKENDS = ("triton", "cutile", "tilelang")

def _device_launches(prof) -> list[str]:
    """Device launches recorded by torch.profiler, in order. On ROCm torch
    still reports HIP kernels as DeviceType.CUDA (checked on MI300X against a
    rocprofv3 kernel trace: same names, same counts)."""
    return [e.key for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA]


def _profile_call(impl, inputs) -> list[str]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        impl.run(*inputs)
        torch.cuda.synchronize()
    return _device_launches(prof)


def count_one(op: str, backend: str, params: dict, cfg: dict | None, dtype: str) -> dict:
    params = resolve_params(params, dtype)
    # A fresh module per pair, as in the harness's own process: the winner of
    # another dtype of the same operator must not leak into this one (the
    # overrides mutate module-level config dicts).
    impl = importlib.reload(importlib.import_module(
        f"tilebench.benchmarks.operators.{op}.impl_{backend}"))
    apply_winner(impl, cfg, params.get("dtype"), strict=(backend == "tilelang"))

    inputs = GENERATORS[op](**params)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    torch.cuda.synchronize()

    first = _profile_call(impl, inputs)     # JIT / caches cold
    launches = _profile_call(impl, inputs)  # the steady-state call that is recorded

    # Full list, no truncation: the profiler drivers derive the expected launch
    # count from it, so a truncated list makes pipelines of dozens to hundreds
    # of launches (radix_sort, bitonic_sort, top_k) capture only a fragment of
    # the call and report a bogus pipeline duration.
    names = [n for n in launches if not ks.is_aux_kernel(n)]
    return {"count": len(names), "names": names,
            "excluded": [n for n in launches if ks.is_aux_kernel(n)],
            "first_call_identical": [n for n in first if not ks.is_aux_kernel(n)] == names}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe kernel launch counts on the current GPU.")
    ap.add_argument("--gpu", type=hardware_label, required=True, metavar="LABEL",
                    help="Hardware label of the GPU being probed (e.g. B200): reads and writes "
                         "outputs/profiling/<gpu>/ only")
    ap.add_argument("--tile-language", type=str, default="triton,cutile",
                    help="Backends to probe, among the profiled ones (triton, cutile, tilelang); "
                         "default: triton,cutile. A GPU without cuTile (e.g. MI300X) uses triton. "
                         "A run replaces only the rows of the backends it probes")
    args = ap.parse_args()
    try:
        backends = [b for b in parse_backends(args.tile_language) if b in PROFILED_BACKENDS]
    except ValueError as e:
        ap.error(f"--tile-language: {e}")
    if not backends:
        ap.error(f"--tile-language must include one of {', '.join(PROFILED_BACKENDS)}")
    try:
        catalogue = ks.load_catalogue(args.gpu)
    except ks.MissingProfilingMetadataError as e:
        sys.exit(f"error: {e}")
    if "tilelang" in backends and not any(
            (w or {}).get("tilelang") for e in catalogue
            for w in e.get("autotune_winner_per_dtype", {}).values()):
        sys.exit(f"error: the {args.gpu} catalogue holds no TileLang winners; rebuild it with "
                 f"ncu_catalogue.py --tile-language from a run that tuned TileLang")
    out_path = kernel_counts_path(args.gpu)
    counts: list[dict] = []

    only_op = os.environ.get("ONLY_OP")

    for entry in catalogue:
        op = entry["op"]
        if only_op and op != only_op:
            continue
        for dt in entry["dtypes"]:
            params = entry["default_params_per_dtype"][dt]
            winner = entry["autotune_winner_per_dtype"].get(dt)
            for backend in backends:
                if entry.get("has_autotune_log") and not (winner or {}).get(backend):
                    # The autotune run of this GPU produced no result for the pair
                    # (e.g. fp8_e4m3fn on MI300X: UNSUPPORTED_DTYPE, skipped by the
                    # engine). Nothing was benchmarked, so there is nothing to
                    # replay; another dtype or a default config is never used instead.
                    counts.append({"op": op, "dtype": dt, "backend": backend, "count": None,
                                   "skipped": "no autotune result for this pair on this GPU"})
                    continue
                cfg = (winner or {}).get(backend)
                try:
                    r = count_one(op, backend, dict(params), cfg, dt)
                    counts.append({"op": op, "dtype": dt, "backend": backend,
                                   **{k: r[k] for k in ("count", "names", "excluded",
                                                        "first_call_identical") if k in r}})
                except Exception as e:
                    counts.append({"op": op, "dtype": dt, "backend": backend,
                                   "count": None, "error": f"{type(e).__name__}: {e}"[:200]})

    if out_path.exists():
        # A refresh replaces only what it probed: an ONLY_OP run keeps every
        # other operator's rows, and a run over some backends (e.g. tilelang
        # alone) keeps the rows of the others.
        probed = {(r["op"], r["dtype"], r["backend"]) for r in counts}
        kept = [r for r in json.loads(out_path.read_text())
                if (r["op"], r["dtype"], r["backend"]) not in probed
                and (only_op or r["backend"] not in backends)]
        counts = sorted(kept + counts,
                        key=lambda r: (r["op"], r["dtype"], r["backend"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

    # A first call that launches a different compute sequence than later calls
    # (lazy initialisation inside run()): a harness whose measured call is the
    # first one of its process would not see the recorded sequence.
    differing = [r for r in counts if r.get("first_call_identical") is False]
    if differing:
        print(f"\nPairs whose first call launches a different compute sequence: {len(differing)}")
        for r in differing:
            print(f"  {r['op']}/{r['dtype']}/{r['backend']}")

    skipped = [r for r in counts if r.get("skipped")]
    if skipped:
        print(f"\nSkipped (no autotune result on this GPU): {len(skipped)}")
        for r in skipped:
            print(f"  {r['op']}/{r['dtype']}/{r['backend']}")

    # Errors
    errs = [r for r in counts if r.get("error")]
    if errs:
        print(f"\nErrored probes: {len(errs)}")
        for e in errs[:10]:
            print(f"  {e['op']}/{e['dtype']}/{e['backend']}: {e['error']}")


if __name__ == "__main__":
    main()
