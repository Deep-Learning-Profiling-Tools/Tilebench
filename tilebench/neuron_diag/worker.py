"""Execute one (operator, case, mode) in a fresh process.

    python -m tilebench.neuron_diag.worker --spec spec.json

The parent (``scripts/neuron_diag.py``) starts this with the operator's overlay as
the only ``tilebench`` on ``PYTHONPATH`` and enforces the timeout. The worker
writes exactly one JSON record to ``spec["out"]``.

Order of work (only step 5 is timed as steady state):
  1. import the pinned impl module and the runtime adapter
  2. move the bundle inputs to the device (untimed)
  3. first call: compile + run + sync, reported as ``first_call_ms``
  4. verify against the CPU reference; verify a second, different bundle; detect
     whether run() mutates its inputs
  5. ``warmup`` untimed and ``repeat`` timed calls; each sample is host wall time of
     run() plus device synchronization. Inputs that run() mutates are restored
     from the bundle before every iteration, outside the timed window.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import statistics
import sys
import time
import traceback


def _classify(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, NotImplementedError):
        return "unsupported_dtype_shape"
    if isinstance(exc, (ImportError, AttributeError)) and "import" in text.lower():
        return "import_incompatible"
    if any(k in text for k in ("NCC_", "neuronx-cc", "Compilation", "compile", "Compiler")):
        return "compile_failure"
    return "runtime_failure"


def _error_metrics(out, ref, atol, rtol) -> dict:
    """max |out-ref| and max |out-ref|/|ref| over all outputs, plus the effective
    tolerance (the verifier's per-dtype default when the config sets none)."""
    import torch

    from tilebench.core.verifier import _TOLERANCES

    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    mabs = mrel = 0.0
    tol = []
    for o, r in zip(outs, refs):
        o64, r64 = o.detach().double(), r.detach().double()
        d = (o64 - r64).abs()
        if d.numel():
            mabs = max(mabs, float(d.max()))
            mrel = max(mrel, float((d / r64.abs().clamp_min(1e-30)).max()))
        da, dr = _TOLERANCES.get(o.dtype, (1e-2, 1e-2))
        tol.append({"dtype": str(o.dtype).replace("torch.", ""),
                    "atol": atol if atol is not None else da, "rtol": rtol if rtol is not None else dr})
    return {"max_abs_err": mabs, "max_rel_err": mrel, "tolerance": tol}


def _same(a, b) -> bool:
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def run(spec: dict) -> dict:
    import torch

    from tilebench.core.verifier import verify
    from tilebench.neuron_diag.bundles import int64_tensors, load_bundle
    from tilebench.neuron_diag.runtime import RuntimeUnavailable, get_runtime
    from tilebench.neuron_diag.schema import EXECUTION_MODES

    op, mode = spec["operator"], spec["mode"]
    stack, target = EXECUTION_MODES[mode]
    rec = {"operator": op, "case_id": spec["case_id"], "mode": mode, "status": "not_run",
           "shape": spec["params"], "dtype": spec["dtype"], "source_hash": spec["source_hash"],
           "config": spec.get("config", {}), "warmup": spec["warmup"], "repeat": spec["repeat"],
           "smoke": spec.get("smoke", False), "compile_info": {}}

    try:
        rt = get_runtime(stack)
    except RuntimeUnavailable as e:
        rec.update(status="blocked_env", reason=str(e))
        return rec
    rec["compile_info"]["runtime_api"] = rt.api

    try:
        if target == "torch":
            mod = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_torch")
        else:
            mod = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_nki")
            if getattr(mod, "nki", object()) is None:
                rec.update(status="import_incompatible", reason="impl_nki imported with nki=None")
                return rec
    except Exception as e:  # noqa: BLE001
        rec.update(status="import_incompatible", error=f"{type(e).__name__}: {e}"[:2000])
        return rec

    fn = mod.run
    kw: dict = {}
    if target == "nki":
        sig = inspect.signature(fn).parameters
        if "block_size" in sig:
            kw["block_size"] = spec.get("block_size", 1024)
        if "autotune" in sig:
            kw["autotune"] = False
    if mode == "native_torch_compiled":
        fn = rt.compile(fn)

    main = load_bundle(spec["bundle"], spec["bundle_sha256"])
    alt = load_bundle(spec["alt_bundle"], spec["alt_bundle_sha256"])
    if stack == "native":
        i64 = int64_tensors(main["inputs"])
        if i64:
            rec["compile_info"]["int64_inputs"] = i64
    x = rt.to_device(main["inputs"])
    rt.sync()
    if stack == "native":
        moved = [str(t.dtype) for t in x if isinstance(t, torch.Tensor)]
        orig = [str(t.dtype) for t in main["inputs"] if isinstance(t, torch.Tensor)]
        if moved != orig:
            rec.update(status="unsupported_dtype_shape",
                       reason=f"device transfer changed dtypes {orig} -> {moved}")
            return rec

    atol, rtol = spec.get("atol"), spec.get("rtol")
    try:
        t0 = time.perf_counter()
        out = fn(*x, **kw)
        rt.sync(out)
        rec["first_call_ms"] = (time.perf_counter() - t0) * 1e3
        out_cpu = rt.to_cpu(out)
    except Exception as e:  # noqa: BLE001
        rec.update(status=_classify(e), error=f"{type(e).__name__}: {e}"[:4000],
                   reason="first call raised")
        return rec

    ok, err = verify(out_cpu, main["reference"], atol=atol, rtol=rtol)
    rec["compile_info"]["correctness"] = _error_metrics(out_cpu, main["reference"], atol, rtol)
    rec["compile_info"]["correctness"]["reference_source"] = (
        "CPU impl_torch.run of the pinned PR head, bundle " + spec["bundle_sha256"][:16])
    rec["verification_status"] = "pass" if ok else "fail"
    rec["verification_detail"] = err[:2000]
    if not ok:
        rec.update(status="correctness_failure")
        return rec

    x_after = rt.to_cpu(x)
    mutates = not _same(x_after, main["inputs"])
    rec["compile_info"]["run_mutates_inputs"] = mutates

    try:
        x2 = rt.to_device(alt["inputs"])
        out2 = rt.to_cpu(fn(*x2, **kw))
        rt.sync()
        ok2, err2 = verify(out2, alt["reference"], atol=atol, rtol=rtol)
        stale = _same(out2, out_cpu) and not _same(alt["reference"], main["reference"])
        rec["changed_input_status"] = "pass" if ok2 and not stale else "fail"
        if stale:
            err2 = "second input produced the first output (stale/cached result)"
        if rec["changed_input_status"] == "fail":
            rec.update(status="correctness_failure",
                       verification_detail=f"changed input: {err2}"[:2000])
            return rec
        del x2, out2
    except Exception as e:  # noqa: BLE001
        rec.update(status=_classify(e), error=f"changed input: {type(e).__name__}: {e}"[:4000])
        return rec

    def restore():
        nonlocal x
        if mutates:
            x = rt.to_device(main["inputs"])
            rt.sync()

    compiles_before = rt.compile_count()
    try:
        for _ in range(spec["warmup"]):
            restore()
            o = fn(*x, **kw)
            rt.sync(o)
        samples, dispatch, waits = [], [], []
        rt.fallback_begin()
        for _ in range(spec["repeat"]):
            restore()
            t0 = time.perf_counter()
            o = fn(*x, **kw)
            t1 = time.perf_counter()
            rt.sync(o)
            t2 = time.perf_counter()
            samples.append((t2 - t0) * 1e3)
            dispatch.append((t1 - t0) * 1e3)
            waits.append((t2 - t1) * 1e3)
            del o
        fb_state, fb_evidence = rt.fallback_end()
    except Exception as e:  # noqa: BLE001
        rec.update(status=_classify(e), error=f"timed loop: {type(e).__name__}: {e}"[:4000])
        return rec

    rec["compile_info"].update(rt.compile_info())
    compiles_after = rt.compile_count()
    rec["compile_info"]["recompiles_during_warmup_and_timed_loop"] = (
        None if compiles_before is None or compiles_after is None
        else compiles_after - compiles_before)
    # Host side of each timed call: time until run() returned (tracing / dispatch /
    # custom-call setup; lazy on XLA) and the wait inside the synchronization.
    rec["compile_info"]["wall_split_ms"] = {
        "dispatch_mean": statistics.fmean(dispatch) if dispatch else None,
        "sync_wait_mean": statistics.fmean(waits) if waits else None,
        "dispatch_median": statistics.median(dispatch) if dispatch else None,
        "sync_wait_median": statistics.median(waits) if waits else None,
        "wall_median": statistics.median(samples) if samples else None,
    }
    rec["compile_info"]["profiling_env"] = {k: v for k, v in os.environ.items()
                                            if k.startswith("NEURON_RT_INSPECT")}
    rec.update(wall_ms=statistics.fmean(samples) if samples else None,
               wall_samples_ms=samples, wall_timing_method="wall_sync" if samples else None,
               timing_scope="full_run", fallback_status=fb_state, fallback_evidence=fb_evidence,
               device_unavailable_reason=spec.get("device_unavailable_reason",
                                                  "device timing is taken in a separate phase"))
    rec["status"] = "cpu_fallback" if fb_state == "confirmed_cpu_fallback" else "pass"
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    try:
        rec = run(spec)
    except Exception as e:  # noqa: BLE001 - reported, never swallowed silently
        rec = {"operator": spec.get("operator"), "case_id": spec.get("case_id"),
               "mode": spec.get("mode"), "status": "runtime_failure",
               "error": f"worker crashed: {type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"}
    with open(spec["out"], "x") as f:
        json.dump(rec, f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
