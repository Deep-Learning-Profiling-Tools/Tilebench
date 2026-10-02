"""Native-stack device timing for one (operator, case, mode): a separate process after the
correctness/wall worker, mirroring xla_device for the XLA stack.

Each timed run() executes inside its own torch.profiler session with
``torch_neuronx.profiling.NeuronConfig(modes=[RUNTIME])``, bracketed by device syncs, so every
execution in that session's NRT system trace (``ntrace.pb``) belongs to that run(). The trace
is read with the same loader as the XLA path (``tilebench.core.nki_timer``: nc_exec_running
rows, per-core copies of one execution merged by (model id, flow id)) and aggregated with
``intervals.aggregate`` (busy sum / union / span; overlap reported, never double counted).

device_ms = mean over iterations of the busy sum (the XLA metric). The timing is accepted only
when every iteration ran at least one execution, every iteration ran the same multiset of
runtime model ids (no compile or different executable inside the window), and the profiled
output still verifies; otherwise device_ms stays null with the reason.

Host wall time is never used here: profiler sessions add host overhead, so this phase reports
device intervals only.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import inspect
import json
import os
import statistics
import sys
import traceback


def run(spec: dict) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from torch_neuronx.profiling import NeuronConfig, ProfileMode

    from tilebench.core import nki_timer
    from tilebench.core.verifier import verify
    from tilebench.neuron_diag.bundles import load_bundle
    from tilebench.neuron_diag.intervals import aggregate
    from tilebench.neuron_diag.runtime import get_runtime
    from tilebench.neuron_diag.schema import EXECUTION_MODES

    op, mode = spec["operator"], spec["mode"]
    stack, target = EXECUTION_MODES[mode]
    assert stack == "native", mode
    rt = get_runtime("native")
    mod = importlib.import_module(f"tilebench.benchmarks.operators.{op}."
                                  f"{'impl_torch' if target == 'torch' else 'impl_nki'}")
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
    x = rt.to_device(main["inputs"])
    rt.sync()
    atol, rtol = spec.get("atol"), spec.get("rtol")
    out = fn(*x, **kw)
    rt.sync(out)
    ok, err = verify(rt.to_cpu(out), main["reference"], atol=atol, rtol=rtol)
    if not ok:
        return {"status": "unavailable", "reason": f"first call does not verify: {err[:500]}"}
    mutates = spec.get("run_mutates_inputs", False)

    def restore():
        nonlocal x
        if mutates:
            x = rt.to_device(main["inputs"])
            rt.sync()

    for _ in range(spec["warmup"]):
        restore()
        o = fn(*x, **kw)
        rt.sync(o)

    iters, last = [], None
    for i in range(spec["repeat"]):
        restore()
        d = os.path.join(spec["profile_dir"], f"iter{i}")
        os.makedirs(d, exist_ok=False)
        cfg = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=d)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                     experimental_config=cfg):
            rt.sync()
            o = fn(*x, **kw)
            rt.sync(o)
        last = o
        sess = nki_timer.find_session_dir(d)
        execs, _ = nki_timer.load_session_executions(sess, data_path=os.path.join(d, "ne"),
                                                     display_name=f"{op}-{mode}-{i}")
        agg = aggregate([{"start_ns": e["start_ns"], "end_ns": e["end_ns"],
                          # flow ids are small per-model counters: key by model too
                          "execution_id": (str(e.get("model_id")), tuple(e["flow_id"]))} for e in execs])
        agg["model_ids"] = [str(e.get("model_id")) for e in execs]
        agg["pcores_per_execution"] = [e.get("pcores") for e in execs]
        iters.append(agg)

    ok_last, err_last = verify(rt.to_cpu(last), main["reference"], atol=atol, rtol=rtol)
    ids = [tuple(sorted(collections.Counter(it["model_ids"]).items())) for it in iters]
    res = {"iterations": iters, "profiled_output_verifies": ok_last,
           "model_id_multiset_stable": len(set(ids)) == 1,
           "metric": "per-iteration busy sum of nc_exec_running executions (per-core rows of one "
                     "execution merged by flow id); one torch.profiler NeuronConfig(RUNTIME) "
                     "session per synced run()"}
    problems = []
    if not ok_last:
        problems.append(f"profiled output does not verify: {err_last[:300]}")
    if any(it["n_executions"] == 0 for it in iters):
        problems.append("an iteration recorded no device execution")
    if len(set(ids)) != 1:
        problems.append("model ids differ between iterations (compile or different executable in window)")
    if problems:
        res.update(status="unavailable", reason="; ".join(problems))
        return res
    busy = [it["busy_sum_ms"] for it in iters]
    res.update(status="ok", device_ms=statistics.fmean(busy), device_samples_ms=busy,
               union_ms=[it["union_ms"] for it in iters], span_ms=[it["span_ms"] for it in iters],
               overlap_between_executions=[it.get("overlap_between_executions") for it in iters],
               executions_per_iteration=iters[0]["n_executions"],
               artifact_identity=sorted({m for it in iters for m in it["model_ids"]}))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    try:
        res = run(spec)
    except Exception as e:  # noqa: BLE001
        res = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"[:2000],
               "traceback": traceback.format_exc()[-4000:]}
    with open(spec["out"], "x") as f:
        json.dump(res, f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
