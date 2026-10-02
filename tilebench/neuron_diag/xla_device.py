"""XLA-stack device timing of one case with the existing runtime-inspect timer.

    python -m tilebench.neuron_diag.xla_device --spec spec.json

Runs ``tilebench.core.nki_orchestrator.profile_case_on_neuron`` unchanged (torch
baseline and NKI in fresh worker processes, executed NEFFs matched by SHA256) and
additionally keeps the raw execution list, so each timed iteration can be
summarized three ways (see ``intervals.py``) and the number of physical cores
each execution ran on (``pcores``, from the per-core ``nc_exec_running`` rows)
is recorded. The orchestrator's own latency is the per-iteration *sum* of
execution durations.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback


def run(spec: dict) -> dict:
    from tilebench.core import nki_timer
    from tilebench.core.nki_orchestrator import profile_case_on_neuron
    from tilebench.neuron_diag.bundles import load_bundle
    from tilebench.neuron_diag.intervals import aggregate

    op = spec["operator"]
    captured: dict = {}

    def loader(session_dir, *, data_path, display_name):
        execs, parquet = nki_timer.load_session_executions(
            session_dir, data_path=data_path, display_name=display_name)
        captured["executions"] = execs
        return execs, parquet

    impl_nki = None
    import_error = ""
    try:
        impl_nki = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_nki")
        if getattr(impl_nki, "nki", object()) is None:
            impl_nki, import_error = None, "impl_nki imported with nki=None"
    except Exception as e:  # noqa: BLE001
        import_error = f"{type(e).__name__}: {e}"[:2000]

    bundle = load_bundle(spec["bundle"], spec["bundle_sha256"])
    res = profile_case_on_neuron(
        operator=op, params=spec["params"], dtype_str=spec["dtype"],
        inputs=bundle["inputs"], ref_output=bundle["reference"], impl_nki=impl_nki,
        block_size=spec.get("block_size", 1024), autotune=False,
        verify_atol=spec.get("atol"), verify_rtol=spec.get("rtol"),
        warmup=spec["warmup"], repeat=spec["repeat"],
        base_dir=spec["base_dir"], index_path=spec["index_path"],
        executions_loader=loader)

    out = {"orchestrator": {k: res.get(k) for k in ("torch_ms", "torch_err", "nki_ms", "nki_ok",
                                                   "nki_err", "spec_id", "manifest_path",
                                                   "identity_source")},
           "import_error": import_error, "targets": {}}
    for t in ("torch", "nki"):
        st = res.get(f"{t}_stats")
        out["targets"][t] = {"stats": st}
    spec_dir = os.path.dirname(res["manifest_path"]) if res.get("manifest_path") else None
    wres_path = os.path.join(spec_dir, "worker_result.json") if spec_dir else None
    execs = captured.get("executions")
    if execs is not None and wres_path and os.path.isfile(wres_path):
        with open(wres_path) as f:
            wres = json.load(f)
        for t in ("torch", "nki"):
            windows = ((wres.get(t) or {}).get("windows")) or []
            per_iter = []
            for b, e in windows:
                inside = execs[b:e]
                agg = aggregate([{"start_ns": x["start_ns"], "end_ns": x["end_ns"],
                                  "execution_id": tuple(x["flow_id"])} for x in inside])
                agg["pcores_per_execution"] = [x["pcores"] for x in inside]
                agg["model_ids"] = [x["model_id"] for x in inside]
                per_iter.append(agg)
            out["targets"][t]["per_iteration_intervals"] = per_iter
    out["spec_dir"] = spec_dir
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    try:
        res = run(spec)
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
    with open(spec["out"], "x") as f:
        json.dump(res, f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
