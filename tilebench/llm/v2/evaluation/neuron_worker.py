"""Isolated evaluation worker for Trn2 (subprocess entry point).

    python -m tilebench.llm.v2.evaluation.neuron_worker --job job.json --out result.json

Same job schema, result schema, suite rules and per-case procedure as evaluation.worker (its run_case,
summarize_suite, PrecisionGuard, interface and autotuner checks are reused, not copied); the device pieces
(native neuron device, CPU reference, busy-sum timing, host-fallback detection) come from evaluation.neuron.
Refuses to start without TILEBENCH_V2_SANDBOX=1 (set by the launcher)."""
from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from tilebench.llm.v2.evaluation import worker as W


class _TorchView:
    """The torch module as run_case sees it: identical except that `torch.cuda.synchronize` is the neuron
    device synchronize (run_case synchronizes after every candidate/reference call)."""

    def __init__(self, torch, sync):
        self._torch = torch
        self.cuda = SimpleNamespace(synchronize=sync)

    def __getattr__(self, name):
        return getattr(self._torch, name)


def _nki_autotuner_hits(mod) -> list[str]:
    """Execution-side detection of hidden-search objects reachable from the module (NKI has no autotuner of its
    own in 0.7; the repository's NkiAutotuner is the one available)."""
    hits: list[str] = []
    classes = []
    try:
        from tilebench.core.nki_autotune import NkiAutotuner
        classes.append(("tilebench.core.nki_autotune.NkiAutotuner", NkiAutotuner))
    except Exception:  # noqa: BLE001
        pass
    for name, obj in list(vars(mod).items()):
        cur, depth = obj, 0
        while cur is not None and depth < 6:
            if any(isinstance(cur, cls) for _, cls in classes):
                hits.append(f"{name}: {next(lbl for lbl, cls in classes if isinstance(cur, cls))} instance")
                break
            cur = getattr(cur, "fn", None) or getattr(cur, "func", None)
            depth += 1
        if getattr(obj, "__module__", "") == "tilebench.core.nki_autotune":
            hits.append(f"{name}: object from tilebench.core.nki_autotune")
    return hits


def run_job(job: dict) -> dict:
    t0 = time.time()
    result: dict = {"schema": W.RESULT_SCHEMA, "status": "infrastructure_incomplete", "stages": {}, "diagnostic": None,
                    "identity": job.get("identity"), "expected_timing_mode": job.get("expected_timing_mode"),
                    "cases": [], "cases_total": len(job.get("cases") or []), "cases_evaluated": 0, "valid_cases": 0,
                    "latency_ms_geomean": None, "first_failing_case": None, "suite_stopped": None,
                    "device_backend": "neuron_native"}
    try:
        # launch settings first: `import torch` auto-loads torch_neuronx (backend entry point)
        from tilebench.core import neuron_native
        cache = Path(os.environ.get("TILEBENCH_NEURON_CACHE") or Path(job.get("sandbox_dir") or ".") / "neuron_cache")
        env = neuron_native.prepare_environment(cache)
        from tilebench.llm.v2.evaluation.job import validate_worker_job
        errs = validate_worker_job(job)
        if errs:
            result["diagnostic"] = "invalid job: " + "; ".join(errs)
            return result
        result["neuron_environment"] = {"neuronx_cc": env.get("neuronx_cc"),
                                        "env": {k: v for k, v in env["env"].items() if not k.endswith("_DIR") and "CACHE" not in k}}
        import torch
        import torch_neuronx
        from tilebench.core.dtypes import resolve_dtype
        from tilebench.data.tensors import get_generator
        from tilebench.llm.v2.evaluation.neuron import measure_neuron, run_numerical_checks_neuron, track_host_fallback
        rt = neuron_native.NativeNeuron()
    except Exception:  # noqa: BLE001
        result["diagnostic"] = "worker import / neuron device initialization failure:\n" + traceback.format_exc()
        return result
    W._progress(job, "worker_started")
    op, dtype_str = job["operator"], job["dtype"]
    dtype = resolve_dtype(dtype_str)
    gen_cpu = get_generator(op)
    rules = job.get("rules", {}) or {}
    cases = [dict(c, _position=i) for i, c in enumerate(job["cases"])]
    try:
        from tilebench.paths import operator_dir
        ref_mod = W._load_module(operator_dir(op) / "impl_torch.py")
        mutable = W.mutable_indices(rules, ref_mod.run)
    except Exception:  # noqa: BLE001
        result["diagnostic"] = "reference import failure:\n" + traceback.format_exc()
        return result

    def gen_device(**kw):
        inp = gen_cpu(**kw)
        return rt.to_device(inp if isinstance(inp, tuple) else (inp,))

    def ref_cpu(*inputs):
        return ref_mod.run(*rt.to_cpu(tuple(inputs)))

    guard = W.PrecisionGuard()
    result["precision_guard"] = guard.record()
    ref_run = guard.guard_reference(ref_cpu)

    def stop_global(status: str, diagnostic: str) -> dict:
        result.update(status=status, diagnostic=diagnostic)
        recs = [{"case_id": c["case_id"], "case_index": c.get("case_index"), "params": c["params"],
                 "status": "not_evaluated", "diagnostic": f"suite stopped before this case ({status})"} for c in cases]
        result["suite_stopped"] = {"reason": status, "at_case": None}
        W.summarize_suite(result, recs, len(cases))
        result.update(status=status, diagnostic=diagnostic)
        result["precision_guard"] = guard.record()
        result["elapsed_s"] = time.time() - t0
        return result

    stage = {"started": time.time()}
    W._progress(job, "candidate_loaded")
    try:
        cand = W._load_module(Path(job["source_path"]))
        guard.verify("after importing the candidate module")
        stage.update(ok=True, elapsed_s=time.time() - stage["started"])
        result["stages"]["import_compile"] = stage
    except W.PrecisionTampered as e:
        return stop_global("contract_violation", f"contract_violation: {e}")
    except Exception:  # noqa: BLE001
        result["stages"]["import_compile"] = {**stage, "ok": False, "phase": "import"}
        return stop_global("compile_error", traceback.format_exc()[-8000:])
    if not callable(getattr(cand, "run", None)):
        result["stages"]["interface"] = {"run": False}
        return stop_global("interface_error", "interface_error: the generated module defines no callable run()")
    if not callable(getattr(cand, "get_last_config", None)):
        result["stages"]["interface"] = {"run": True, "get_last_config": False}
        return stop_global("interface_error", "interface_error: the generated module defines no callable get_last_config()")
    result["stages"]["interface"] = {"run": True, "get_last_config": True}
    hits = W.autotuner_instances(cand, job["dsl"]) + _nki_autotuner_hits(cand)
    result["stages"]["autotuner_scan"] = {"hits": hits}
    if hits:
        return stop_global("contract_violation", "contract_violation: autotuner object(s) found in the generated module: "
                           + "; ".join(hits))
    fallback_log: list = []
    cand_run = guard.guard_candidate(track_host_fallback(cand.run, torch_neuronx, fallback_log))
    torch_view = _TorchView(torch, rt.sync)
    measure = functools.partial(measure_neuron, rt=rt)
    checks = run_numerical_checks_neuron
    result["stages"]["device"] = {"device": str(rt.device), "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
                                  "num_cores": os.environ.get("NEURON_RT_NUM_CORES"),
                                  "reference": "impl_torch.run on CPU copies of the device inputs",
                                  "torch_compile_disabled": True}

    records: list[dict] = []
    with neuron_native.torch_compile_disabled():
        for pos, case in enumerate(cases):
            try:
                rec = W.run_case(job, case, cand=cand, cand_run=cand_run, ref_run=ref_run, guard=guard, mutable=mutable,
                                 rules=rules, torch=torch_view, gen=gen_device, dtype=dtype, measure=measure,
                                 run_numerical_checks=checks)
            except W.PrecisionRestoreFailed as e:
                result["precision_guard"] = guard.record()
                result.update(status="infrastructure_incomplete", diagnostic=f"evaluator precision restore failed: {e}")
                break
            rec["host_fallback_probe"] = fallback_log[:2]
            fallback_log.clear()
            records.append(rec)
            W._append_case(job, rec)
            gc.collect()
            try:
                torch_neuronx.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            if rec["status"] in W.SUITE_STOP_STATUSES:
                result["suite_stopped"] = {"reason": rec["status"], "at_case": rec["case_id"], "position": pos}
                for rest in cases[pos + 1:]:
                    records.append({"case_id": rest["case_id"], "case_index": rest.get("case_index"), "params": rest["params"],
                                    "status": "not_evaluated",
                                    "diagnostic": f"suite stopped at case {pos + 1} ({rec['status']})"})
                break
    if result.get("status") == "infrastructure_incomplete" and result.get("diagnostic"):
        W.summarize_suite(result, records, len(cases))
        result.update(status="infrastructure_incomplete")
    else:
        result["status"] = None
        W.summarize_suite(result, records, len(cases))
        if result["status"] is None:
            result["status"] = "runtime_error"
    result["precision_guard"] = guard.record()
    W._progress(job, "suite_done")
    result["elapsed_s"] = time.time() - t0
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    job = json.loads(Path(a.job).read_text())
    if os.environ.get("TILEBENCH_V2_SANDBOX") != "1":
        print("worker must be started by the launcher (TILEBENCH_V2_SANDBOX=1)", file=sys.stderr)
        return 2
    res = run_job(job)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
