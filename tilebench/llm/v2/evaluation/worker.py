"""Isolated evaluation worker (subprocess entry point).

    python -m tilebench.llm.v2.evaluation.worker --job job.json --out result.json

job.json is `EvaluationJob.worker_job(...)` (evaluation.job): operator, dtype,
params, dsl, source_path, sandbox_dir, atol, rtol, rules (evaluator rules of
the contract), timing {warmup, repeat, use_cuda_graph, flush,
capture_failure_policy, record_prep_runs}, expected_timing_mode, seed,
identity.

What the worker enforces (and nothing more):
- interface: the module defines a callable `run` and a callable
  `get_last_config` that returns a JSON-serializable dict; a missing or
  broken export is `interface_error` (an ordinary failure: the round is
  consumed, no repair). `get_last_config()` is read after the first
  execution, after the numerical checks and after timing; a value that
  changes between reads is an `interface_error` too (the configuration is
  not fixed), reported with both values;
- execution-side autotuner detection: a `triton.runtime.autotuner.Autotuner`
  (or TileLang AutoTuner) instance reachable from the module namespace is a
  `contract_violation` regardless of how it was imported;
- three numerical checks on fresh / same-address-refilled / repeated inputs
  against the operator's manual impl_torch reference, with frozen oracle
  outputs; undeclared input mutation and output aliasing fail the check;
- timing on one fixed input set: 1 warmup + 3 timed launches, restore hook
  for declared mutable inputs before every launch including capture
  preparation; three raw samples and their mean; a failed graph capture is
  recorded, timed eagerly and flagged (study capture_failure_policy).

Isolation is the launcher's job (evaluation.launcher): this process only
refuses to start without TILEBENCH_V2_SANDBOX=1. Reading get_last_config()
does not prove the absence of a hidden search; the static evidence, the
autotuner scan and the config-stability reads are the evidence recorded."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

RESULT_STATUSES = ("valid", "interface_error", "compile_error", "runtime_error", "numerical_error", "timing_error",
                   "contract_violation", "infrastructure_incomplete")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_config(mod) -> tuple[dict | None, str | None]:
    fn = getattr(mod, "get_last_config", None)
    if not callable(fn):
        return None, "get_last_config() is not defined"
    try:
        value = fn()
    except Exception:  # noqa: BLE001
        return None, "get_last_config() raised:\n" + traceback.format_exc()[-2000:]
    if not isinstance(value, dict):
        return None, f"get_last_config() returned {type(value).__name__}, not a dict"
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        return None, f"get_last_config() returned a dict that is not JSON-serializable: {e}"
    return value, None


def autotuner_instances(mod, dsl: str) -> list[str]:
    """Execution-side detection of autotuner objects reachable from the module."""
    hits: list[str] = []
    classes = []
    if dsl == "triton":
        try:
            from triton.runtime.autotuner import Autotuner
            classes.append(("triton.runtime.autotuner.Autotuner", Autotuner))
        except Exception:  # noqa: BLE001
            pass
    elif dsl == "tilelang":
        try:
            from tilelang.autotuner import AutoTuner
            classes.append(("tilelang.autotuner.AutoTuner", AutoTuner))
        except Exception:  # noqa: BLE001
            pass
    if not classes:
        return hits
    for name, obj in list(vars(mod).items()):
        cur, depth = obj, 0
        while cur is not None and depth < 6:
            for label, cls in classes:
                if isinstance(cur, cls):
                    hits.append(f"{name}: {label} instance")
                    cur = None
                    break
            if cur is None:
                break
            cur = getattr(cur, "fn", None)
            depth += 1
    return hits


def mutable_indices(rules: dict, ref_run) -> set[int]:
    names = (rules.get("mutation", {}) or {}).get("inputs_mutated", []) or []
    if not names:
        return set()
    params = list(inspect.signature(ref_run).parameters)
    out: set[int] = set()
    for n in names:
        if isinstance(n, int):
            out.add(n)
        elif n in params:
            out.add(params.index(n))
        else:
            raise ValueError(f"evaluator rules name an unknown input {n!r} (reference parameters {params})")
    return out


def aliasing_allowed(rules: dict) -> bool:
    text = str((rules.get("outputs", {}) or {}).get("aliasing", "none")).strip().lower()
    return not text.startswith("none")


PHASES = ("worker_started", "candidate_loaded", "first_execution_done", "numerical_checks_done", "timing_started",
          "timing_done")


def _progress(job: dict, phase: str) -> None:
    """Phase marker read by the launcher when the worker never returns
    (timeout, crash): a hang after `candidate_loaded` is the candidate's
    failure, a hang before it is the infrastructure's."""
    try:
        d = job.get("sandbox_dir")
        if d:
            Path(d, "progress.json").write_text(json.dumps({"phase": phase, "t": time.time()}) + "\n")
    except OSError:
        pass


def run_job(job: dict) -> dict:
    t0 = time.time()
    result: dict = {"schema": "tilebench-llm-v2-eval/2", "status": "infrastructure_incomplete",
                    "stages": {}, "diagnostic": None, "config": None, "latency_ms_mean": None,
                    "latency_ms_samples": None, "timing": None, "identity": job.get("identity"),
                    "expected_timing_mode": job.get("expected_timing_mode"), "timing_mode_differs": None}
    try:
        from tilebench.llm.v2.evaluation.job import validate_worker_job
        errs = validate_worker_job(job)
        if errs:
            result["diagnostic"] = "invalid job: " + "; ".join(errs)
            return result
        import torch
        from tilebench.core.dtypes import resolve_dtype
        from tilebench.data.tensors import get_generator
        from tilebench.llm.v2.evaluation.anticache import run_numerical_checks
        from tilebench.llm.v2.evaluation.timing import measure
    except Exception:  # noqa: BLE001
        result["diagnostic"] = "worker import failure:\n" + traceback.format_exc()
        return result
    if not torch.cuda.is_available():
        result["diagnostic"] = "no CUDA/HIP device visible to the worker"
        return result
    _progress(job, "worker_started")
    sync = torch.cuda.synchronize
    op, dtype_str, params = job["operator"], job["dtype"], job["params"]
    dtype = resolve_dtype(dtype_str)
    gen = get_generator(op)
    seed = int(job.get("seed", 0))
    rules = job.get("rules", {}) or {}

    def make_inputs():
        nonlocal seed
        torch.manual_seed(seed)
        seed += 1
        inp = gen(**params, dtype=dtype)
        return inp if isinstance(inp, tuple) else (inp,)

    # reference (functional semantics) comes from the manual impl_torch of the operator
    try:
        from tilebench.paths import operator_dir
        ref_mod = _load_module(operator_dir(op) / "impl_torch.py")
        mutable = mutable_indices(rules, ref_mod.run)
    except Exception:  # noqa: BLE001
        result["diagnostic"] = "reference import failure:\n" + traceback.format_exc()
        return result

    # 1. import / compile (first execution on fresh inputs)
    stage = {"started": time.time()}
    _progress(job, "candidate_loaded")        # from here on, a hang is the candidate's
    try:
        cand = _load_module(Path(job["source_path"]))
    except Exception:  # noqa: BLE001
        result["status"] = "compile_error"
        result["diagnostic"] = traceback.format_exc()[-8000:]
        result["stages"]["import_compile"] = {**stage, "ok": False, "phase": "import"}
        return result
    # interface: required exports
    if not callable(getattr(cand, "run", None)):
        result["status"] = "interface_error"
        result["diagnostic"] = "interface_error: the generated module defines no callable run()"
        result["stages"]["interface"] = {"run": False}
        return result
    if not callable(getattr(cand, "get_last_config", None)):
        result["status"] = "interface_error"
        result["diagnostic"] = "interface_error: the generated module defines no callable get_last_config()"
        result["stages"]["interface"] = {"run": True, "get_last_config": False}
        return result
    # execution-side autotuner scan (module namespace)
    hits = autotuner_instances(cand, job["dsl"])
    result["stages"]["autotuner_scan"] = {"hits": hits}
    if hits:
        result["status"] = "contract_violation"
        result["diagnostic"] = "contract_violation: autotuner object(s) found in the generated module: " + "; ".join(hits)
        return result
    try:
        inputs = make_inputs()
        cand.run(*inputs)
        sync()
        stage["ok"] = True
        stage["executions"] = 1
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        result["status"] = "compile_error" if ("compil" in tb.lower() or "CompilationError" in tb) else "runtime_error"
        result["diagnostic"] = tb[-8000:]
        result["stages"]["import_compile"] = {**stage, "ok": False, "phase": "first_execution"}
        return result
    result["stages"]["import_compile"] = stage
    _progress(job, "first_execution_done")
    cfg1, cfg_err = read_config(cand)
    if cfg_err:
        result["status"] = "interface_error"
        result["diagnostic"] = "interface_error: " + cfg_err
        result["stages"]["interface"] = {"run": True, "get_last_config": False, "error": cfg_err}
        return result
    result["stages"]["interface"] = {"run": True, "get_last_config": True}
    result["config"] = cfg1

    # 2. numerical checks on fresh inputs / same-address refill / repeat
    try:
        checks = run_numerical_checks(cand.run, ref_mod.run, make_inputs, atol=job["atol"], rtol=job["rtol"],
                                      mutable_indices=mutable, sync=sync, aliasing_allowed=aliasing_allowed(rules))
    except Exception:  # noqa: BLE001
        result["status"] = "runtime_error"
        result["diagnostic"] = traceback.format_exc()[-8000:]
        return result
    result["stages"]["numerical_checks"] = {"executions_candidate": len(checks), "executions_reference": len(checks),
                                            "checks": [c.to_dict() for c in checks], "mutable_indices": sorted(mutable)}
    _progress(job, "numerical_checks_done")
    failed = [c for c in checks if not c.ok]
    if failed:
        if any(c.output_aliases_inputs for c in failed):
            result["status"] = "contract_violation"
            result["diagnostic"] = "contract_violation: " + "\n".join(f"{c.name}: {c.message}" for c in failed)[-8000:]
        else:
            result["status"] = "numerical_error"
            result["diagnostic"] = "\n".join(f"{c.name}: {c.message}" for c in failed)[-8000:]
        return result
    cfg2, cfg_err = read_config(cand)
    if cfg_err or cfg2 != cfg1:
        result["status"] = "interface_error"
        result["diagnostic"] = ("interface_error: " + cfg_err) if cfg_err else \
            f"interface_error: get_last_config() changed between calls on the same task: first {json.dumps(cfg1)}, later {json.dumps(cfg2)}; the configuration must be fixed"
        result["stages"]["config_stability"] = {"reads": 2, "stable": False}
        return result

    # 3. timing on fixed inputs: 1 warmup + 3 timed; restore mutated inputs outside the scope
    _progress(job, "timing_started")
    try:
        timed_inputs = make_inputs()
        snap = [x.clone() if isinstance(x, torch.Tensor) else x for x in timed_inputs]

        def before_launch():
            for i in mutable:
                x = timed_inputs[i]
                if isinstance(x, torch.Tensor):
                    x.copy_(snap[i])

        t = job.get("timing", {})
        rec = measure(lambda: cand.run(*timed_inputs), warmup=int(t.get("warmup", 1)), repeat=int(t.get("repeat", 3)),
                      use_cuda_graph=bool(t.get("use_cuda_graph", True)), flush=bool(t.get("flush", True)),
                      before_launch=before_launch if mutable else None,
                      proton_output_dir=job.get("sandbox_dir"), keep_profile=True)
        result["timing"] = rec.to_dict()
        result["stages"]["timing"] = {"warmup": rec.warmup_runs, "graph_prep": rec.graph_prep_runs, "timed": rec.timed_runs}
        expected = job.get("expected_timing_mode")
        result["timing_mode_differs"] = bool(expected) and rec.timing_execution_mode != expected
        if rec.mean_ms is None:
            result["status"] = "timing_error"
            result["diagnostic"] = rec.note or "timing produced no positive sample"
            return result
        cfg3, cfg_err = read_config(cand)
        if cfg_err or cfg3 != cfg1:
            result["status"] = "interface_error"
            result["diagnostic"] = ("interface_error: " + cfg_err) if cfg_err else \
                f"interface_error: get_last_config() changed after timing: first {json.dumps(cfg1)}, later {json.dumps(cfg3)}; the configuration must be fixed"
            result["stages"]["config_stability"] = {"reads": 3, "stable": False}
            return result
        result["stages"]["config_stability"] = {"reads": 3, "stable": True}
        result["latency_ms_mean"] = rec.mean_ms
        result["latency_ms_samples"] = rec.samples_ms
        result["status"] = "valid"
    except Exception:  # noqa: BLE001
        result["status"] = "timing_error"
        result["diagnostic"] = traceback.format_exc()[-8000:]
    _progress(job, "timing_done")
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
