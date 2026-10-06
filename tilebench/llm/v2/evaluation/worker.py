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
    """An independent canonical snapshot of get_last_config(): the returned
    object is round-tripped through JSON (sorted keys), so a later in-place
    mutation of the module's dict (nested levels included) cannot change an
    earlier read, and the three reads are compared value by value."""
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
        snapshot = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as e:
        return None, f"get_last_config() returned a dict that is not JSON-serializable: {e}"
    return snapshot, None


def config_snapshots_equal(a: dict | None, b: dict | None) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


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


class PrecisionTampered(RuntimeError):
    def __init__(self, where: str, diffs: dict):
        super().__init__(f"process-wide precision/backend state changed {where}: {diffs}")
        self.where, self.diffs = where, diffs


def precision_state() -> dict:
    """The process-wide PyTorch numerical settings a candidate must leave
    alone: they decide how the reference and later candidates compute. The
    legacy matmul getter is read first; after a candidate switches to the
    fp32_precision API, reading it raises, which is itself evidence."""
    import torch
    m = torch.backends.cuda.matmul
    st: dict = {}

    def get(name, fn):
        try:
            v = fn()
            st[name] = v if isinstance(v, (bool, int, float, str, type(None))) else str(v)
        except Exception as e:  # noqa: BLE001
            st[name] = f"<unreadable: {type(e).__name__}>"
    get("cuda.matmul.allow_tf32", lambda: m.allow_tf32)
    get("cuda.matmul.fp32_precision", lambda: getattr(m, "fp32_precision", None))
    get("cuda.matmul.allow_fp16_reduced_precision_reduction", lambda: m.allow_fp16_reduced_precision_reduction)
    get("cuda.matmul.allow_bf16_reduced_precision_reduction", lambda: m.allow_bf16_reduced_precision_reduction)
    get("cudnn.allow_tf32", lambda: torch.backends.cudnn.allow_tf32)
    get("cudnn.fp32_precision", lambda: getattr(torch.backends.cudnn, "fp32_precision", None))
    get("cudnn.benchmark", lambda: torch.backends.cudnn.benchmark)
    get("cudnn.deterministic", lambda: torch.backends.cudnn.deterministic)
    get("float32_matmul_precision", torch.get_float32_matmul_precision)
    get("default_dtype", lambda: str(torch.get_default_dtype()))
    get("deterministic_algorithms", torch.are_deterministic_algorithms_enabled)
    return st


def precision_diff(baseline: dict, now: dict) -> dict:
    return {k: (baseline.get(k), now.get(k)) for k in baseline if baseline.get(k) != now.get(k)}


_RESTORE = {
    "cuda.matmul.allow_tf32": lambda t, v: setattr(t.backends.cuda.matmul, "allow_tf32", v),
    "cuda.matmul.allow_fp16_reduced_precision_reduction": lambda t, v: setattr(t.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", v),
    "cuda.matmul.allow_bf16_reduced_precision_reduction": lambda t, v: setattr(t.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", v),
    "cudnn.allow_tf32": lambda t, v: setattr(t.backends.cudnn, "allow_tf32", v),
    "cudnn.benchmark": lambda t, v: setattr(t.backends.cudnn, "benchmark", v),
    "cudnn.deterministic": lambda t, v: setattr(t.backends.cudnn, "deterministic", v),
    "deterministic_algorithms": lambda t, v: t.use_deterministic_algorithms(bool(v)),
}


def restore_precision(baseline: dict) -> dict:
    """Set every restorable field back to the baseline where it differs;
    returns the fields that were restored."""
    import torch
    now = precision_state()
    restored: dict = {}
    for k, v in baseline.items():
        if now.get(k) != v and k in _RESTORE and not (isinstance(v, str) and v.startswith("<unreadable")):
            try:
                _RESTORE[k](torch, v)
                restored[k] = (now.get(k), v)
            except Exception as e:  # noqa: BLE001
                restored[k] = (now.get(k), f"<restore failed: {type(e).__name__}>")
    return restored


class PrecisionGuard:
    """Snapshot before the candidate is loaded; restore before every reference
    call and before every candidate call; verify after every candidate call
    (and after the module import). A difference after the candidate ran is a
    contract violation: the candidate changed evaluator state."""

    def __init__(self):
        self.baseline = precision_state()
        self.restores: list[dict] = []
        self.verifications = 0

    def restore(self, where: str) -> None:
        r = restore_precision(self.baseline)
        if r:
            self.restores.append({"where": where, "restored": r})

    def verify(self, where: str) -> None:
        self.verifications += 1
        diffs = precision_diff(self.baseline, precision_state())
        if diffs:
            raise PrecisionTampered(where, diffs)

    def guard_candidate(self, fn):
        def run(*a, **k):
            self.restore("before candidate call")
            out = fn(*a, **k)
            self.verify("after candidate call")
            return out
        return run

    def guard_reference(self, fn):
        def run(*a, **k):
            self.restore("before reference call")
            return fn(*a, **k)
        return run

    def record(self) -> dict:
        return {"baseline": self.baseline, "verifications": self.verifications, "restores": self.restores}


def capture_failure_status(policy: str | None, expected_mode: str | None, capture_succeeded) -> str | None:
    """Policy timing_error (formal, frozen 2026-10-06): a failed CUDA-graph
    capture on a device whose adapter replays graphs makes the round a
    timing_error (consumed; the eager sample is never accepted).
    time_eagerly_and_flag: the eager sample stands and is flagged."""
    if policy == "timing_error" and expected_mode == "graph" and capture_succeeded is False:
        return "timing_error"
    return None


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
    guard = PrecisionGuard()                   # process-wide precision flags are evaluator state
    result["precision_guard"] = guard.record()
    ref_run = guard.guard_reference(ref_mod.run)

    # 1. import / compile (first execution on fresh inputs)
    stage = {"started": time.time()}
    _progress(job, "candidate_loaded")        # from here on, a hang is the candidate's
    try:
        cand = _load_module(Path(job["source_path"]))
        guard.verify("after importing the candidate module")
    except PrecisionTampered as e:
        result["status"] = "contract_violation"
        result["diagnostic"] = f"contract_violation: {e}"
        result["precision_guard"] = guard.record()
        return result
    except Exception:  # noqa: BLE001
        result["status"] = "compile_error"
        result["diagnostic"] = traceback.format_exc()[-8000:]
        result["stages"]["import_compile"] = {**stage, "ok": False, "phase": "import"}
        return result
    cand_run = guard.guard_candidate(cand.run)
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
        cand_run(*inputs)
        sync()
        stage["ok"] = True
        stage["executions"] = 1
    except PrecisionTampered as e:
        result["status"] = "contract_violation"
        result["diagnostic"] = f"contract_violation: {e}"
        result["precision_guard"] = guard.record()
        return result
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
    result["config_reads"] = [{"after": "first_execution", "config": cfg1}]

    # 2. numerical checks on fresh inputs / same-address refill / repeat
    try:
        checks = run_numerical_checks(cand_run, ref_run, make_inputs, atol=job["atol"], rtol=job["rtol"],
                                      mutable_indices=mutable, sync=sync, aliasing_allowed=aliasing_allowed(rules))
    except PrecisionTampered as e:
        result["status"] = "contract_violation"
        result["diagnostic"] = f"contract_violation: {e}"
        result["precision_guard"] = guard.record()
        return result
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
    result["config_reads"].append({"after": "numerical_checks", "config": cfg2, "error": cfg_err})
    if cfg_err or not config_snapshots_equal(cfg2, cfg1):
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
        rec = measure(lambda: cand_run(*timed_inputs), warmup=int(t.get("warmup", 1)), repeat=int(t.get("repeat", 3)),
                      use_cuda_graph=bool(t.get("use_cuda_graph", True)), flush=bool(t.get("flush", True)),
                      before_launch=before_launch if mutable else None,
                      proton_output_dir=job.get("sandbox_dir"), keep_profile=True)
        result["timing"] = rec.to_dict()
        result["stages"]["timing"] = {"warmup": rec.warmup_runs, "graph_prep": rec.graph_prep_runs, "timed": rec.timed_runs}
        expected = job.get("expected_timing_mode")
        result["timing_mode_differs"] = bool(expected) and rec.timing_execution_mode != expected
        cf = capture_failure_status(t.get("capture_failure_policy"), expected, rec.capture_succeeded)
        if cf:
            result["status"] = cf
            result["diagnostic"] = ("timing_error: CUDA-graph capture failed and the campaign's capture_failure_policy is "
                                    f"'timing_error' (round consumed; no eager sample is accepted): {rec.capture_error}")
            result["precision_guard"] = guard.record()
            return result
        if rec.mean_ms is None:
            result["status"] = "timing_error"
            result["diagnostic"] = rec.note or "timing produced no positive sample"
            return result
        cfg3, cfg_err = read_config(cand)
        result["config_reads"].append({"after": "timing", "config": cfg3, "error": cfg_err})
        if cfg_err or not config_snapshots_equal(cfg3, cfg1):
            result["status"] = "interface_error"
            result["diagnostic"] = ("interface_error: " + cfg_err) if cfg_err else \
                f"interface_error: get_last_config() changed after timing: first {json.dumps(cfg1)}, later {json.dumps(cfg3)}; the configuration must be fixed"
            result["stages"]["config_stability"] = {"reads": 3, "stable": False}
            return result
        result["stages"]["config_stability"] = {"reads": 3, "stable": True}
        result["latency_ms_mean"] = rec.mean_ms
        result["latency_ms_samples"] = rec.samples_ms
        result["status"] = "valid"
    except PrecisionTampered as e:
        result["status"] = "contract_violation"
        result["diagnostic"] = f"contract_violation: {e}"
    except Exception:  # noqa: BLE001
        result["status"] = "timing_error"
        result["diagnostic"] = traceback.format_exc()[-8000:]
    result["precision_guard"] = guard.record()
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
