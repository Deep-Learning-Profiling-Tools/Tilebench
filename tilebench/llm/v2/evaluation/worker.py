"""Isolated evaluation worker (subprocess entry point).

    python -m tilebench.llm.v2.evaluation.worker --job job.json --out result.json

job.json is `EvaluationJob.worker_job(...)` (evaluation.job): operator, dtype,
cases (the task's frozen case suite, protocol revision 4), dsl, source_path,
sandbox_dir, atol, rtol, rules (evaluator rules of the contract), timing
{warmup, repeat, use_cuda_graph, flush, capture_failure_policy,
record_prep_runs}, expected_timing_mode, seed, identity.

One candidate source is loaded ONCE; then every case of the suite is
evaluated in order, each on its own freshly generated inputs:

- interface (global): the module defines a callable `run` and a callable
  `get_last_config` that returns a JSON-serializable dict; a missing export
  is `interface_error` for the whole suite;
- execution-side autotuner detection (global): a
  `triton.runtime.autotuner.Autotuner` (or TileLang AutoTuner) instance
  reachable from the module namespace is a `contract_violation`;
- per case: first execution (compile/specialize), `get_last_config()` read
  after the first execution, after the numerical checks and after timing
  (a value that changes within one case is that case's `interface_error`;
  different cases may report different, deterministic configurations);
  three numerical checks on fresh / same-address-refilled / repeated inputs
  against the operator's manual impl_torch reference with frozen oracle
  outputs (undeclared mutation and output aliasing fail the check); timing on
  one fixed input set: 1 warmup + 3 timed launches with the input-restore
  hook; three raw samples and their mean.

Suite rules (study.yaml evaluation / task_unit): a round is `valid` only when
every case is valid; its latency is the geometric mean of the case means. A
`contract_violation` in any case, or a compile failure of a case, stops the
suite (the remaining cases are recorded as not evaluated, never fabricated);
other per-case failures are recorded and the next case is evaluated. The
round status of an invalid suite is the status of its first failing case.
Every finished case is appended to `cases.jsonl` in the sandbox before the
next one starts, so a timeout or a crash keeps the cases already obtained.

Process-wide PyTorch precision state is evaluator state (PrecisionGuard):
one canonical API family is snapshotted before the candidate is loaded,
restored before every reference and candidate call and verified after every
candidate call; only a change the candidate itself made is a violation.

Isolation is the launcher's job (evaluation.launcher): this process only
refuses to start without TILEBENCH_V2_SANDBOX=1."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

RESULT_STATUSES = ("valid", "interface_error", "compile_error", "runtime_error", "numerical_error", "timing_error",
                   "contract_violation", "infrastructure_incomplete")
CASE_STATUSES = RESULT_STATUSES[:-1] + ("not_evaluated",)
# a case with one of these statuses stops the suite (study.yaml evaluation.stop_suite_on); the remaining cases
# are `not_evaluated`. A module-level failure (import, interface, autotuner object) and a dead worker stop it too.
SUITE_STOP_STATUSES = ("contract_violation", "compile_error")
RESULT_SCHEMA = "tilebench-llm-v2-eval/3"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_config(mod) -> tuple[dict | None, str | None]:
    """An independent canonical snapshot of get_last_config(): the returned
    object is round-tripped through JSON (sorted keys), so a later in-place
    mutation of the module's dict (nested levels included) cannot change an
    earlier read, and the reads are compared value by value."""
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


def _progress(job: dict, phase: str, case_index: int | None = None) -> None:
    """Phase marker read by the launcher when the worker never returns
    (timeout, crash): a hang after `candidate_loaded` is the candidate's
    failure, a hang before it is the infrastructure's."""
    try:
        d = job.get("sandbox_dir")
        if d:
            Path(d, "progress.json").write_text(json.dumps({"phase": phase, "case": case_index, "t": time.time()}) + "\n")
    except OSError:
        pass


def _append_case(job: dict, rec: dict) -> None:
    try:
        d = job.get("sandbox_dir")
        if d:
            with open(Path(d, "cases.jsonl"), "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# process-wide precision state (evaluator state)
# --------------------------------------------------------------------------

class PrecisionTampered(RuntimeError):
    def __init__(self, where: str, diffs: dict):
        super().__init__(f"process-wide precision/backend state changed {where}: {diffs}")
        self.where, self.diffs = where, diffs


class PrecisionRestoreFailed(RuntimeError):
    """The evaluator could not bring the canonical state back (an evaluator problem, never the candidate's)."""


def precision_api_family() -> str:
    """`fp32_precision` when the installed PyTorch has the per-backend
    precision API (2.9+), else `legacy` (allow_tf32 / float32_matmul_precision).
    In the new family the legacy fields are derived VIEWS that writing the new
    fields changes and that can even raise when read in a mixed state; they are
    never snapshotted, compared or restored as separate state."""
    import torch
    m = torch.backends.cuda.matmul
    try:
        return "fp32_precision" if isinstance(getattr(m, "fp32_precision", None), str) else "legacy"
    except Exception:  # noqa: BLE001
        return "legacy"


def _canonical_fields(torch) -> dict:
    """name -> (getter, setter) of the canonical family's fields."""
    m, c = torch.backends.cuda.matmul, torch.backends.cudnn
    common = {
        "cuda.matmul.allow_fp16_reduced_precision_reduction": (lambda: m.allow_fp16_reduced_precision_reduction,
                                                               lambda v: setattr(m, "allow_fp16_reduced_precision_reduction", v)),
        "cuda.matmul.allow_bf16_reduced_precision_reduction": (lambda: m.allow_bf16_reduced_precision_reduction,
                                                               lambda v: setattr(m, "allow_bf16_reduced_precision_reduction", v)),
        "cudnn.benchmark": (lambda: c.benchmark, lambda v: setattr(c, "benchmark", v)),
        "cudnn.deterministic": (lambda: c.deterministic, lambda v: setattr(c, "deterministic", v)),
        "default_dtype": (lambda: str(torch.get_default_dtype()),
                          lambda v: torch.set_default_dtype(getattr(torch, str(v).replace("torch.", "")))),
        "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled,
                                     lambda v: torch.use_deterministic_algorithms(bool(v))),
    }
    if precision_api_family() == "fp32_precision":
        fam = {
            "backends.fp32_precision": (lambda: torch.backends.fp32_precision,
                                        lambda v: setattr(torch.backends, "fp32_precision", v)),
            "cuda.matmul.fp32_precision": (lambda: m.fp32_precision, lambda v: setattr(m, "fp32_precision", v)),
            "cudnn.fp32_precision": (lambda: c.fp32_precision, lambda v: setattr(c, "fp32_precision", v)),
            "cudnn.conv.fp32_precision": (lambda: c.conv.fp32_precision, lambda v: setattr(c.conv, "fp32_precision", v)),
            "cudnn.rnn.fp32_precision": (lambda: c.rnn.fp32_precision, lambda v: setattr(c.rnn, "fp32_precision", v)),
        }
    else:
        fam = {
            "float32_matmul_precision": (torch.get_float32_matmul_precision, torch.set_float32_matmul_precision),
            "cuda.matmul.allow_tf32": (lambda: m.allow_tf32, lambda v: setattr(m, "allow_tf32", v)),
            "cudnn.allow_tf32": (lambda: c.allow_tf32, lambda v: setattr(c, "allow_tf32", v)),
        }
    return {**fam, **common}


def _quiet(fn, *a):
    """Run a precision getter/setter without PyTorch's API-mixing UserWarnings (evaluator bookkeeping only)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a)


def _read(getter):
    try:
        v = _quiet(getter)
        return v if isinstance(v, (bool, int, float, str, type(None))) else str(v)
    except Exception as e:  # noqa: BLE001
        return f"<unreadable: {type(e).__name__}>"


def precision_state() -> dict:
    """Snapshot of the canonical precision family (see precision_api_family)."""
    import torch
    return {k: _read(g) for k, (g, _) in _canonical_fields(torch).items()}


def _legacy_global_matmul_precision():
    import torch
    try:
        return _quiet(torch.get_float32_matmul_precision)
    except Exception:  # noqa: BLE001
        return None


def precision_diff(baseline: dict, now: dict) -> dict:
    return {k: (baseline.get(k), now.get(k)) for k in baseline if baseline.get(k) != now.get(k)}


def restore_precision(baseline: dict, legacy_global: str | None = None) -> dict:
    """Bring the canonical state back to `baseline`; returns the fields that
    differed. In the fp32_precision family the stored legacy global
    float32_matmul_precision is restored FIRST (legacy setters write the new
    fields), then every canonical field is set, so the new fields end exactly
    at the baseline. Raises PrecisionRestoreFailed if they do not."""
    import torch
    fields = _canonical_fields(torch)
    before = precision_state()
    diffs = precision_diff(baseline, before)
    if precision_api_family() == "fp32_precision" and legacy_global is not None and \
            _legacy_global_matmul_precision() != legacy_global:
        try:
            _quiet(torch.set_float32_matmul_precision, legacy_global)
            diffs.setdefault("float32_matmul_precision(legacy view)", ("changed", legacy_global))
        except Exception:  # noqa: BLE001
            pass
    if not diffs:
        return {}
    for k, (_, setter) in fields.items():
        v = baseline.get(k)
        if isinstance(v, str) and v.startswith("<unreadable"):
            continue
        if _read(fields[k][0]) != v:
            try:
                _quiet(setter, v)
            except Exception as e:  # noqa: BLE001
                raise PrecisionRestoreFailed(f"cannot restore {k} to {v!r}: {type(e).__name__}: {e}") from e
    after = precision_diff(baseline, precision_state())
    if after:
        raise PrecisionRestoreFailed(f"canonical precision state not restored: {after}")
    return diffs


class PrecisionGuard:
    """Snapshot of the canonical precision family before the candidate is
    loaded (after the reference module is imported: a reference module-level
    setting is part of the frozen evaluator state of that operator); restore
    before every reference call and before every candidate call; verify after
    every candidate call and after the module import. Only a difference the
    candidate itself produced (state restored -> candidate call -> state
    differs) is a contract violation."""

    def __init__(self):
        self.family = precision_api_family()
        self.baseline = precision_state()
        self.legacy_global = _legacy_global_matmul_precision()
        self.restores: list[dict] = []
        self.verifications = 0

    def restore(self, where: str) -> None:
        r = restore_precision(self.baseline, self.legacy_global)
        if r:
            self.restores.append({"where": where, "restored": {k: list(v) for k, v in r.items()}})

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
        return {"family": self.family, "baseline": self.baseline, "legacy_global_matmul_precision": self.legacy_global,
                "verifications": self.verifications, "restores": self.restores[-50:], "restore_count": len(self.restores)}


def capture_failure_status(policy: str | None, expected_mode: str | None, capture_succeeded) -> str | None:
    """Policy timing_error (formal, frozen 2026-10-06): a failed CUDA-graph
    capture on a device whose adapter replays graphs makes the case a
    timing_error (the eager sample is never accepted).
    time_eagerly_and_flag: the eager sample stands and is flagged."""
    if policy == "timing_error" and expected_mode == "graph" and capture_succeeded is False:
        return "timing_error"
    return None


def geometric_mean(values: list[float]) -> float | None:
    if not values or any((not isinstance(v, (int, float))) or not math.isfinite(v) or v <= 0 for v in values):
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _classify_exception(tb: str) -> str:
    return "compile_error" if ("compil" in tb.lower() or "CompilationError" in tb) else "runtime_error"


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------

def run_case(job: dict, case: dict, *, cand, cand_run, ref_run, guard: PrecisionGuard, mutable: set[int],
             rules: dict, torch, gen, dtype, measure, run_numerical_checks) -> dict:
    """Evaluate one case; returns its record (status in CASE_STATUSES)."""
    t_case = time.time()
    idx = int(case.get("case_index", 0))
    params = case["params"]
    rec: dict = {"case_id": case["case_id"], "case_index": idx, "params": params, "problem_size": case.get("problem_size"),
                 "status": "runtime_error", "diagnostic": None, "config": None, "config_reads": [],
                 "latency_ms_mean": None, "latency_ms_samples": None, "timing": None, "timing_mode_differs": None,
                 "stages": {}}
    seed = int(job.get("seed", 0)) + 1000 * (int(case.get("_position", 0)) + 1)
    sync = torch.cuda.synchronize

    def make_inputs():
        nonlocal seed
        torch.manual_seed(seed)
        seed += 1
        inp = gen(**params, dtype=dtype)
        return inp if isinstance(inp, tuple) else (inp,)

    pos = case.get("_position")
    # 1. first execution (compile / shape specialization) on fresh inputs
    _progress(job, "first_execution", pos)
    stage = {"started": time.time()}
    try:
        inputs = make_inputs()
        cand_run(*inputs)
        sync()
        stage.update(ok=True, executions=1, elapsed_s=time.time() - stage["started"])
        rec["stages"]["first_execution"] = stage
    except PrecisionTampered as e:
        rec.update(status="contract_violation", diagnostic=f"contract_violation: {e}")
        return _finish(rec, t_case)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        rec.update(status=_classify_exception(tb), diagnostic=tb[-8000:])
        rec["stages"]["first_execution"] = {**stage, "ok": False}
        return _finish(rec, t_case)
    finally:
        inputs = None
    _progress(job, "first_execution_done", pos)
    cfg1, cfg_err = read_config(cand)
    rec["config_reads"].append({"after": "first_execution", "config": cfg1, "error": cfg_err})
    if cfg_err:
        rec.update(status="interface_error", diagnostic="interface_error: " + cfg_err)
        return _finish(rec, t_case)
    rec["config"] = cfg1
    # 2. numerical checks
    try:
        checks = run_numerical_checks(cand_run, ref_run, make_inputs, atol=job["atol"], rtol=job["rtol"],
                                      mutable_indices=mutable, sync=sync, aliasing_allowed=aliasing_allowed(rules))
    except PrecisionTampered as e:
        rec.update(status="contract_violation", diagnostic=f"contract_violation: {e}")
        return _finish(rec, t_case)
    except Exception:  # noqa: BLE001
        rec.update(status="runtime_error", diagnostic=traceback.format_exc()[-8000:])
        return _finish(rec, t_case)
    rec["stages"]["numerical_checks"] = {"executions_candidate": len(checks), "executions_reference": len(checks),
                                         "checks": [c.to_dict() for c in checks]}
    _progress(job, "numerical_checks_done", pos)
    failed = [c for c in checks if not c.ok]
    if failed:
        status = "contract_violation" if any(c.output_aliases_inputs for c in failed) else "numerical_error"
        rec.update(status=status, diagnostic=(("contract_violation: " if status == "contract_violation" else "")
                                              + "\n".join(f"{c.name}: {c.message}" for c in failed))[-8000:])
        return _finish(rec, t_case)
    cfg2, cfg_err = read_config(cand)
    rec["config_reads"].append({"after": "numerical_checks", "config": cfg2, "error": cfg_err})
    if cfg_err or not config_snapshots_equal(cfg2, cfg1):
        rec.update(status="interface_error",
                   diagnostic=("interface_error: " + cfg_err) if cfg_err else
                   f"interface_error: get_last_config() changed between calls on the same case: first {json.dumps(cfg1)}, "
                   f"later {json.dumps(cfg2)}; the configuration must be a fixed function of the case")
        return _finish(rec, t_case)
    # 3. timing on one fixed input set
    _progress(job, "timing_started", pos)
    try:
        timed_inputs = make_inputs()
        snap = [x.clone() if isinstance(x, torch.Tensor) else x for x in timed_inputs]

        def before_launch():
            for i in mutable:
                x = timed_inputs[i]
                if isinstance(x, torch.Tensor):
                    x.copy_(snap[i])

        t = job.get("timing", {})
        tr = measure(lambda: cand_run(*timed_inputs), warmup=int(t.get("warmup", 1)), repeat=int(t.get("repeat", 3)),
                     use_cuda_graph=bool(t.get("use_cuda_graph", True)), flush=bool(t.get("flush", True)),
                     before_launch=before_launch if mutable else None,
                     proton_output_dir=job.get("sandbox_dir"), label=f"case{idx:03d}", keep_profile=True)
        rec["timing"] = tr.to_dict()
        rec["stages"]["timing"] = {"warmup": tr.warmup_runs, "graph_prep": tr.graph_prep_runs, "timed": tr.timed_runs}
        expected = job.get("expected_timing_mode")
        rec["timing_mode_differs"] = bool(expected) and tr.timing_execution_mode != expected
        cf = capture_failure_status(t.get("capture_failure_policy"), expected, tr.capture_succeeded)
        if cf:
            rec.update(status=cf, diagnostic=("timing_error: CUDA-graph capture failed and the campaign's "
                                              "capture_failure_policy is 'timing_error' (no eager sample is accepted): "
                                              f"{tr.capture_error}"))
            return _finish(rec, t_case)
        if tr.mean_ms is None:
            rec.update(status="timing_error", diagnostic=tr.note or "timing produced no positive sample")
            return _finish(rec, t_case)
        cfg3, cfg_err = read_config(cand)
        rec["config_reads"].append({"after": "timing", "config": cfg3, "error": cfg_err})
        if cfg_err or not config_snapshots_equal(cfg3, cfg1):
            rec.update(status="interface_error",
                       diagnostic=("interface_error: " + cfg_err) if cfg_err else
                       f"interface_error: get_last_config() changed after timing: first {json.dumps(cfg1)}, later "
                       f"{json.dumps(cfg3)}; the configuration must be a fixed function of the case")
            return _finish(rec, t_case)
        rec.update(status="valid", latency_ms_mean=tr.mean_ms, latency_ms_samples=tr.samples_ms)
    except PrecisionTampered as e:
        rec.update(status="contract_violation", diagnostic=f"contract_violation: {e}")
    except Exception:  # noqa: BLE001
        rec.update(status="timing_error", diagnostic=traceback.format_exc()[-8000:])
    finally:
        timed_inputs = snap = None
    _progress(job, "timing_done", pos)
    return _finish(rec, t_case)


def _finish(rec: dict, t0: float) -> dict:
    rec["elapsed_s"] = time.time() - t0
    return rec


# --------------------------------------------------------------------------
# suite
# --------------------------------------------------------------------------

def summarize_suite(result: dict, cases: list[dict], case_total: int) -> dict:
    """Round-level fields from the per-case records (pure; shared with the launcher's partial-result path)."""
    evaluated = [c for c in cases if c.get("status") != "not_evaluated"]
    valid = [c for c in evaluated if c.get("status") == "valid"]
    first_bad = next((c for c in cases if c.get("status") != "valid"), None)
    result["cases"] = cases
    result["cases_total"] = case_total
    result["cases_evaluated"] = len(evaluated)
    result["valid_cases"] = len(valid)
    result["first_failing_case"] = None if first_bad is None else {
        k: first_bad.get(k) for k in ("case_id", "case_index", "params", "status")}
    distinct = []
    for c in cases:
        if c.get("config") is not None and c["config"] not in distinct:
            distinct.append(c["config"])
    result["configs_distinct"] = distinct
    result["timing_mode_differs"] = any(bool(c.get("timing_mode_differs")) for c in cases)
    if len(valid) == case_total and case_total > 0 and len(cases) == case_total:
        gm = geometric_mean([float(c["latency_ms_mean"]) for c in valid])
        if gm is None:
            result.update(status="timing_error", diagnostic="geometric mean of the case means is undefined")
        else:
            result.update(status="valid", latency_ms_geomean=gm, diagnostic=None)
    else:
        result["latency_ms_geomean"] = None
        if first_bad is not None and result.get("status") in (None, "valid", "infrastructure_incomplete_pending"):
            result["status"] = first_bad["status"] if first_bad["status"] != "not_evaluated" else "runtime_error"
            result["diagnostic"] = first_bad.get("diagnostic")
    return result


def run_job(job: dict) -> dict:
    t0 = time.time()
    result: dict = {"schema": RESULT_SCHEMA, "status": "infrastructure_incomplete", "stages": {}, "diagnostic": None,
                    "identity": job.get("identity"), "expected_timing_mode": job.get("expected_timing_mode"),
                    "cases": [], "cases_total": len(job.get("cases") or []), "cases_evaluated": 0, "valid_cases": 0,
                    "latency_ms_geomean": None, "first_failing_case": None, "suite_stopped": None}
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
    op, dtype_str = job["operator"], job["dtype"]
    dtype = resolve_dtype(dtype_str)
    gen = get_generator(op)
    rules = job.get("rules", {}) or {}
    cases = [dict(c, _position=i) for i, c in enumerate(job["cases"])]
    try:
        from tilebench.paths import operator_dir
        ref_mod = _load_module(operator_dir(op) / "impl_torch.py")
        mutable = mutable_indices(rules, ref_mod.run)
    except Exception:  # noqa: BLE001
        result["diagnostic"] = "reference import failure:\n" + traceback.format_exc()
        return result
    guard = PrecisionGuard()
    result["precision_guard"] = guard.record()
    ref_run = guard.guard_reference(ref_mod.run)

    def stop_global(status: str, diagnostic: str) -> dict:
        result.update(status=status, diagnostic=diagnostic)
        recs = [{"case_id": c["case_id"], "case_index": c.get("case_index"), "params": c["params"],
                 "status": "not_evaluated", "diagnostic": f"suite stopped before this case ({status})"} for c in cases]
        result["suite_stopped"] = {"reason": status, "at_case": None}
        summarize_suite(result, recs, len(cases))
        result.update(status=status, diagnostic=diagnostic)
        result["precision_guard"] = guard.record()
        result["elapsed_s"] = time.time() - t0
        return result

    # global: import / interface / autotuner scan
    stage = {"started": time.time()}
    _progress(job, "candidate_loaded")            # from here on, a hang is the candidate's
    try:
        cand = _load_module(Path(job["source_path"]))
        guard.verify("after importing the candidate module")
        stage.update(ok=True, elapsed_s=time.time() - stage["started"])
        result["stages"]["import_compile"] = stage
    except PrecisionTampered as e:
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
    hits = autotuner_instances(cand, job["dsl"])
    result["stages"]["autotuner_scan"] = {"hits": hits}
    if hits:
        return stop_global("contract_violation", "contract_violation: autotuner object(s) found in the generated module: "
                           + "; ".join(hits))
    cand_run = guard.guard_candidate(cand.run)

    records: list[dict] = []
    for pos, case in enumerate(cases):
        try:
            rec = run_case(job, case, cand=cand, cand_run=cand_run, ref_run=ref_run, guard=guard, mutable=mutable,
                           rules=rules, torch=torch, gen=gen, dtype=dtype, measure=measure,
                           run_numerical_checks=run_numerical_checks)
        except PrecisionRestoreFailed as e:
            result["precision_guard"] = guard.record()
            result.update(status="infrastructure_incomplete", diagnostic=f"evaluator precision restore failed: {e}")
            break
        records.append(rec)
        _append_case(job, rec)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        if rec["status"] in SUITE_STOP_STATUSES:
            result["suite_stopped"] = {"reason": rec["status"], "at_case": rec["case_id"], "position": pos}
            for rest in cases[pos + 1:]:
                records.append({"case_id": rest["case_id"], "case_index": rest.get("case_index"), "params": rest["params"],
                                "status": "not_evaluated",
                                "diagnostic": f"suite stopped at case {pos + 1} ({rec['status']})"})
            break
    if result.get("status") == "infrastructure_incomplete" and result.get("diagnostic"):
        summarize_suite(result, records, len(cases))
        result.update(status="infrastructure_incomplete")
    else:
        result["status"] = None
        summarize_suite(result, records, len(cases))
        if result["status"] is None:
            result["status"] = "runtime_error"
    result["precision_guard"] = guard.record()
    _progress(job, "suite_done")
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
