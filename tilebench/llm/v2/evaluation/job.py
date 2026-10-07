"""Evaluator-only job schema.

The generator sees `prompts.renderer.TaskContext`; the evaluator receives an
`EvaluationJob`. They are built from the same eligibility record and the same
effective tolerance, and the campaign asserts that the tolerance the prompt
states equals the one the worker applies. Nothing in an EvaluationJob is ever
rendered into a prompt: it carries the evaluator rules of the contract, the
normalized timing settings and the identity of the trajectory it serves.

Timing key normalization (study.yaml -> worker):
    timing.warmup                 -> warmup
    timing.repeat                 -> repeat
    timing.cuda_graph_requested   -> use_cuda_graph
    timing.cache_flush == operator_boundary_outside_timed_scope -> flush = True
    timing.capture_failure_policy -> capture_failure_policy
    timing.record_prep_runs       -> record_prep_runs
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from tilebench.llm.v2.evaluation.adapters import require_adapter
from tilebench.llm.v2.manifests.schema import CAPTURE_FAILURE_POLICIES, canonical_json
from tilebench.llm.v2.tasks.case_selection import load_operator_config
from tilebench.llm.v2.tasks.fields import effective_tolerance

JOB_SCHEMA = "tilebench-llm-v2-evaljob/2"   # /2: a suite of cases (protocol revision 4)


def timing_settings(study: dict, run_type: str = "formal") -> dict:
    """Formal runs use timing.capture_failure_policy_formal when the study
    declares one (frozen 2026-10-06: timing_error); validation runs use
    timing.capture_failure_policy (time_eagerly_and_flag)."""
    t = study["timing"]
    policy = t.get("capture_failure_policy")
    if run_type == "formal" and t.get("capture_failure_policy_formal") is not None:
        policy = t["capture_failure_policy_formal"]
    if policy not in CAPTURE_FAILURE_POLICIES:
        raise ValueError(f"study.yaml: capture_failure_policy {policy!r} is not defined")
    return {
        "warmup": int(t["warmup"]),
        "repeat": int(t["repeat"]),
        "use_cuda_graph": bool(t["cuda_graph_requested"]),
        "flush": t["cache_flush"] == "operator_boundary_outside_timed_scope",
        "capture_failure_policy": policy,
        "record_prep_runs": bool(t.get("record_prep_runs", True)),
    }


def expected_timing_mode(study: dict, device: str) -> str:
    """graph on adapters that replay a CUDA graph, eager otherwise (ROCm)."""
    adapter = study["support_matrix"][device]["timing_adapter"]
    if adapter == "proton_cuda_graph" and study["timing"]["cuda_graph_requested"]:
        return "graph"
    return "eager"


def single_case(operator: str, dtype: str, params: dict) -> list[dict]:
    """A one-case suite (re-evaluation of legacy single-case states, unit tests)."""
    from tilebench.data.tensors import infer_problem_size
    from tilebench.llm.v2.tasks.case_selection import make_case_id
    return [{"case_id": make_case_id(dict(params), dtype), "case_index": 0, "params": dict(params),
             "problem_size": int(infer_problem_size(operator, dict(params)))}]


@dataclass
class EvaluationJob:
    operator: str
    dtype: str
    cases: list                       # [{case_id, case_index, params, problem_size}], evaluated in this order
    dsl: str
    device: str
    arch: str | None
    atol: float
    rtol: float
    tolerance_source: str
    rules: dict                       # evaluator_rules.json content (never rendered)
    timing: dict                      # normalized keys, see timing_settings()
    expected_timing_mode: str         # graph | eager
    timing_adapter: str
    identity: dict = field(default_factory=dict)   # trajectory_id, model, condition, campaign, run_type
    schema: str = JOB_SCHEMA

    def rules_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.rules).encode()).hexdigest()

    def case_ids(self) -> list[str]:
        return [c["case_id"] for c in self.cases]

    def record(self) -> dict:
        """Identity/settings record for trajectory.json (rules by hash only)."""
        d = asdict(self)
        d.pop("rules")
        d["rules_sha256"] = self.rules_sha256()
        return d

    def worker_job(self, *, source_path: str, sandbox_dir: str, seed: int, round_index: int, attempt: int) -> dict:
        return {
            "schema": self.schema, "operator": self.operator, "dtype": self.dtype,
            "cases": [{k: c.get(k) for k in ("case_id", "case_index", "params", "problem_size")} for c in self.cases],
            "dsl": self.dsl, "device": self.device, "arch": self.arch,
            "source_path": source_path, "sandbox_dir": sandbox_dir,
            "atol": self.atol, "rtol": self.rtol, "tolerance_source": self.tolerance_source,
            "rules": self.rules, "timing": self.timing, "expected_timing_mode": self.expected_timing_mode,
            "seed": int(seed), "identity": {**self.identity, "round": round_index, "attempt": attempt},
        }


def validate_worker_job(job: dict) -> list[str]:
    errs = []
    if job.get("schema") != JOB_SCHEMA:
        errs.append(f"schema must be {JOB_SCHEMA}")
    for key in ("operator", "dtype", "cases", "dsl", "source_path", "sandbox_dir", "atol", "rtol", "rules", "timing", "seed"):
        if key not in job:
            errs.append(f"missing key {key}")
    cases = job.get("cases")
    if "cases" in job and (not isinstance(cases, list) or not cases or
                           any(not isinstance(c, dict) or "params" not in c or "case_id" not in c for c in cases)):
        errs.append("cases must be a non-empty list of {case_id, params, ...}")
    t = job.get("timing") or {}
    for key in ("warmup", "repeat", "use_cuda_graph", "flush", "capture_failure_policy"):
        if key not in t:
            errs.append(f"timing.{key} missing")
    return errs


def build_evaluation_job(*, operator: str, dtype: str, dsl: str, device: str, arch: str | None,
                         rules: dict, study: dict, cases: list | None = None, params: dict | None = None,
                         identity: dict | None = None, config: dict | None = None) -> EvaluationJob:
    """`cases`: the task's frozen case set (revision 4). `params` alone builds a one-case suite (legacy states, tests)."""
    if cases is None:
        if params is None:
            raise ValueError("build_evaluation_job needs cases (or params for a one-case suite)")
        cases = single_case(operator, dtype, params)
    cfg = config if config is not None else load_operator_config(operator)
    tol = effective_tolerance(cfg, dtype, arch)
    adapter = study["support_matrix"][device]["timing_adapter"]
    require_adapter(adapter)
    cases = [{k: c.get(k) for k in ("case_id", "case_index", "params", "problem_size")} for c in cases]
    return EvaluationJob(operator=operator, dtype=dtype, cases=cases, dsl=dsl, device=device, arch=arch,
                         atol=tol["atol"], rtol=tol["rtol"], tolerance_source=tol["source"], rules=rules,
                         timing=timing_settings(study, run_type=str((identity or {}).get("run_type", "formal"))),
                         expected_timing_mode=expected_timing_mode(study, device),
                         timing_adapter=adapter, identity=dict(identity or {}))


def job_to_json(job: EvaluationJob) -> str:
    return json.dumps(asdict(job), indent=1, sort_keys=True)
