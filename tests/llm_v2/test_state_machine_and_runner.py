"""Ten-round state machine; mock end-to-end runner with persistence and resume."""
import json
from pathlib import Path

import pytest

from tilebench.llm.v2.devtools import run_mock_trajectory, synthetic_context
from tilebench.llm.v2.evaluation.launcher import MockEvaluator
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.orchestration import state_machine as sm
from tilebench.llm.v2.orchestration.campaign import new_trajectory_state
from tilebench.llm.v2.orchestration.runner import ConfigMismatch, RunnerConfig, TrajectoryRunner
from tilebench.llm.v2.orchestration.state import AttemptRecord, TrajectoryState
from tilebench.llm.v2.providers.mock import MockProvider, scripted_text
from tilebench.llm.v2.tasks.support import eligibility


def _state():
    return TrajectoryState(schema="s", trajectory_id="t", task={}, model="m", condition="base", config_hash="h",
                           content_hashes={}, output_file="impl_triton.py")


def _att(n, verdict, cost=10):
    return AttemptRecord(attempt=n, kind="initial" if n == 1 else "repair", request_hash="r", prompt_chars=1,
                         transport_attempts=1, verdict=verdict, cost=cost, source_path="/dev/null")


def test_violation_allows_at_most_three_generations_and_charges_all():
    st = _state()
    assert sm.next_action(st).kind == "generate_initial"
    sm.apply_attempt(st, 1, _att(1, "confirmed_violation"))
    assert sm.next_action(st) == sm.Action("repair", 1, 2)
    sm.apply_attempt(st, 1, _att(2, "confirmed_violation"))
    assert sm.next_action(st) == sm.Action("repair", 1, 3)
    sm.apply_attempt(st, 1, _att(3, "confirmed_violation"))
    assert st.rounds[0].status == "contract_violation" and sum(a.cost for a in st.rounds[0].attempts) == 30
    assert sm.next_action(st) == sm.Action("generate_initial", 2, 1)
    with pytest.raises(RuntimeError):
        sm.apply_attempt(st, 1, _att(4, "clear"))


def test_ordinary_failures_consume_the_round_without_repair():
    st = _state()
    sm.apply_attempt(st, 1, _att(1, "format_error"))
    assert st.rounds[0].status == "format_error" and sm.next_action(st) == sm.Action("generate_initial", 2, 1)
    sm.apply_attempt(st, 2, _att(1, "clear"))
    assert sm.next_action(st).kind == "evaluate"
    sm.apply_evaluation(st, 2, {"status": "numerical_error", "diagnostic": "x"})
    assert st.rounds[1].status == "numerical_error" and sm.next_action(st) == sm.Action("generate_initial", 3, 1)


def test_regression_keeps_best_and_ten_rounds_complete():
    st = _state()
    for r in range(1, 11):
        sm.apply_attempt(st, r, _att(1, "clear"))
        lat = 1.0 if r == 2 else 2.0 + r
        sm.apply_evaluation(st, r, {"status": "valid", "latency_ms_mean": lat, "latency_ms_samples": [lat] * 3, "config": {}})
    assert st.best_valid["round"] == 2 and st.status == "complete" and sm.next_action(st).kind == "done"
    assert len(st.valid_rounds()) == 10


def test_review_required_blocks_and_resolves():
    st = _state()
    sm.apply_attempt(st, 1, _att(1, "review_required"))
    assert st.status == "review_required" and sm.next_action(st).kind == "blocked_review"
    sm.resolve_review(st, 1, "compliant", "human checked")
    assert sm.next_action(st).kind == "evaluate"


def test_infrastructure_failure_marks_incomplete():
    st = _state()
    sm.apply_attempt(st, 1, _att(1, "clear"))
    sm.apply_evaluation(st, 1, {"status": "infrastructure_incomplete", "diagnostic": "worker died"})
    assert st.status == "incomplete" and sm.next_action(st).kind == "incomplete"


def test_two_samples_is_a_timing_error():
    st = _state()
    sm.apply_attempt(st, 1, _att(1, "clear"))
    sm.apply_evaluation(st, 1, {"status": "valid", "latency_ms_mean": 1.0, "latency_ms_samples": [1.0, 1.0]})
    assert st.rounds[0].status == "timing_error"


def test_mock_ten_rounds_persist_and_resume_without_new_requests(tmp_path):
    out = run_mock_trajectory(tmp_path, operator="vector_add", dtype="fp16", device="B200", dsl="triton", condition="base")
    assert out["status"] == "complete" and len(out["rounds"]) == 10
    statuses = [s for _, s, _ in out["rounds"]]
    assert statuses[2] == "valid" and statuses[4] == "numerical_error"      # round 3 repaired; round 5 consumed
    assert out["attempts"] == 11 and out["transport_failures"] == 1
    tdir = Path(out["trajectory_dir"])
    ledger = [json.loads(l) for l in (tdir / "usage.jsonl").read_text().splitlines()]
    assert len(ledger) == 11                                                 # every generation, violations included
    assert (tdir / "round_03" / "attempt_2" / "impl_triton.py").exists() and (tdir / "best_valid" / "impl_triton.py").exists()
    # resume: provider that raises on any call; completed state must be reloaded untouched
    again = run_mock_trajectory(tmp_path, operator="vector_add", dtype="fp16", device="B200", dsl="triton", condition="base", resume=True)
    assert again["status"] == "complete" and again["attempts"] == 11
    assert len((tdir / "usage.jsonl").read_text().splitlines()) == 11


def test_resume_mid_trajectory_reuses_archived_responses(tmp_path, study, folds):
    ctx, job, rules = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    e = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    st = new_trajectory_state(e, ctx, "m", "base", "h", "c", "t")
    script = [{"text": scripted_text("impl_triton.py", f"# MOCK: valid {1.0 + i / 10}\ndef run(*a): pass\ndef get_last_config(): return {{}}")} for i in range(10)]
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, feedback_limits=study["feedback"])
    tdir = tmp_path / "t"
    r1 = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=MockProvider(script[:4]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    for _ in range(8):      # 4 generations + 4 evaluations
        r1.step()
    assert len(st.rounds) == 4 and st.rounds[-1].status == "valid"
    # resume from disk with a provider holding only the remaining 6 responses
    st2 = TrajectoryState.load(tdir / "trajectory.json")
    r2 = TrajectoryRunner(state=st2, tdir=tdir, ctx=ctx, provider=MockProvider(script[4:]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    assert r2.run() == "complete"
    assert len(st2.rounds) == 10 and len((tdir / "usage.jsonl").read_text().splitlines()) == 10


def test_resume_refuses_foreign_archived_response(tmp_path, study, folds):
    ctx, job, rules = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    e = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    st = new_trajectory_state(e, ctx, "m", "base", "h", "c", "t")
    tdir = tmp_path / "t"
    adir = tdir / "round_01" / "attempt_1"
    adir.mkdir(parents=True)
    (adir / "response.json").write_text(json.dumps({"request_hash": "not-this-request", "result": {}}))
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, feedback_limits=study["feedback"])
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=MockProvider([]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    with pytest.raises(ConfigMismatch):
        r.step()


def test_transport_exhaustion_is_incomplete_not_fabricated(tmp_path, study, folds):
    ctx, job, rules = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    e = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    st = new_trajectory_state(e, ctx, "m", "base", "h", "c", "t")
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, max_transport_retries=1, feedback_limits=study["feedback"])
    r = TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=MockProvider([{"text": "x", "transport_failures": 5}]),
                         evaluator=MockEvaluator(), job=job, cfg=cfg, sleep=lambda s: None)
    assert r.run() == "incomplete"
    # connection-level failures never reached the provider: cost is a known 0, not unknown
    assert st.rounds[0].status == "infrastructure_incomplete" and st.rounds[0].attempts[0].cost == 0
    assert st.rounds[0].attempts[0].cost_status == "known" and len(st.transport_log) == 2
    # a failure that may have been billed (timeout after sending) leaves the cost unknown
    st2 = new_trajectory_state(e, ctx, "m", "base", "h", "c", "t")
    r2 = TrajectoryRunner(state=st2, tdir=tmp_path / "t2", ctx=ctx,
                          provider=MockProvider([{"text": "x", "transport_failures": 5, "transport_charged": "unknown"}]),
                          evaluator=MockEvaluator(), job=job, cfg=cfg, sleep=lambda s: None)
    assert r2.run() == "incomplete" and st2.rounds[0].attempts[0].cost is None
    assert st2.rounds[0].attempts[0].cost_status == "unknown"


def test_no_cross_task_context_in_prompts(tmp_path, study, folds):
    """A trajectory for operator A never sees code or runtimes of operator B."""
    ctx_a, job_a, rules = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    ctx_b, job_b, _ = synthetic_context("relu", "fp16", "B200", "triton", "base", study, folds)
    e_a = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    e_b = eligibility("relu", "fp16", "B200", "triton", study, folds)
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, feedback_limits=study["feedback"])
    marker = "RELU_SECRET_MARKER_42"
    pb = MockProvider([{"text": scripted_text("impl_triton.py", f"# MOCK: valid 1.0\n# {marker}\ndef run(*a): pass\ndef get_last_config(): return {{}}")}] * 2)
    sb = new_trajectory_state(e_b, ctx_b, "m", "base", "h", "c", "t")
    TrajectoryRunner(state=sb, tdir=tmp_path / "b", ctx=ctx_b, provider=pb, evaluator=MockEvaluator(), job=job_b, cfg=cfg).step()
    pa = MockProvider([{"text": scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")}] * 4)
    sa = new_trajectory_state(e_a, ctx_a, "m", "base", "h", "c", "t")
    ra = TrajectoryRunner(state=sa, tdir=tmp_path / "a", ctx=ctx_a, provider=pa, evaluator=MockEvaluator(), job=job_a, cfg=cfg)
    for _ in range(4):
        ra.step()
    assert all(marker not in req.user for req in pa.requests)
