"""Expected-behaviour regression tests for the 4a1f4591 review counterexamples
(COUNTEREXAMPLES.json R1-R10) plus append-only evaluation evidence,
clean-tree publication verification, restricted-path isolation and transport
fault injection. Each test asserts the CORRECT behaviour; the old
counterexample scripts asserted the defect."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import types
from pathlib import Path

import pytest

from tilebench.llm.v2.devtools import synthetic_context
from tilebench.llm.v2.distillation import access
from tilebench.llm.v2.distillation.fixtures import synthetic_index
from tilebench.llm.v2.distillation.orchestrator import (DistillationIncomplete, DistillerConfig, extract_observations,
                                                        synthesize, verify_state_identity, write_skill)
from tilebench.llm.v2.evaluation import worker
from tilebench.llm.v2.evaluation.fingerprint import diff as fp_diff, evaluator_fingerprint
from tilebench.llm.v2.evaluation.launcher import MockEvaluator, detect_isolation, isolation_probe
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import efficiency as e
from tilebench.llm.v2.orchestration import campaign, state_machine as sm
from tilebench.llm.v2.orchestration.campaign import ResumeRefused, _check_resume, new_trajectory_state, reopen_transport_attempt, retry_incomplete
from tilebench.llm.v2.orchestration.runner import RunnerConfig, TrajectoryRunner, durable_transport
from tilebench.llm.v2.orchestration.state import TrajectoryState
from tilebench.llm.v2.providers.base import GenerationResult, ProviderConfigError
from tilebench.llm.v2.providers.ledger import append_jsonl, read_jsonl
from tilebench.llm.v2.providers.mock import MockProvider, scripted_text
from tilebench.llm.v2.providers.usage import normalize_openai_responses
from tilebench.llm.v2.skills import loader
from tilebench.llm.v2.tasks.support import eligibility

REPO = Path(__file__).resolve().parents[2]
GOOD = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {'BLOCK': 64}")


def rounds(*specs):
    return [{"round": i, "attempts": [{"cost": c} for c in costs], "valid": v, "latency_ms": lat}
            for i, (costs, v, lat) in enumerate(specs, 1)]


def _setup(study, folds, tmp_path, run_type="validation"):
    ctx, job, rules = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    el = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    fp = evaluator_fingerprint(job, worker_timeout_s=1800, isolation_backend="mock")
    st = new_trajectory_state(el, ctx, "m", "base", "h", "c", "t", run_type=run_type, campaign="t", evaluator_fp=fp)
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, feedback_limits=study["feedback"], retry_backoff_s=0.0)
    return ctx, job, st, cfg, tmp_path / "t"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- R1
def test_r1_incomplete_task_stays_in_the_frozen_denominator():
    curves = {("op", "fp16"): e.curve(1.0, rounds(([10], True, 1.0))),
              ("op", "fp32"): e.curve(1.0, rounds(([10], True, 1.0)), status="incomplete")}
    agg = e.aggregate(curves, [100], {("op", "fp16"): "eligible", ("op", "fp32"): "eligible"})
    assert agg["mean"] == [None]                                   # exact suite value unknown
    assert agg["lower_bound_mean"] == [0.5]                        # unknown counted as 0 over the full denominator
    assert agg["tasks_included"] == [("op", "fp16"), ("op", "fp32")]
    assert agg["completed_only_mean"] == [1.0] and agg["completed_only_tasks"] == [("op", "fp16")]
    assert agg["incomplete"] == [("op", "fp32")] and agg["partial"]
    with pytest.raises(ValueError, match="pre-declared"):         # execution state is not an eligibility status
        e.aggregate(curves, [100], {("op", "fp16"): "eligible", ("op", "fp32"): "incomplete"})


# ---------------------------------------------------------------- R2
def test_r2_config_snapshot_is_independent_of_later_mutation():
    mod = types.ModuleType("cand")
    cfg = {"BLOCK": 128, "nested": {"stages": 2}}
    mod.get_last_config = lambda: cfg
    first, err = worker.read_config(mod)
    assert err is None and first == {"BLOCK": 128, "nested": {"stages": 2}}
    cfg["BLOCK"] = 256
    cfg["nested"]["stages"] = 3
    second, _ = worker.read_config(mod)
    assert first == {"BLOCK": 128, "nested": {"stages": 2}}       # the snapshot did not follow the mutation
    assert second == {"BLOCK": 256, "nested": {"stages": 3}}
    assert not worker.config_snapshots_equal(first, second)        # 128 -> 256 is detected


# ---------------------------------------------------------------- R3 / R4 / lost response
def test_r3_durable_unknown_charge_survives_a_crash_before_state(study, folds, tmp_path):
    ctx, job, st, cfg, tdir = _setup(study, folds, tmp_path)
    append_jsonl(tdir / "transport.jsonl", {"round": 1, "attempt": 1, "transport_attempt": 1, "event": "sending"})
    append_jsonl(tdir / "transport.jsonl", {"round": 1, "attempt": 1, "transport_attempt": 1, "event": "failed",
                                            "outcome": "failed", "charged": "unknown", "error": "APITimeoutError"})
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=MockProvider([{"text": GOOD}]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step()
    att = st.rounds[0].attempts[0]
    assert att.cost is None and att.cost_status == "unknown"       # the prior unknown charge was not forgotten
    assert [(t["transport_attempt"], t["outcome"]) for t in att.transport] == [(1, "failed"), (2, "succeeded")]


def test_r4_repeated_recovery_reuses_the_persisted_orphan(tmp_path):
    tdir = tmp_path / "t"
    append_jsonl(tdir / "transport.jsonl", {"round": 1, "attempt": 1, "transport_attempt": 1, "event": "sending"})
    first = durable_transport(tdir, 1, 1)
    second = durable_transport(tdir, 1, 1)
    assert [t["outcome"] for t in first] == ["orphaned"] and [t["outcome"] for t in second] == ["orphaned"]
    rows = read_jsonl(tdir / "transport.jsonl")
    assert sum(1 for x in rows if x.get("event") == "orphaned") == 1          # persisted once, reused, never forgotten
    assert second[0]["charged"] == "unknown"


def test_lost_response_keeps_its_known_charge_and_a_new_request_follows(study, folds, tmp_path):
    ctx, job, st, cfg, tdir = _setup(study, folds, tmp_path)
    usage = normalize_openai_responses({"input_tokens": 30, "output_tokens": 10}).to_dict()
    append_jsonl(tdir / "transport.jsonl", {"round": 1, "attempt": 1, "transport_attempt": 1, "event": "sending"})
    append_jsonl(tdir / "transport.jsonl", {"round": 1, "attempt": 1, "transport_attempt": 1, "event": "succeeded",
                                            "outcome": "succeeded", "charged": "known", "usage": usage, "response_id": "lost"})
    prov = MockProvider([{"text": GOOD, "usage": {"input_tokens": 5, "output_tokens": 5}}])
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=prov, evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step()
    att = st.rounds[0].attempts[0]
    assert len(prov.requests) == 1                                 # the lost response is re-requested once
    assert att.transport[0]["response_lost"] is True and att.transport[1]["transport_attempt"] == 2
    assert att.cost == 40 + 10 and att.cost_status == "known"      # both charges counted
    assert any("never archived" in n for n in st.notes)


# ---------------------------------------------------------------- R5
def test_r5_exhausted_transport_has_an_explicit_recorded_resume(study, folds, tmp_path):
    ctx, job, st, cfg, tdir = _setup(study, folds, tmp_path)
    cfg.max_transport_retries = 1
    bad = MockProvider([{"text": GOOD, "transport_failures": 5, "transport_charged": "unknown"}])
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=bad, evaluator=MockEvaluator(), job=job, cfg=cfg, sleep=lambda s: None)
    assert r.run() == "incomplete" and st.rounds[0].attempts[0].verdict == "transport_failed"
    with pytest.raises(ResumeRefused, match="reason"):
        reopen_transport_attempt(st, tdir, reason="", executor="tester")
    note = reopen_transport_attempt(st, tdir, reason="network outage over; operator decision", executor="tester")
    assert note and st.status == "in_progress" and sm.next_action(st) == sm.Action("generate_initial", 1, 1)
    assert st.rounds[0].reopened_attempts[0]["verdict"] == "transport_failed"
    assert any(x.get("event") == "reopened" for x in read_jsonl(tdir / "transport.jsonl"))
    good = MockProvider([{"text": GOOD}])
    r2 = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=good, evaluator=MockEvaluator(), job=job, cfg=cfg, sleep=lambda s: None)
    r2.step(); r2.step()
    att = st.rounds[0].attempts[0]
    assert st.rounds[0].status == "valid" and len(st.rounds) == 1   # same round/attempt: no extra optimization round
    assert att.cost is None and att.cost_status == "unknown"       # the earlier unknown charges keep propagating
    assert [t["transport_attempt"] for t in att.transport] == [1, 2, 3] and att.transport[-1]["outcome"] == "succeeded"


def test_r5_provider_refusal_reopen_records_unchanged_settings(study, folds, tmp_path):
    ctx, job, st, cfg, tdir = _setup(study, folds, tmp_path)

    class Refusing:
        name = "mock"; usage_schema = "openai_responses_v1"

        def generate(self, req):
            raise ProviderConfigError("BadRequestError (400): unsupported parameter", status_code=400)
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=Refusing(), evaluator=MockEvaluator(), job=job, cfg=cfg)
    assert r.run() == "incomplete"
    note = reopen_transport_attempt(st, tdir, reason="parameter corrected", executor="tester", settings_hash_before="s", settings_hash_after="s")
    assert note and any("unchanged settings" in n for n in st.notes)
    ev = [x for x in read_jsonl(tdir / "transport.jsonl") if x.get("event") == "reopened"][0]
    assert ev["previous_verdict"] == "provider_refused" and ev["reason"] == "parameter corrected"


# ---------------------------------------------------------------- R6
def test_r6_resume_is_bound_to_the_evaluator_fingerprint(study, folds, tmp_path):
    ctx, job, st, cfg, tdir = _setup(study, folds, tmp_path, run_type="formal")
    fp_same = evaluator_fingerprint(job, worker_timeout_s=1800, isolation_backend="mock")
    fp_diff_timeout = evaluator_fingerprint(job, worker_timeout_s=600, isolation_backend="mock")
    assert fp_diff(fp_same, fp_diff_timeout) == ["worker_timeout_s"]
    from tilebench.llm.v2.providers.factory import GeneratorSpec
    gen = GeneratorSpec(name="m", provider="mock", model_id="m", api_key_env="X", settings={}, status="approved", set_by=None)
    st.generator = {"model_id": "m", "settings": {}}
    _check_resume(st, config_hash="h", content_hashes=st.content_hashes, run_type="formal", gen=gen, evaluator_fp=fp_same)
    with pytest.raises(ResumeRefused, match="evaluator changed"):
        _check_resume(st, config_hash="h", content_hashes=st.content_hashes, run_type="formal", gen=gen, evaluator_fp=fp_diff_timeout)
    st.run_type = "validation"
    with pytest.raises(ResumeRefused, match="allow_evaluator_change"):
        _check_resume(st, config_hash="h", content_hashes=st.content_hashes, run_type="validation", gen=gen, evaluator_fp=fp_diff_timeout)
    _check_resume(st, config_hash="h", content_hashes=st.content_hashes, run_type="validation", gen=gen, evaluator_fp=fp_diff_timeout,
                  allow_evaluator_change=True, tdir=tdir, executor="tester")
    assert st.evaluator_changes[0]["keys"] == ["worker_timeout_s"] and st.evaluator_fingerprint == fp_diff_timeout
    assert read_jsonl(tdir / "evaluator_changes.jsonl")[0]["executor"] == "tester"
    legacy = TrajectoryState(schema="s", trajectory_id="t", task={}, model="m", condition="base", config_hash="h",
                             content_hashes={}, output_file="f", run_type="formal", generator={"model_id": "m", "settings": {}})
    with pytest.raises(ResumeRefused, match="legacy"):
        _check_resume(legacy, config_hash="h", content_hashes={}, run_type="formal", gen=gen, evaluator_fp=fp_same)


# ---------------------------------------------------------------- append-only evaluation / compliance evidence
def test_evaluation_revisions_are_append_only_and_compliance_is_recorded_once(study, folds, tmp_path):
    ctx, job, st, cfg, tdir = _setup(study, folds, tmp_path)
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=MockProvider([{"text": GOOD}]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step(); r.step()
    adir = tdir / "round_01" / "attempt_1"
    comp = json.loads((adir / "compliance.json").read_text())
    assert len(comp["candidate_sha256"]) == 64 and comp["checker"]["checker_version"] and len(comp["rules_sha256"]) == 64
    comp_hash = _sha(adir / "compliance.json")
    meta1 = json.loads((adir / "eval_0001" / "META.json").read_text())
    assert meta1["reason"] == "initial" and meta1["supersedes"] is None and meta1["candidate_sha256"] == comp["candidate_sha256"]
    eval1_hash = _sha(adir / "eval_0001" / "evaluation.json")
    # evaluation-side infrastructure failure -> explicit retry -> a NEW revision, the old one untouched
    rec = st.rounds[0]
    rec.status, rec.diagnostic, st.status = "infrastructure_incomplete", "worker exceeded 10s before the candidate was loaded", "incomplete"
    assert retry_incomplete(st)
    r.step()
    assert rec.evaluation_revisions == ["eval_0001", "eval_0002"] and rec.evaluation_revision == "eval_0002"
    meta2 = json.loads((adir / "eval_0002" / "META.json").read_text())
    assert meta2["supersedes"] == "eval_0001" and meta2["reason"].startswith("retry_incomplete")
    assert _sha(adir / "eval_0001" / "evaluation.json") == eval1_hash and not (adir / "evaluation.json").exists()
    # resume replay of the same request reuses the stored compliance verdict (no silent re-check)
    st2 = TrajectoryState.load(tdir / "trajectory.json")
    st2.rounds = []; st2.status = "in_progress"
    r2 = TrajectoryRunner(state=st2, tdir=tdir, ctx=ctx, provider=MockProvider([]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r2.step()
    assert _sha(adir / "compliance.json") == comp_hash and st2.rounds[0].attempts[0].candidate_sha256 == comp["candidate_sha256"]


def test_review_recheck_files_are_numbered_and_the_first_record_kept(study, folds, tmp_path):
    from tilebench.llm.v2.cli import main as cli_main
    ctx, job, st, cfg, tdir = _setup(study, folds, tmp_path)
    susp = scripted_text("impl_triton.py", "import os\n# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=MockProvider([{"text": susp}]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    assert r.run() == "review_required"
    adir = tdir / "round_01" / "attempt_1"
    first = _sha(adir / "compliance.json")
    st.task["operator"] = "vector_add"; st.save(tdir / "trajectory.json")
    rc = cli_main(["review-resolve", "--trajectory-dir", str(tdir), "--recheck", "--note", "n"])
    assert rc == 1                                                 # still suspicious under the current checker: not resolved
    assert (adir / "compliance_recheck_0001.json").exists() and _sha(adir / "compliance.json") == first
    rc_rec = json.loads((adir / "compliance_recheck_0001.json").read_text())
    assert rc_rec["supersedes"] == "compliance.json" and rc_rec["checker"]["checker_version"]


# ---------------------------------------------------------------- R7
def test_r7_fold_recomputed_from_manifest_and_incomplete_sources_refused(tmp_path, study, folds):
    root = tmp_path / "camp"
    good = synthetic_index(root / "ok", folds, dsl="triton", devices=["B200"], models=["gpt"], operators=["layernorm"])
    # a formal state that labels relu (fold A) as fold B, and a 0-round incomplete formal state
    mis = {"trajectory_id": "mis", "model": "gpt", "condition": "base", "status": "complete", "run_type": "formal",
           "schema": "tilebench-llm-v2-trajectory/2", "task": {"operator": "relu", "dtype": "fp16", "dsl": "triton", "device": "B200", "fold": "B"},
           "rounds": [{"round": 1, "status": "valid", "attempts": []}]}
    inc = {**mis, "trajectory_id": "inc", "status": "incomplete", "rounds": [],
           "task": {"operator": "softmax", "dtype": "fp16", "dsl": "triton", "device": "B200", "fold": "C"}}
    for s in (mis, inc):
        d = root / s["trajectory_id"]; d.mkdir(parents=True); (d / "trajectory.json").write_text(json.dumps(s))
    index = access.build_index(root, folds=folds)
    by = {r.trajectory_id: r for r in index}
    assert by["mis"].fold == "A" and by["mis"].state_fold == "B"      # recomputed from the frozen manifest
    scope = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study, root=root, folds=folds)
    assert "mis" not in {r.trajectory_id for r in scope.selected} and "inc" not in {r.trajectory_id for r in scope.selected}
    assert any("disagrees with the frozen fold manifest" in k for k in scope.excluded)
    assert any("not complete" in k for k in scope.excluded)
    with pytest.raises(access.EvidenceAccessError, match="fold"):
        verify_state_identity(access.TrajectoryRef(**{**by["mis"].__dict__, "fold": "B", "state_fold": None}), mis, folds)
    cov = access.coverage_report(scope, study=study, folds=folds, models=["gpt", "claude"],
                                 expected_tasks=[("layernorm", "fp16"), ("softmax", "fp16")])
    assert not cov["complete"] and ("layernorm", "fp16", "claude") in cov["missing"] and ("softmax", "fp16", "gpt") in cov["missing"]


# ---------------------------------------------------------------- R8
def _scope_and_obs(tmp_path, study, folds):
    index = synthetic_index(tmp_path / "idx", folds, dsl="triton", devices=["B200"], models=["gpt"], operators=["layernorm"])
    scope = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study, root=tmp_path / "idx", folds=folds)
    cfg = DistillerConfig(model_id="m", provider_name="mock", settings={}, dsl_version="3.6.0")
    read = lambda p: json.loads(p.read_text())
    return index, scope, cfg, read


def test_r8_synthesis_is_idempotent_and_rejects_truncation(tmp_path, study, folds):
    index, scope, cfg, read = _scope_and_obs(tmp_path, study, folds)
    out = tmp_path / "d"
    obs = extract_observations(scope, MockProvider([{"text": "[obs]"}]), cfg, read_state=read, out_dir=out, folds=folds)
    prov = MockProvider([{"text": "# skill\n"}])
    res1 = synthesize(scope, obs, prov, cfg, models=["gpt"], out_dir=out)
    h1 = _sha(out / "synthesis.json")
    res2 = synthesize(scope, obs, prov, cfg, models=["gpt"], out_dir=out)
    assert len(prov.requests) == 1 and _sha(out / "synthesis.json") == h1 and res2["manifest"]["synthesis_identity"] == res1["manifest"]["synthesis_identity"]

    class Trunc:
        name = "mock"; usage_schema = "openai_responses_v1"; calls = 0

        def generate(self, req):
            self.calls += 1
            return GenerationResult(text="PARTIAL OUTPUT\n", model_id="m", provider="mock", response_id="r", usage_raw=None,
                                    usage=normalize_openai_responses({"input_tokens": 9, "output_tokens": 9}), transport_attempts=1,
                                    elapsed_s=0.0, terminal_status="max_tokens", truncated=True)
    out2 = tmp_path / "d2"
    obs2 = extract_observations(scope, MockProvider([{"text": "[obs]"}]), cfg, read_state=read, out_dir=out2, folds=folds)
    with pytest.raises(DistillationIncomplete):
        synthesize(scope, obs2, Trunc(), cfg, models=["gpt"], out_dir=out2)
    assert not (out2 / "synthesis.json").exists() and (out2 / "synthesis.partial_1.json").exists()
    partial = json.loads((out2 / "synthesis.partial_1.json").read_text())
    assert partial["accepted"] is False and partial["manifest"]["status"] == "partial" and partial["manifest"]["usage"]["synthesis"]["logical_total"] == 18
    with pytest.raises(DistillationIncomplete):
        write_skill(out2 / "skill", partial, test_only=True)
    assert not (out2 / "skill").exists()


def test_r8_extraction_truncation_duplicates_and_source_changes(tmp_path, study, folds):
    index, scope, cfg, read = _scope_and_obs(tmp_path, study, folds)
    out = tmp_path / "d"

    class Trunc:
        name = "mock"; usage_schema = "openai_responses_v1"

        def generate(self, req):
            return GenerationResult(text="partial", model_id="m", provider="mock", response_id="r", usage_raw=None,
                                    usage=normalize_openai_responses({"input_tokens": 1, "output_tokens": 1}), transport_attempts=1,
                                    elapsed_s=0.0, terminal_status="max_tokens", truncated=True)
    with pytest.raises(DistillationIncomplete):
        extract_observations(scope, Trunc(), cfg, read_state=read, out_dir=out, folds=folds)
    tid = scope.selected[0].trajectory_id
    assert (out / "observations" / f"{tid}.partial_1.json").exists() and not (out / "observations" / f"{tid}.json").exists()
    assert (out / "observations" / f"{tid}.raw.json").exists() and (out / "observations" / f"{tid}.request.md").exists()
    prov = MockProvider([{"text": "[obs]"}] * 3)
    obs = extract_observations(scope, prov, cfg, read_state=read, out_dir=out, folds=folds)
    assert len(prov.requests) == 1 and obs[0]["accepted"]
    with pytest.raises(access.EvidenceAccessError, match="duplicate"):
        synthesize(scope, obs + obs, MockProvider([{"text": "s"}]), cfg, models=["gpt"])
    with pytest.raises(access.EvidenceAccessError, match="missing"):
        synthesize(scope, [], MockProvider([{"text": "s"}]), cfg, models=["gpt"])
    # a changed candidate source changes the materials -> the stored observation is not reused
    state_path = Path(scope.selected[0].path)
    st = json.loads(state_path.read_text())
    src = state_path.parent / "cand.py"; src.write_text("def run(x): return x\n")
    st["rounds"][0]["source_path"] = str(src); st["rounds"][0]["attempts"] = [{"attempt": 1, "kind": "initial", "verdict": "clear", "cost": 1, "source_path": str(src)}]
    state_path.write_text(json.dumps(st))
    index2 = access.refs_from_states([{**st, "path": str(state_path), "state_sha256": _sha(state_path)}], folds)   # fixture files are not named trajectory.json
    scope2 = access.evaluation_scope(index2, dsl="triton", held_out_fold="A", study=study, root=tmp_path / "idx", folds=folds)
    assert scope2.selected
    extract_observations(scope2, prov, cfg, read_state=read, out_dir=out, folds=folds)
    assert len(prov.requests) == 2 and (out / "observations" / f"{tid}.partial_2.json").exists()


# ---------------------------------------------------------------- R9 / R10
def test_r9_enhanced_loader_validates_skill_provenance(skill_env, study):
    m, root = skill_env
    loader.compose_context(m, study, dsl="triton", device="B200", fold="A", condition="enhanced")   # valid manifest
    mpath = root / "skills/optimization/triton/source-B200/fold-A/manifest.json"
    good = json.loads(mpath.read_text())
    bad = {**good, "source_device": "GH200", "evaluation_or_release": "release", "held_out_fold": None, "training_folds": ["A", "B", "C"],
           "compatible_versions": ["3.5.0"]}
    mpath.write_text(json.dumps(bad))
    with pytest.raises(loader.SkillPermissionError, match="provenance rejected") as ei:
        loader.compose_context(m, study, dsl="triton", device="B200", fold="A", condition="enhanced")
    msg = str(ei.value)
    assert "source_device" in msg and "mode" in msg and "compatible_versions" in msg
    mpath.write_text(json.dumps({**good, "content_sha256": "0" * 64}))
    with pytest.raises(loader.SkillPermissionError, match="content_sha256"):
        loader.compose_context(m, study, dsl="triton", device="B200", fold="A", condition="enhanced")
    mpath.unlink()
    with pytest.raises(loader.SkillError, match="manifest"):
        loader.compose_context(m, study, dsl="triton", device="B200", fold="A", condition="enhanced")


def test_r10_component_hash_covers_ordered_attachments(skill_env, study):
    m, root = skill_env
    att = root / "skills/reference/triton/3.6.0/ATTACH.md"
    att.write_text("BEFORE\n")
    loader.register_asset(m, "reference", "triton", "3.6.0", "skills/reference/triton/3.6.0/SKILL.md", permission="public",
                          status="approved", source="test", attachments=["skills/reference/triton/3.6.0/ATTACH.md"])
    c1 = loader.load_component(m, "reference", "triton", "3.6.0")
    h1 = loader.hash_record([c1])["reference:triton@3.6.0"]
    assert h1 == c1.sha256_composed != c1.sha256_injected          # composed text != main body alone
    att.write_text("AFTER\n")
    loader.register_asset(m, "reference", "triton", "3.6.0", "skills/reference/triton/3.6.0/SKILL.md", permission="public",
                          status="approved", source="test", attachments=["skills/reference/triton/3.6.0/ATTACH.md"])
    c2 = loader.load_component(m, "reference", "triton", "3.6.0")
    assert c2.sha256_injected == c1.sha256_injected and loader.hash_record([c2])["reference:triton@3.6.0"] != h1


# ---------------------------------------------------------------- publication / isolation
def test_published_index_verifies_in_a_clean_git_archive():
    rel = "artifacts/llm_v2/validation_b200_2026-10-05"
    if not shutil.which("git") or not (REPO / rel / "INDEX.json").exists():
        pytest.skip("no git or no published campaign")
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", rel], capture_output=True)
        if archive.returncode != 0:
            pytest.skip("git archive unavailable")
        subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
        root = Path(tmp) / rel
        idx = json.load(open(root / "INDEX.json"))["files"]
        missing = [p for p in idx if not (root / p).exists()]
        bad = [p for p in idx if p not in missing and _sha(root / p) != idx[p]["sha256"]]
        assert not missing and not bad, (len(missing), len(bad))


def test_restricted_paths_are_unreadable_inside_the_sandbox():
    if detect_isolation("auto")["backend"] != "bwrap":
        pytest.skip("bwrap sandbox unavailable on this host")
    rep = isolation_probe("vector_add")
    assert rep["ok"], rep["results"]
    res = rep["results"]
    assert not res["manual_dsl_impl"]["readable"] and not res["home_bashrc"]["readable"] and not res["sentinel_in_outputs"]["readable"]
    assert res["task_reference"]["readable"] and not res["network"]["reachable"] and res["secret_env_names"] == []
    assert rep["device_nodes"]["amd"]["verified"] == "pending" and rep["device_nodes"]["neuron"]["verified"] == "pending"


def test_formal_preflight_requires_the_sandbox(monkeypatch, study):
    from tilebench.llm.v2.evaluation import launcher
    monkeypatch.setattr(launcher, "detect_isolation", lambda mode="auto": {"backend": "none", "probe_error": "bwrap not found", "device_nodes": {}})
    pf = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="formal")
    assert any("require the bwrap sandbox" in b for b in pf.blockers)
    pfv = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="validation", models_selected=("gpt",), provider="openai")
    assert not any("sandbox" in b for b in pfv.blockers) and any("without the sandbox" in w for w in pfv.warnings)
