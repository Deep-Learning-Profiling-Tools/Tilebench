"""Source-device / fold access boundaries and the synthetic distillation pipeline."""
import json

import pytest

from tilebench.llm.v2.distillation import access
from tilebench.llm.v2.distillation.fixtures import synthetic_index
from tilebench.llm.v2.distillation.orchestrator import DistillerConfig, extract_observations, synthesize, write_skill
from tilebench.llm.v2.manifests.schema import fold_of
from tilebench.llm.v2.providers.mock import MockProvider

OPS = ["relu", "batched_matmul", "flash_attention", "layernorm", "1d_conv", "softmax"]   # folds A,A,B,B,C,C


@pytest.fixture
def index(tmp_path, folds):
    return synthetic_index(tmp_path, folds, dsl="triton", devices=["B200", "GH200", "MI300X"],
                           models=["gpt", "claude"], operators=OPS)


def test_evaluation_scope_reads_only_source_device_training_folds_base(index, study, folds):
    scope = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study)
    assert scope.training_folds == ("B", "C") and scope.allowed_devices == ("B200",)
    for ref in scope.selected:
        assert ref.device == "B200" and ref.condition == "base" and fold_of(folds, ref.operator) in ("B", "C")
    assert {ref.model for ref in scope.selected} == {"gpt", "claude"}          # both models pooled
    assert len(scope.selected) == 2 * 4                                          # 2 models x 4 ops in B+C
    assert "fold A is held out" in scope.excluded and "device GH200 is not an allowed source" in scope.excluded
    assert "not a base trajectory" in scope.excluded


def test_held_out_operator_other_dtype_or_device_is_refused(index, study):
    scope = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study)
    held = next(r for r in index if r.operator == "relu" and r.device == "B200" and r.condition == "base")
    with pytest.raises(access.EvidenceAccessError):
        scope.open(held)
    target = next(r for r in index if r.operator == "layernorm" and r.device == "GH200" and r.condition == "base")
    with pytest.raises(access.EvidenceAccessError):
        scope.open(target)
    enhanced = next(r for r in index if r.operator == "layernorm" and r.device == "B200" and r.condition == "enhanced")
    with pytest.raises(access.EvidenceAccessError):
        scope.open(enhanced)


def test_other_dsl_is_refused(tmp_path, study, folds):
    other = synthetic_index(tmp_path / "c", folds, dsl="cutile", devices=["B200"], models=["gpt"], operators=OPS)
    scope = access.evaluation_scope(other, dsl="triton", held_out_fold="A", study=study)
    assert not scope.selected and "other DSL cutile" in scope.excluded


def test_nki_source_is_trn2_only(tmp_path, study, folds):
    idx = synthetic_index(tmp_path / "n", folds, dsl="nki", devices=["B200", "Trn2"], models=["gpt"], operators=OPS)
    scope = access.evaluation_scope(idx, dsl="nki", held_out_fold="C", study=study)
    assert scope.source_device == "Trn2" and all(r.device == "Trn2" for r in scope.selected)
    assert "device B200 is not an allowed source" in scope.excluded


def test_release_scope_is_separate_and_labelled(index, study):
    rel = access.release_scope(index, dsl="triton", study=study)
    assert rel.mode == "release" and set(rel.allowed_devices) == {"B200", "GH200", "MI300X"}
    assert rel.training_folds == ("A", "B", "C") and rel.manifest()["mode"] == "release"
    ev = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study)
    assert len(rel.selected) > len(ev.selected)


def test_synthetic_distillation_pipeline_writes_test_only_skill(tmp_path, index, study):
    scope = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study)
    n = len(scope.selected)
    provider = MockProvider([{"text": json.dumps([{"rule": "r", "applicability": "a", "effect": "e", "confounders": [], "evidence": {}, "limits": ""}])}] * n
                            + [{"text": "# Triton Optimization Skill\n- rule (supporting: t1)\n"}])
    cfg = DistillerConfig(model_id="mock", provider_name="mock", settings={}, dsl_version="3.6.0")
    obs = extract_observations(scope, provider, cfg, read_state=lambda p: json.loads(p.read_text()))
    assert len(obs) == n
    res = synthesize(scope, obs, provider, cfg, models=["gpt", "claude"])
    m = res["manifest"]
    assert m["held_out_fold"] == "A" and m["training_folds"] == ["B", "C"] and m["source_device"] == "B200"
    assert m["evaluation_or_release"] == "evaluation" and m["status"] == "draft" and len(m["source_trajectory_ids"]) == n
    out = tmp_path / "skill"
    write_skill(out, res, test_only=True)
    written = json.loads((out / "manifest.json").read_text())
    assert written["status"] == "test-only" and (out / "SKILL.md").read_text().startswith("<!-- TEST-ONLY")
