"""Manifests, folds, case selection, eligibility and prompt-field export."""
import copy

import pytest

from tilebench.paths import list_operators
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.tasks import case_selection as cs, fields, support


def test_study_manifest_validates_and_is_frozen_protocol(study):
    assert study["trajectory"]["rounds"] == 10 and study["trajectory"]["max_generations_per_round"] == 3
    assert (study["timing"]["warmup"], study["timing"]["repeat"]) == (1, 3)
    assert study["support_matrix"]["MI300X"]["dsls"] == ["triton"] and study["support_matrix"]["Trn2"]["dsls"] == ["nki"]


@pytest.mark.parametrize("mutation", [
    ("trajectory", "rounds", 9), ("trajectory", "max_generations_per_round", 4), ("timing", "repeat", 100)])
def test_study_rejects_protocol_changes(study, mutation):
    bad = copy.deepcopy(study)
    bad[mutation[0]][mutation[1]] = mutation[2]
    with pytest.raises(ms.ManifestError):
        ms.validate_study(bad)


def test_folds_partition_all_45_operators_and_keep_families_whole(folds):
    ops = list_operators()
    assert len(ops) == 45
    ms.validate_folds(folds, ops)
    sizes = [len(folds["folds"][f]["operators"]) for f in "ABC"]
    assert sizes == [15, 15, 15]
    assert folds["status"] == "frozen" and "FINAL_FREEZE_AND_START_B200.md" in folds["approved_by"]
    assert ms.blockers_folds(folds) == []
    assert ms.blockers_folds({**folds, "status": "proposed"}) and ms.blockers_folds({**folds, "approved_by": None})


def test_folds_reject_split_family(folds):
    bad = copy.deepcopy(folds)
    bad["folds"]["A"]["operators"].remove("relu")
    bad["folds"]["B"]["operators"].append("relu")
    with pytest.raises(ms.ManifestError, match="split across folds"):
        ms.validate_folds(bad, list_operators())


def test_training_folds():
    assert ms.training_folds("A") == ("B", "C")
    with pytest.raises(ms.ManifestError):
        ms.training_folds("D")


def test_models_manifest_gates_by_status():
    models = ms.load_models()
    assert ms.blockers_models(models) == []                           # generators: owner-approved (provenance recorded)
    for name in ("gpt", "claude"):
        cfg = models["roles"]["generator"][name]
        assert cfg["status"] == "approved" and "NEXT_STEP_CLAUDE.md" in cfg["approved_by"] and cfg["approved_evidence"]
    assert not ms.blockers_models(models, accept_status=("approved", "candidate"))
    assert ms.blockers_models(models, roles=("distiller",)) == []     # owner decision of 2026-10-06
    unset = {**models, "roles": {**models["roles"], "distiller": {"provider": None, "model_id": None, "status": "unset"}}}
    assert any("distiller" in b for b in ms.blockers_models(unset, roles=("distiller",)))
    for name, key_env in (("gpt", "OPENAI_API_KEY"), ("claude", "CLAUDE_API_KEY")):
        cfg = models["roles"]["generator"][name]
        assert cfg["api_key_env"] == key_env and cfg["max_output_tokens"] == 128000
    assert models["roles"]["generator"]["gpt"]["reasoning_effort"] == "xhigh"
    assert models["roles"]["generator"]["claude"]["output_effort"] == "xhigh"
    assert models["roles"]["generator"]["claude"]["thinking"] == {"type": "adaptive"}


def test_case_selection_takes_largest_expanded_case_not_combined_maxima():
    sel = cs.select_representative_case("matmul_fp32_fp16_fp8", "fp16")
    cases = cs.expand_cases("matmul_fp32_fp16_fp8", cs.load_operator_config("matmul_fp32_fp16_fp8"))
    chosen = cases[sel.case_index]
    assert cs.case_params(chosen) == sel.params                 # params come from ONE real case
    assert sel.problem_size == max(c["problem_size"] for c in sel.candidates)
    assert len(sel.candidates) == 20 and sel.case_id and len(sel.case_id) == 16


def test_case_selection_ties_keep_last_case():
    sel = cs.select_representative_case("flash_attention", "fp16")
    assert sel.case_index == max(c["case_index"] for c in sel.candidates if c["problem_size"] == sel.problem_size)


def test_task_table_statuses(study, folds):
    table = support.task_table(study, folds, ["matmul_fp32_fp16_fp8", "vector_add"])
    by = {(e.key.device, e.key.dsl, e.key.operator, e.key.dtype): e for e in table}
    assert by[("MI300X", "triton", "matmul_fp32_fp16_fp8", "fp8_e4m3fn")].status == "unsupported"
    assert by[("MI300X", "triton", "matmul_fp32_fp16_fp8", "fp8_e4m3fn")].fp8_format == "float8_e4m3fn"
    assert by[("B200", "cutile", "vector_add", "int8")].status == "eligible"
    assert by[("Trn2", "nki", "vector_add", "fp16")].status == "needs_review"
    assert all(e.key.dsl in study["support_matrix"][e.key.device]["dsls"] for e in table)
    assert all(e.fold == ms.fold_of(folds, e.key.operator) for e in table)


def test_prompt_fields_whitelist_excludes_metrics_and_uses_effective_tolerance():
    cfg = cs.load_operator_config("2d_conv")
    sel = cs.select_representative_case("2d_conv", "fp16")
    tf_b200 = fields.task_fields("2d_conv", "fp16", sel.params, cfg, "blackwell")
    tf_cdna = fields.task_fields("2d_conv", "fp16", sel.params, cfg, "cdna3")
    assert tf_b200.tolerance["atol"] == 0.1 and tf_cdna.tolerance["atol"] == 0.2
    assert "arch_overrides[cdna3]" in tf_cdna.tolerance["source"]
    assert tf_b200.run_signature.startswith("def run(")
    fields.assert_no_forbidden_config(tf_b200.functional_reference.source)
    assert "flops_expr" not in tf_b200.functional_reference.source


def test_dtype_default_tolerance_when_config_has_none():
    tol = fields.effective_tolerance({}, "fp16", None)
    assert tol == {"atol": 1e-3, "rtol": 1e-3, "source": "config.verify+dtype_default"}
