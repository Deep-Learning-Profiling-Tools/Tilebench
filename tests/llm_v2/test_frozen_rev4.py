"""The revision-4 frozen manifest (S_llm-rev4-20case-2026-10-07) must match the
code and manifests it pins: any drift fails here before a campaign could start
under a silently different protocol."""
import hashlib
import json

from tilebench.llm.v2.evaluation import fingerprint as fpm
from tilebench.llm.v2.evaluation.worker import SUITE_STOP_STATUSES
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import baselines, cost
from tilebench.llm.v2.metrics import empirical as E
from tilebench.llm.v2.orchestration.campaign import protocol_identity
from tilebench.llm.v2.prompts.renderer import templates_sha256
from tilebench.llm.v2.skills.loader import load_manifest
from tilebench.llm.v2.tasks import representative
from tilebench.paths import REPO_ROOT

FROZEN = ms.MANIFEST_DIR / "frozen" / "B200_base_rev4_2026-10-07.json"


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_rev4_manifest_matches_the_tree():
    f = json.loads(FROZEN.read_text())
    study, models, folds, modes = ms.load_study(), ms.load_models(), ms.load_folds(), ms.load_arithmetic_modes()
    p = f["protocol"]
    assert (p["revision"], p["rounds"], p["max_generations_per_round"], p["cases_per_task"], p["adjudicator"]) == \
        (4, 5, 1, 20, "claude-code")
    assert {k: p[k] for k in protocol_identity(study)} == protocol_identity(study)
    assert f["study_config_hash"] == ms.study_config_hash(study, models, folds, modes)
    for name, digest in f["manifests"].items():
        assert _sha(ms.MANIFEST_DIR / name) == digest, name
    cs = ms.load_case_sets()
    assert f["case_sets"]["sha256"] == _sha(ms.MANIFEST_DIR / "case_sets.yaml")
    assert f["case_sets"]["case_set_ids"] == {op: v["case_set_id"] for op, v in cs["operators"].items()}
    assert all(len(v["cases"]) == 20 for v in cs["operators"].values()) and len(cs["operators"]) == 45
    assert f["representative_dtypes"]["mapping"] == representative.selected() and f["representative_dtypes"]["tasks_per_dsl"] == 45
    assert f["checker"] == fpm.checker_fingerprint()
    pol = f["evaluator_fingerprint_policy"]
    assert pol["evaluation_sources_sha256"] == fpm._sources_sha256(fpm.V2_ROOT, fpm.EVALUATION_SOURCES)
    assert pol["core_sources_sha256"] == fpm._sources_sha256(fpm.CORE_ROOT, fpm.CORE_SOURCES)
    assert pol["worker_timeout_s"] == study["evaluation"]["worker_timeout_s"] == 3600
    assert tuple(study["evaluation"]["stop_suite_on"]) == SUITE_STOP_STATUSES == tuple(pol["suite"]["stop_suite_on"])
    assert f["pricing"]["pricing_snapshot_sha256"] == cost.pricing_sha256()
    assert f["scoring_binding"] == E.scoring_binding("B200") and f["scoring_table"]["status_counts"] == {"ok": 900}
    assert json.loads((REPO_ROOT / f["scoring_table"]["path"]).read_text())["scoring_sha256"] == f["scoring_table"]["scoring_sha256"]
    bl = baselines.load(REPO_ROOT / f["torch_baseline"]["path"])
    assert bl["sha256"] == f["torch_baseline"]["sha256"] and f["torch_baseline"]["cases_matched"] == 900
    assert f["templates_sha256"] == templates_sha256()
    assert f["excluded_campaigns_sha256"] == _sha(ms.MANIFEST_DIR / "excluded_campaigns.yaml")
    assert len(f["contracts"]) == 45
    for op, c in f["contracts"].items():
        d = REPO_ROOT / "tilebench/llm/v2/contracts/data" / op
        assert (_sha(d / "contract.md"), _sha(d / "evaluator_rules.json"), _sha(d / "audit.json")) == \
            (c["contract_md"], c["evaluator_rules_json"], c["audit_json"]), op
    man = load_manifest()
    for key, rec in f["skills"].items():
        if key == "optimization":
            continue
        kind, rest = key.split("/")
        k, ver = rest.split("@")
        e = man[kind][k][ver]
        assert (e["sha256_raw"], e["sha256_injected"], e["status"]) == (rec["sha256_raw"], rec["sha256_injected"], "approved"), key
        assert sorted(e["sendable_to"]) == ["anthropic", "openai"]
    assert set(f["dsls"]) == {"triton", "cutile", "tilelang"}
    g = models["roles"]["generator"]
    assert (g["gpt"]["model_id"], g["gpt"]["reasoning_effort"], g["gpt"]["max_output_tokens"]) == ("gpt-6.1-sol", "xhigh", 128000)
    assert (g["claude"]["model_id"], g["claude"]["output_effort"], g["claude"]["max_output_tokens"]) == ("claude-opus-5-5", "xhigh", 128000)
    assert g["claude"]["thinking"] == {"type": "adaptive"}
