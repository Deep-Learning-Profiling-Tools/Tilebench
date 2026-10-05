"""Contract loading/leak checks, campaign context composition and preflight gates."""
import pytest

from tilebench.llm.v2.contracts import loader as cl
from tilebench.llm.v2.orchestration import campaign
from tilebench.llm.v2.prompts.renderer import render_initial
from tilebench.llm.v2.tasks.support import eligibility


def test_contract_loads_and_gates_on_approval(contract_env):
    c = cl.load_contract("vector_add")
    assert c.status == "approved" and c.rules["operator"] == "vector_add" and len(c.sha256) == 64
    with pytest.raises(cl.ContractError, match="draft"):
        cl.load_contract("relu")
    assert cl.load_contract("relu", require_approved=False).status == "draft"


def test_contract_leak_check_blocks_tunables_and_latencies(contract_env):
    with pytest.raises(cl.ContractLeakError):
        cl.load_contract("mul2", require_approved=False)
    assert cl.leak_check("the winner config was 2x speedup") and not cl.leak_check("one program per row; fp32 accumulation")


def test_campaign_context_uses_manifest_components_and_contract(contract_env, skill_env, study, folds):
    m, _ = skill_env
    e = eligibility("vector_add", "fp16", "GH200", "triton", study, folds)
    ctx = campaign.build_task_context(e, study, m, "enhanced", require_approved=False)
    assert [c.kind for c in ctx.components] == ["reference", "device", "optimization"]
    assert ctx.device == "GH200" and ctx.output_file == "impl_triton.py" and "out = a + b" in ctx.contract_text
    u = render_initial(ctx)
    assert "# triton reference" in u and "# GH200 facts" in u and "# triton skill C" in u
    st = campaign.new_trajectory_state(e, ctx, "gpt", "enhanced", "cfg", "chash", "thash")
    assert set(st.content_hashes) == {"reference:triton@3.6.0", "device:GH200@2026-10-05", "optimization:triton/C@v1", "contract", "templates"}
    assert st.task["fold"] == "C" and st.task["case_id"] == e.key.case_id


def test_preflight_blocks_live_without_models_and_nki(monkeypatch, study):
    from tilebench.llm.v2.skills import loader
    monkeypatch.setattr(loader, "MANIFEST_PATH", loader.REPO_ROOT / "does-not-exist.json")
    pf = campaign.preflight("B200", "triton", "enhanced", live=True, study=study)
    # formal gate: candidate model ids are not approved; validation gate accepts them
    assert not pf.ok and any("status is 'candidate'" in b for b in pf.blockers) and any("manifest missing" in b for b in pf.blockers)
    assert any("not frozen" in b for b in pf.blockers)
    pfv = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="validation",
                             models_selected=("gpt",), provider="openai")
    assert not any("models.yaml" in b for b in pfv.blockers)
    pf2 = campaign.preflight("Trn2", "nki", "base", live=False, study=study)
    assert any("NKI timing adapter" in b for b in pf2.blockers)
    pf3 = campaign.preflight("MI300X", "cutile", "base", live=False, study=study)
    assert any("not supported on MI300X" in b for b in pf3.blockers)


def test_private_metadata_only_entry_is_never_readable(skill_env):
    m, _ = skill_env
    from tilebench.llm.v2.skills import loader
    m["reference"]["nki"] = {"beta5": {"path": None, "sha256_raw": "x", "sha256_injected": None, "permission": "private",
                                       "status": "draft", "attachments": []}}
    with pytest.raises(loader.SkillPermissionError):
        loader.load_component(m, "reference", "nki", "beta5", require_status=())
    with pytest.raises(loader.SkillMissingError, match="private asset"):
        loader.load_component(m, "reference", "nki", "beta5", require_status=(), provider_sendable=False)


def test_snapshots_come_from_the_renderer(tmp_path):
    from tilebench.llm.v2.devtools import make_snapshots
    out = make_snapshots(tmp_path / "snap", use_manifest=False)
    names = out["files"]
    assert names[:5] == ["00_system.md", "01_initial_base.md", "02_refinement_after_compile_failure.md",
                         "03_refinement_after_regression.md", "04_compliance_repair.md"]
    text = (tmp_path / "snap" / "02_refinement_after_compile_failure.md").read_text()
    assert "[line withheld]" in text and "roofline" not in text
    enh = (tmp_path / "snap" / "05_initial_enhanced_TESTONLY_skill.md").read_text()
    assert "TEST-ONLY" in enh and "Optimization guidance" in enh
    cli_out = (tmp_path / "snap" / "INDEX.json").read_text()
    assert "sha256" in cli_out
