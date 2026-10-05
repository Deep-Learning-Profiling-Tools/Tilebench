"""Development utilities behind the CLI: mock trajectories, metric recompute,
coverage, review queue, review-bundle export, device capture. Nothing here
calls a paid API or launches a kernel."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from tilebench.paths import REPO_ROOT, list_operators

from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import efficiency as eff
from tilebench.llm.v2.metrics.sol import t_sol
from tilebench.llm.v2.orchestration.identity import trajectory_dir
from tilebench.llm.v2.orchestration.runner import RunnerConfig, TrajectoryRunner
from tilebench.llm.v2.orchestration.state import TrajectoryState
from tilebench.llm.v2.prompts.renderer import TaskContext
from tilebench.llm.v2.providers.base import TransportError
from tilebench.llm.v2.providers.mock import MockProvider, scripted_text
from tilebench.llm.v2.skills.loader import SkillComponent
from tilebench.llm.v2.tasks.case_selection import load_operator_config
from tilebench.llm.v2.tasks.fields import task_fields
from tilebench.llm.v2.tasks.support import eligibility

TEST_ONLY_BANNER = "<!-- TEST-ONLY synthetic component; never referenced by a production manifest -->\n"


def synthetic_component(kind: str, key: str, version: str, body: str, status: str = "test-only") -> SkillComponent:
    text = TEST_ONLY_BANNER + body
    h = hashlib.sha256(text.encode()).hexdigest()
    return SkillComponent(kind=kind, key=key, version=version, path=f"<synthetic:{kind}/{key}>", sha256_raw=h,
                          sha256_injected=h, permission="public", status=status, text=text, chars=len(text), attachments=[])


def synthetic_context(operator: str, dtype: str, device: str, dsl: str, condition: str, study: dict, folds: dict,
                      *, contract_text: str | None = None) -> tuple[TaskContext, dict, dict]:
    e = eligibility(operator, dtype, device, dsl, study, folds)
    cfg = load_operator_config(operator)
    tf = task_fields(operator, dtype, e.params, cfg, None)
    comps = [synthetic_component("reference", dsl, study["dsls"][dsl]["reference_version"],
                                 f"# {dsl} reference (synthetic)\nUse the DSL's load/compute/store primitives.\n"),
             synthetic_component("device", device, "synthetic", f"# {device} device context (synthetic)\nunknown facts.\n")]
    if condition == "enhanced":
        comps.append(synthetic_component("optimization", f"{dsl}/{e.fold}", "synthetic",
                                         "# Optimization skill (synthetic)\n- When tiles exceed on-chip capacity, split them.\n",
                                         status="test-only"))
    contract = contract_text or (f"# {operator}: canonical algorithm contract (synthetic)\n## Functional semantics\n"
                                 "Compute the reference function.\n## Permitted PyTorch operations\n- torch.empty_like\n")
    ctx = TaskContext(operator=operator, dtype=dtype, torch_dtype=tf.torch_dtype, dsl=dsl,
                      dsl_version=study["dsls"][dsl]["reference_version"], device=device,
                      output_file=study["dsls"][dsl]["output_file"], params=e.params, atol=tf.tolerance["atol"],
                      rtol=tf.tolerance["rtol"], tolerance_source=tf.tolerance["source"], run_signature=tf.run_signature,
                      returns=tf.returns, functional_reference=tf.functional_reference.source, fp8_format=e.fp8_format,
                      components=comps, contract_text=contract, rounds=study["trajectory"]["rounds"])
    rules = {"forbidden_substitutions": [], "required_evidence": [], "allowed_torch_calls": ["torch.empty_like", "torch.empty"],
             "mutation": {"inputs_mutated": [], "restore_required": False}}
    task = {"operator": operator, "dtype": dtype, "case_id": e.key.case_id, "params": e.params, "device": device,
            "dsl": dsl, "fold": e.fold, "fp8_format": e.fp8_format, "atol": tf.tolerance["atol"], "rtol": tf.tolerance["rtol"],
            "rules": rules}
    return ctx, task, rules


def mock_script(output_file: str, rounds: int = 10) -> list[dict]:
    """10 rounds: round 3 has a confirmed violation then a repair, round 5 a
    numerical error, round 7 a transport failure before success."""
    items = []
    for r in range(1, rounds + 1):
        if r == 3:
            items.append({"text": scripted_text(output_file, "import triton\n@triton.autotune(configs=[], key=[])\ndef k(): pass\ndef run(*a): pass\ndef get_last_config(): return {}")})
            items.append({"text": scripted_text(output_file, f"# MOCK: valid {1.2 - 0.02 * r:.3f}\ndef run(*a): pass\ndef get_last_config(): return {{'BLOCK': 64}}")})
        elif r == 5:
            items.append({"text": scripted_text(output_file, "# MOCK: numerical_error\ndef run(*a): pass\ndef get_last_config(): return {'BLOCK': 32}")})
        elif r == 7:
            items.append({"text": scripted_text(output_file, f"# MOCK: valid {1.2 - 0.02 * r:.3f}\ndef run(*a): pass\ndef get_last_config(): return {{'BLOCK': 128}}"),
                          "transport_failures": 1})
        else:
            items.append({"text": scripted_text(output_file, f"# MOCK: valid {1.2 - 0.02 * r:.3f}\ndef run(*a): pass\ndef get_last_config(): return {{'BLOCK': 128}}")})
    return items


class _FailingProvider:
    name = "mock-fail"
    usage_schema = "openai_responses_v1"

    def generate(self, request):
        raise TransportError("resume must not re-request an archived attempt")


def run_mock_trajectory(out: Path, *, operator: str, dtype: str, device: str, dsl: str, condition: str,
                        resume: bool = False) -> dict:
    from tilebench.llm.v2.evaluation.launcher import MockEvaluator
    from tilebench.llm.v2.orchestration.campaign import new_trajectory_state
    study, folds = ms.load_study(), ms.load_folds()
    ctx, task, rules = synthetic_context(operator, dtype, device, dsl, condition, study, folds)
    e = eligibility(operator, dtype, device, dsl, study, folds)
    chash = ms.study_config_hash(study, ms.load_models(), folds, ms.load_arithmetic_modes())
    tdir = trajectory_dir("mock", task, "mock-model", condition, root=out)
    tpath = tdir / "trajectory.json"
    if resume and tpath.exists():
        state = TrajectoryState.load(tpath)
        if state.config_hash != chash:
            raise RuntimeError("config hash changed; refusing to resume")
        provider = _FailingProvider()
    else:
        state = new_trajectory_state(e, ctx, "mock-model", condition, chash, "synthetic-contract", "synthetic-templates")
        provider = MockProvider(mock_script(ctx.output_file))
    cfg = RunnerConfig(model_id="mock-model", provider_name="mock", settings={}, retry_backoff_s=0.0,
                       feedback_limits=study["feedback"], total_prompt_max_chars=study["context_limits"]["total_prompt"])
    runner = TrajectoryRunner(state=state, tdir=tdir, ctx=ctx, provider=provider, evaluator=MockEvaluator(), rules=rules,
                              cfg=cfg, sleep=lambda s: None)
    status = runner.run()
    return {"trajectory_dir": str(tdir), "status": status, "rounds": [(r.round, r.status, r.latency_ms_mean) for r in state.rounds],
            "best_valid": state.best_valid, "attempts": sum(len(r.attempts) for r in state.rounds),
            "transport_failures": len(state.transport_log)}


def _walk_states(campaign_dir: Path):
    for p in sorted(campaign_dir.rglob("trajectory.json")):
        yield p, TrajectoryState.load(p)


def recompute_metrics(campaign_dir: Path, budgets: list[int] | None = None) -> dict:
    modes = ms.load_arithmetic_modes()
    groups: dict[tuple, dict] = {}
    budgets = budgets or [50_000, 100_000, 200_000, 400_000, 800_000]
    for path, st in _walk_states(campaign_dir):
        t = st.task
        cfg = load_operator_config(t["operator"])
        sol = t_sol(t["device"], t["operator"], t["dtype"], t["params"], t.get("problem_size", 1),
                    cfg.get("metrics", {}), modes)
        status = "incomplete" if st.status in ("in_progress", "incomplete", "review_required") else "eligible"
        curve = eff.curve(sol.t_sol_ms, st.metric_rounds(), status="complete" if status == "eligible" else "incomplete")
        key = (t["device"], t["dsl"], st.model, st.condition)
        g = groups.setdefault(key, {"curves": {}, "eligible": {}, "sol_status": {}})
        g["curves"][(t["operator"], t["dtype"])] = curve
        g["eligible"][(t["operator"], t["dtype"])] = status
        g["sol_status"][f"{t['operator']}/{t['dtype']}"] = sol.status
    out = {}
    for key, g in groups.items():
        agg = eff.aggregate(g["curves"], budgets, g["eligible"])
        agg["sol_status"] = g["sol_status"]
        agg["audit_flags"] = {f"{k[0]}/{k[1]}": c.audit_flags for k, c in g["curves"].items() if c.audit_flags}
        out["/".join(key)] = agg
    return out


def coverage(campaign_dir: Path) -> dict:
    counts: dict = {}
    for path, st in _walk_states(campaign_dir):
        key = f"{st.task['device']}/{st.task['dsl']}/{st.model}/{st.condition}"
        c = counts.setdefault(key, {"trajectories": 0, "status": {}, "rounds_done": 0})
        c["trajectories"] += 1
        c["status"][st.status] = c["status"].get(st.status, 0) + 1
        c["rounds_done"] += sum(1 for r in st.rounds if r.status != "pending")
    return counts


def review_queue(campaign_dir: Path) -> list[dict]:
    out = []
    for path, st in _walk_states(campaign_dir):
        if st.status == "review_required":
            rec = st.rounds[-1]
            out.append({"trajectory": str(path.parent), "round": rec.round, "attempt": rec.attempts[-1].attempt,
                        "evidence": (rec.attempts[-1].compliance or {}).get("static", {}).get("evidence", [])})
    return out


def export_review_bundle(out: Path) -> dict:
    """Copy shareable material only: docs, contracts, public/internal skill
    texts, prompt snapshots, tests summary. Never outputs/, credentials, or
    private skill bodies."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"files": [], "excluded": []}
    include = [REPO_ROOT / "docs" / "llm_v2", REPO_ROOT / "tilebench" / "llm" / "v2" / "contracts" / "data",
               REPO_ROOT / "tilebench" / "llm" / "v2" / "prompts" / "templates", REPO_ROOT / "tilebench" / "llm" / "v2" / "manifests",
               REPO_ROOT / "skills" / "reference", REPO_ROOT / "skills" / "device", REPO_ROOT / "skills" / "manifest.json"]
    skill_manifest = {}
    mp = REPO_ROOT / "skills" / "manifest.json"
    if mp.exists():
        skill_manifest = json.loads(mp.read_text())
    private_paths = {e["path"] for kind in ("reference", "device", "optimization") for v in skill_manifest.get(kind, {}).values()
                     for e in v.values() if e.get("permission") == "private" and e.get("path")}
    for src in include:
        if not src.exists():
            continue
        files = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file()]
        for f in files:
            rel = f.relative_to(REPO_ROOT)
            if str(rel) in private_paths or "__pycache__" in rel.parts:
                manifest["excluded"].append(str(rel))
                continue
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            manifest["files"].append({"path": str(rel), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size})
    (out / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return {"out": str(out), "files": len(manifest["files"]), "excluded": manifest["excluded"]}


def capture_device_facts() -> dict:
    out: dict = {"torch": None, "nvidia_smi": None, "rocminfo": None, "neuron_ls": None}
    try:
        import torch
        out["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["torch"]["device"] = {k: getattr(p, k) for k in dir(p) if not k.startswith("_") and
                                      isinstance(getattr(p, k), (int, str, float, bool))}
    except Exception as e:  # noqa: BLE001
        out["torch"] = {"error": str(e)}
    for key, cmd in (("nvidia_smi", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory", "--format=csv"]),
                     ("rocminfo", ["rocminfo"]), ("neuron_ls", ["neuron-ls"])):
        try:
            out[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout[:4000]
        except Exception as e:  # noqa: BLE001
            out[key] = f"unavailable: {type(e).__name__}"
    return out


# --------------------------------------------------------------------------
# Rendered prompt snapshots (produced by the real renderer)
# --------------------------------------------------------------------------

def make_snapshots(out: Path, *, operator: str = "vector_add", dtype: str = "fp16", device: str = "B200",
                   dsl: str = "triton", use_manifest: bool = True) -> dict:
    """Write representative rendered prompts. With use_manifest=True the real
    (possibly draft) Reference/Device skills and the real contract are used;
    the Enhanced snapshot and the distillation fixture use synthetic,
    TEST-ONLY components because no real Optimization Skill exists."""
    from tilebench.llm.v2.contracts.loader import ContractError, load_contract
    from tilebench.llm.v2.prompts.renderer import (render_initial, render_refinement, render_repair, render_system,
                                                   render_distill_synthesis)
    from tilebench.llm.v2.skills.loader import SkillError, compose_context, load_manifest
    study, folds = ms.load_study(), ms.load_folds()
    out.mkdir(parents=True, exist_ok=True)
    ctx, task, rules = synthetic_context(operator, dtype, device, dsl, "base", study, folds)
    provenance = {"components": "synthetic"}
    if use_manifest:
        try:
            comps = compose_context(load_manifest(), study, dsl=dsl, device=device, fold=task["fold"], condition="base",
                                    require_approved=False)
            ctx.components = comps
            provenance["components"] = {f"{c.kind}:{c.key}@{c.version}": c.sha256_injected for c in comps}
        except SkillError as e:
            provenance["components_error"] = str(e)
        try:
            c = load_contract(operator, require_approved=False)
            ctx.contract_text = c.model_text
            provenance["contract"] = {"status": c.status, "sha256": c.sha256}
        except ContractError as e:
            provenance["contract_error"] = str(e)
    limits = study["feedback"]
    files = {}
    files["00_system.md"] = render_system(ctx)
    files["01_initial_base.md"] = render_initial(ctx)
    prev_fail = {"round": 1, "status": "compile_error", "source": "import triton\n# ...candidate...\n", "config": None,
                 "latency_ms_mean": None, "latency_ms_samples": None,
                 "diagnostic": "CompilationError: at 12:8: tl.dot requires K >= 16\nroofline: 0% (this line must be withheld)"}
    files["02_refinement_after_compile_failure.md"] = render_refinement(ctx, round_index=2, prev=prev_fail, best_valid=None,
                                                                         history=[], limits=limits)
    best = {"round": 3, "latency_ms_mean": 0.0410, "source": "# best valid candidate source\n", "config": {"BLOCK": 1024}}
    prev_slow = {"round": 4, "status": "valid", "source": "# slower candidate source\n", "config": {"BLOCK": 256},
                 "latency_ms_mean": 0.0532, "latency_ms_samples": [0.0528, 0.0533, 0.0535], "diagnostic": None}
    hist = [{"round": 3, "status": "valid", "latency_ms_mean": 0.0410, "latency_ms_samples": [0.0405, 0.0410, 0.0415]}, prev_slow]
    files["03_refinement_after_regression.md"] = render_refinement(ctx, round_index=5, prev=prev_slow, best_valid=best,
                                                                    history=hist, limits=limits)
    files["04_compliance_repair.md"] = render_repair(ctx, round_index=6, attempt=2, max_attempts=3,
                                                     violations=["line 4: autotune decorator triton.autotune",
                                                                 "line 19: reference-library computation torch.matmul"],
                                                     rejected_source="# rejected candidate source\n",
                                                     fallback={"round": 3, "source": "# best valid candidate source\n"})
    ctx_e, _, _ = synthetic_context(operator, dtype, device, dsl, "enhanced", study, folds)
    if use_manifest and "components_error" not in provenance:
        ctx_e.components = list(ctx.components) + [c for c in ctx_e.components if c.kind == "optimization"]
        ctx_e.contract_text = ctx.contract_text
    files["05_initial_enhanced_TESTONLY_skill.md"] = render_initial(ctx_e)
    files["06_distill_synthesis_fixture.md"] = render_distill_synthesis({
        "dsl": dsl, "dsl_version": study["dsls"][dsl]["reference_version"], "source_device": study["dsls"][dsl]["source_device"],
        "training_folds": "B, C", "held_out_fold": "A", "models": "gpt, claude", "n_trajectories": "2",
        "trajectory_ids": "t-synthetic-1, t-synthetic-2",
        "observations_block": "### t-synthetic-1 (layernorm)\n[synthetic observation JSON]\n\n### t-synthetic-2 (softmax)\n[synthetic observation JSON]\n"})
    index = {}
    for name, text in files.items():
        (out / name).write_text(text)
        index[name] = {"chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    (out / "INDEX.json").write_text(json.dumps({"task": {"operator": operator, "dtype": dtype, "device": device, "dsl": dsl},
                                                "provenance": provenance, "files": index}, indent=1) + "\n")
    return {"out": str(out), "files": list(files), "provenance": provenance}
