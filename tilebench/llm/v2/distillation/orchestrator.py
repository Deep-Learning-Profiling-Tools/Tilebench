"""Distillation orchestration: per-trajectory evidence extraction (map) and
synthesis (reduce), both under the same EvidenceScope, producing a skill
text plus a manifest.

Binding, layer by layer (all refused by code):
1. the index is built from the campaign root with file hashes (access.build_index);
2. EvidenceScope.open re-verifies hash, root containment and symlinks;
3. verify_state_identity checks that the state's own fields (trajectory id
   recomputed from task/model/condition, dsl, device, operator, dtype, fold,
   model, condition, run_type) equal the reference;
4. candidate sources are read only from under the trajectory's own
   directory, and their hashes are recorded;
5. every observation carries the scope hash and the state hash it was
   extracted under; synthesis refuses observations from another scope.

Persistence: observations/<trajectory_id>.json and synthesis.json under
out_dir; an existing observation with matching scope/state hashes is reused
without a new request (resume). The distiller config hash covers the model
id, the settings, the system prompts and the ACTUAL template texts.

Real skills are produced only when the study owner runs this on formal Base
data with an approved distiller and frozen folds; everything else is written
as test-only and never registered in a production manifest."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tilebench.llm.v2.distillation.access import EvidenceAccessError, EvidenceScope, TrajectoryRef
from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text
from tilebench.llm.v2.orchestration.identity import trajectory_id as compute_trajectory_id
from tilebench.llm.v2.prompts.renderer import load_template, render_distill_extraction, render_distill_synthesis
from tilebench.llm.v2.providers.base import GenerationRequest

EXTRACTION_SYSTEM = "You extract optimization evidence from one kernel-generation trajectory."
SYNTHESIS_SYSTEM = "You synthesize a conditional Optimization Skill from observations."
TEMPLATES = ("distill_evidence_extraction", "distill_synthesis")


@dataclass
class DistillerConfig:
    model_id: str
    provider_name: str
    settings: dict
    dsl_version: str

    def config_hash(self) -> str:
        payload = {"model_id": self.model_id, "provider": self.provider_name, "settings": self.settings,
                   "system_prompts": {"extraction": EXTRACTION_SYSTEM, "synthesis": SYNTHESIS_SYSTEM},
                   "templates": {name: sha256_text(load_template(name)) for name in TEMPLATES}}
        return sha256_text(canonical_json(payload))


def verify_state_identity(ref: TrajectoryRef, state: dict) -> None:
    t = state.get("task", {})
    expect = {"trajectory_id": ref.trajectory_id, "dsl": ref.dsl, "device": ref.device, "operator": ref.operator,
              "dtype": ref.dtype, "fold": ref.fold, "model": ref.model, "condition": ref.condition, "run_type": ref.run_type}
    actual = {"trajectory_id": state.get("trajectory_id"), "dsl": t.get("dsl"), "device": t.get("device"),
              "operator": t.get("operator"), "dtype": t.get("dtype"), "fold": t.get("fold"), "model": state.get("model"),
              "condition": state.get("condition"), "run_type": state.get("run_type", "formal")}
    diff = {k: (expect[k], actual[k]) for k in expect if expect[k] != actual[k]}
    if diff:
        raise EvidenceAccessError(f"{ref.trajectory_id}: state identity disagrees with the index: {diff}")
    if all(k in t for k in ("operator", "dtype", "case_id", "device", "dsl")):
        recomputed = compute_trajectory_id(t, state["model"], state["condition"])
        if recomputed != state.get("trajectory_id"):
            raise EvidenceAccessError(f"{ref.trajectory_id}: trajectory id does not match its task/model/condition")


def _source_text(path_str: str | None, tdir: Path | None) -> tuple[str | None, str | None]:
    if not path_str:
        return None, None
    p = Path(path_str)
    if tdir is not None:
        try:
            p.resolve().relative_to(tdir.resolve())
        except ValueError:
            raise EvidenceAccessError(f"source {p} lies outside the trajectory directory {tdir}")
    if p.is_symlink() or not p.exists():
        return None, None
    text = p.read_text()
    return text, hashlib.sha256(text.encode()).hexdigest()


def rounds_block(state: dict, tdir: Path | None, sol: dict | None = None) -> tuple[str, dict]:
    """Model-visible rounds block for the extraction prompt, plus the hashes
    of every source it embeds. Includes per-attempt verdicts and token
    costs, raw timing samples, the execution mode and (when the caller
    supplies it from the offline SOL record) the declared SOL analysis."""
    lines = []
    hashes: dict = {}
    if sol:
        lines.append(f"SOL analysis (offline, declared mode {sol.get('arithmetic_mode')}): T_SOL = {sol.get('t_sol_ms')} ms; "
                     f"status {sol.get('status')}")
    for r in state.get("rounds", []):
        lines.append(f"### Round {r['round']} — status {r['status']}")
        for a in r.get("attempts", []):
            lines.append(f"attempt {a.get('attempt')} ({a.get('kind')}): verdict {a.get('verdict')}, "
                         f"tokens {a.get('cost')} ({a.get('cost_status', 'known')})")
        if r.get("latency_ms_mean") is not None:
            lines.append(f"runtime_ms: {r['latency_ms_mean']:.4f} (samples {r.get('latency_ms_samples')}; "
                         f"mode {r.get('timing_execution_mode')})")
            if sol and sol.get("t_sol_ms"):
                lines.append(f"sol_efficiency: {float(sol['t_sol_ms']) / float(r['latency_ms_mean']):.4f}")
        if r.get("config"):
            lines.append(f"config: {json.dumps(r['config'])}")
        if r.get("diagnostic"):
            lines.append(f"diagnostic: {str(r['diagnostic'])[:2000]}")
        src, h = _source_text(r.get("source_path"), tdir)
        if src:
            hashes[f"round_{r['round']}"] = h
            lines.append(f"```python\n{src.rstrip()}\n```")
    return "\n".join(lines), hashes


def _obs_path(out_dir: Path | None, tid: str) -> Path | None:
    return (out_dir / "observations" / f"{tid}.json") if out_dir is not None else None


def extract_observations(scope: EvidenceScope, provider, cfg: DistillerConfig, *, read_state,
                         out_dir: Path | None = None, sol_info: dict | None = None,
                         log=lambda s: None) -> list[dict]:
    """Map step: one request per selected trajectory. read_state(path) -> dict."""
    obs = []
    scope_hash = scope.sha256()
    cfg_hash = cfg.config_hash()
    for ref in scope.selected:
        path = scope.open(ref)                      # raises for anything outside the scope
        state = read_state(path)
        verify_state_identity(ref, state)
        if state["task"]["fold"] not in scope.training_folds or state["condition"] != "base":
            raise EvidenceAccessError(f"{ref.trajectory_id}: state disagrees with the scope")
        op = _obs_path(out_dir, ref.trajectory_id)
        if op is not None and op.exists():
            prev = json.loads(op.read_text())
            if prev.get("scope_sha256") == scope_hash and prev.get("state_sha256") == ref.state_sha256 \
                    and prev.get("distiller_config_hash") == cfg_hash:
                obs.append(prev)
                log(f"{ref.trajectory_id}: observation reused (no request)")
                continue
        block, src_hashes = rounds_block(state, path.parent, (sol_info or {}).get(ref.trajectory_id))
        user = render_distill_extraction({
            "dsl": scope.dsl, "dsl_version": cfg.dsl_version, "source_device": scope.source_device,
            "training_folds": ", ".join(scope.training_folds), "held_out_fold": scope.held_out_fold or "(none: release mode)",
            "trajectory_id": ref.trajectory_id, "operator": ref.operator, "dtype": ref.dtype, "model": ref.model,
            "rounds_block": block,
        })
        res = provider.generate(GenerationRequest(system=EXTRACTION_SYSTEM, user=user, model_id=cfg.model_id,
                                                  provider=cfg.provider_name, settings=cfg.settings,
                                                  metadata={"trajectory": ref.trajectory_id}))
        rec = {"trajectory_id": ref.trajectory_id, "operator": ref.operator, "dtype": ref.dtype, "model": ref.model,
               "text": res.text, "usage": res.usage.to_dict(), "response_id": res.response_id,
               "model_id": res.model_id, "terminal_status": res.terminal_status,
               "scope_sha256": scope_hash, "state_sha256": ref.state_sha256, "distiller_config_hash": cfg_hash,
               "request_sha256": sha256_text(EXTRACTION_SYSTEM + "\0" + user), "source_sha256": src_hashes}
        if op is not None:
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        obs.append(rec)
        log(f"{ref.trajectory_id}: observation extracted ({res.usage.logical_total} tokens)")
    return obs


def synthesize(scope: EvidenceScope, observations: list[dict], provider, cfg: DistillerConfig, *,
               models: list[str], out_dir: Path | None = None) -> dict:
    scope_hash = scope.sha256()
    for o in observations:
        if o.get("scope_sha256") != scope_hash:
            raise EvidenceAccessError(f"observation {o.get('trajectory_id')} was extracted under another scope")
    selected = {r.trajectory_id for r in scope.selected}
    extra = {o["trajectory_id"] for o in observations} - selected
    if extra:
        raise EvidenceAccessError(f"observations outside the scope: {sorted(extra)}")
    user = render_distill_synthesis({
        "dsl": scope.dsl, "dsl_version": cfg.dsl_version, "source_device": scope.source_device,
        "training_folds": ", ".join(scope.training_folds), "held_out_fold": scope.held_out_fold or "(none: release mode)",
        "models": ", ".join(models), "n_trajectories": str(len(observations)),
        "trajectory_ids": ", ".join(o["trajectory_id"] for o in observations),
        "observations_block": "\n\n".join(f"### {o['trajectory_id']} ({o['operator']})\n{o['text']}" for o in observations),
    })
    res = provider.generate(GenerationRequest(system=SYNTHESIS_SYSTEM, user=user, model_id=cfg.model_id,
                                              provider=cfg.provider_name, settings=cfg.settings,
                                              metadata={"synthesis": scope.dsl}))
    text = (res.text or "").strip() + "\n"
    manifest = {
        "schema": "tilebench-optimization-skill/2",
        "dsl": scope.dsl, "compatible_versions": [cfg.dsl_version],
        **scope.manifest(),
        "scope_sha256": scope_hash,
        "evaluation_or_release": scope.mode,
        "distiller": {"model_id": cfg.model_id, "provider": cfg.provider_name, "settings": cfg.settings,
                      "response_id": res.response_id, "terminal_status": res.terminal_status},
        "distiller_config_hash": cfg.config_hash(),
        "observation_sha256": {o["trajectory_id"]: sha256_text(o["text"] or "") for o in observations},
        "content_sha256": sha256_text(text),
        "status": "draft",            # frozen only by the study owner, before any Enhanced result is viewed
        "usage": {"extraction": [o["usage"] for o in observations], "synthesis": res.usage.to_dict()},
    }
    result = {"text": text, "manifest": manifest}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "synthesis.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def write_skill(out_dir: Path, result: dict, *, test_only: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = result["text"]
    manifest = dict(result["manifest"])
    if test_only:
        text = "<!-- TEST-ONLY synthetic Optimization Skill; never referenced by a production manifest -->\n" + text
        manifest["status"] = "test-only"
        manifest["content_sha256"] = sha256_text(text)
    (out_dir / "SKILL.md").write_text(text)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
