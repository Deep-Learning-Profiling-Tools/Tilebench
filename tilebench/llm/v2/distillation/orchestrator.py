"""Distillation orchestration: per-trajectory evidence extraction (map) and
synthesis (reduce), both under the same EvidenceScope, producing a skill
text plus a manifest. Real skills are produced only when the study owner
runs it on real Base data; tests use synthetic fixtures and label the
result test-only."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tilebench.llm.v2.distillation.access import EvidenceScope, TrajectoryRef
from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text
from tilebench.llm.v2.prompts.renderer import render_distill_extraction, render_distill_synthesis
from tilebench.llm.v2.providers.base import GenerationRequest


@dataclass
class DistillerConfig:
    model_id: str
    provider_name: str
    settings: dict
    dsl_version: str


def _rounds_block(state: dict, scope: EvidenceScope) -> str:
    lines = []
    for r in state.get("rounds", []):
        src = None
        if r.get("source_path"):
            p = Path(r["source_path"])
            if p.exists():
                src = p.read_text()
        lines.append(f"### Round {r['round']} — status {r['status']}")
        if r.get("latency_ms_mean") is not None:
            lines.append(f"runtime_ms: {r['latency_ms_mean']:.4f} (samples {r.get('latency_ms_samples')})")
        if r.get("config"):
            lines.append(f"config: {json.dumps(r['config'])}")
        if r.get("diagnostic"):
            lines.append(f"diagnostic: {r['diagnostic'][:2000]}")
        if src:
            lines.append(f"```python\n{src.rstrip()}\n```")
    return "\n".join(lines)


def extract_observations(scope: EvidenceScope, provider, cfg: DistillerConfig, *, read_state) -> list[dict]:
    """Map step: one request per selected trajectory. read_state(path) -> dict."""
    obs = []
    for ref in scope.selected:
        path = scope.open(ref)                      # raises for anything outside the scope
        state = read_state(path)
        if state["task"]["fold"] not in scope.training_folds or state["condition"] != "base":
            raise PermissionError(f"{ref.trajectory_id}: state disagrees with the scope")
        user = render_distill_extraction({
            "dsl": scope.dsl, "dsl_version": cfg.dsl_version, "source_device": scope.source_device,
            "training_folds": ", ".join(scope.training_folds), "held_out_fold": scope.held_out_fold or "(none: release mode)",
            "trajectory_id": ref.trajectory_id, "operator": ref.operator, "dtype": ref.dtype, "model": ref.model,
            "rounds_block": _rounds_block(state, scope),
        })
        res = provider.generate(GenerationRequest(system="You extract optimization evidence from one kernel-generation trajectory.",
                                                  user=user, model_id=cfg.model_id, provider=cfg.provider_name,
                                                  settings=cfg.settings, metadata={"trajectory": ref.trajectory_id}))
        obs.append({"trajectory_id": ref.trajectory_id, "operator": ref.operator, "text": res.text,
                    "usage": res.usage.to_dict()})
    return obs


def synthesize(scope: EvidenceScope, observations: list[dict], provider, cfg: DistillerConfig, *, models: list[str]) -> dict:
    user = render_distill_synthesis({
        "dsl": scope.dsl, "dsl_version": cfg.dsl_version, "source_device": scope.source_device,
        "training_folds": ", ".join(scope.training_folds), "held_out_fold": scope.held_out_fold or "(none: release mode)",
        "models": ", ".join(models), "n_trajectories": str(len(observations)),
        "trajectory_ids": ", ".join(o["trajectory_id"] for o in observations),
        "observations_block": "\n\n".join(f"### {o['trajectory_id']} ({o['operator']})\n{o['text']}" for o in observations),
    })
    res = provider.generate(GenerationRequest(system="You synthesize a conditional Optimization Skill from observations.",
                                              user=user, model_id=cfg.model_id, provider=cfg.provider_name,
                                              settings=cfg.settings, metadata={"synthesis": scope.dsl}))
    text = (res.text or "").strip() + "\n"
    manifest = {
        "schema": "tilebench-optimization-skill/1",
        "dsl": scope.dsl, "compatible_versions": [cfg.dsl_version],
        **scope.manifest(),
        "evaluation_or_release": scope.mode,
        "distiller_config_hash": sha256_text(canonical_json({"model_id": cfg.model_id, "settings": cfg.settings,
                                                             "templates": ["distill_evidence_extraction", "distill_synthesis"]})),
        "content_sha256": sha256_text(text),
        "status": "draft",            # frozen only by the study owner, before any Enhanced result is viewed
        "usage": {"extraction": [o["usage"] for o in observations], "synthesis": res.usage.to_dict()},
    }
    return {"text": text, "manifest": manifest}


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
