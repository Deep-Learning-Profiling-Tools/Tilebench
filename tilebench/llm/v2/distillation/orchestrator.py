"""Distillation orchestration: per-trajectory evidence extraction (map) and
synthesis (reduce), both under the same EvidenceScope, producing a skill
text plus a manifest.

Binding, layer by layer (all refused by code):
1. the index is built from the campaign root with file hashes and the
   frozen fold manifest (access.build_index);
2. EvidenceScope.open re-verifies hash, root containment and symlinks;
3. verify_state_identity checks that the state's own fields (trajectory id
   recomputed from task/model/condition, dsl, device, operator, dtype, fold
   recomputed from the manifest, model, condition, run_type, schema,
   completeness) equal the reference;
4. candidate sources are read only from under the trajectory's own
   directory, and their hashes are recorded;
5. every observation carries the scope hash, the state hash, the material
   rules version and the request hash it was extracted under; synthesis
   refuses observations from another scope, duplicates, or observations
   whose source files changed.

Persistence (append-only): observations/<trajectory_id>.json (+ .request.md
and .raw.json with the full request text and the provider's raw response);
synthesis.json (+ synthesis.request.md, synthesis.raw.json). An existing
ACCEPTED record with matching identity is reused without a request; a
partial record (truncated or interrupted response) keeps its cost, is
renamed *.partial_N.json and is never an observation or a skill.

Materials given to the distiller per trajectory (MATERIAL_RULES, versioned):
every round, every attempt in order: verdict, tokens, the candidate source,
the compliance diagnostics, the unified diff from the previous attempt's
source, the evaluation outcome with raw latency samples and execution mode,
plus the offline SOL record (T_SOL, mode, status) and per-round
SOL-efficiency when the caller supplies it. Nothing from other trajectories,
other devices or held-out results is ever selected.

Real skills are produced only when the study owner runs this on formal Base
data with an approved distiller and frozen folds; everything else is written
as test-only and never registered in a production manifest."""
from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from tilebench.llm.v2.distillation.access import ACCEPTED_STATE_SCHEMAS, EvidenceAccessError, EvidenceScope, TrajectoryRef
from tilebench.llm.v2.manifests.schema import ManifestError, canonical_json, fold_of, sha256_text
from tilebench.llm.v2.orchestration.identity import trajectory_id as compute_trajectory_id
from tilebench.llm.v2.prompts.renderer import load_template, render_distill_extraction, render_distill_synthesis
from tilebench.llm.v2.providers.base import GenerationRequest

EXTRACTION_SYSTEM = "You extract optimization evidence from one kernel-generation trajectory."
SYNTHESIS_SYSTEM = "You synthesize a conditional Optimization Skill from observations."
TEMPLATES = ("distill_evidence_extraction", "distill_synthesis")
MATERIAL_RULES = {"version": "distill-material/2", "order": "round ascending, attempt ascending",
                  "per_attempt": ["verdict", "tokens", "source", "compliance_diagnostics", "diff_from_previous_attempt"],
                  "per_round": ["status", "valid_cases/cases_total", "latency_ms_geomean", "per-case status/runtime/efficiency/config",
                                "timing_execution_mode", "diagnostic"],
                  "max_source_chars": 20000, "max_diagnostic_chars": 2000, "diff_context_lines": 3,
                  "selection": "every attempt of every round of the selected trajectory; no sampling, no ranking"}
COMPLETE_TERMINAL = ("completed", "end_turn", "stop_sequence", None)


class DistillationIncomplete(RuntimeError):
    """A map or reduce response was truncated or interrupted: its cost is
    recorded, the partial text is kept as a partial record, nothing is
    accepted as an observation or a skill."""


@dataclass
class DistillerConfig:
    model_id: str
    provider_name: str
    settings: dict
    dsl_version: str

    def config_hash(self) -> str:
        payload = {"model_id": self.model_id, "provider": self.provider_name, "settings": self.settings,
                   "system_prompts": {"extraction": EXTRACTION_SYSTEM, "synthesis": SYNTHESIS_SYSTEM},
                   "templates": {name: sha256_text(load_template(name)) for name in TEMPLATES},
                   "material_rules": MATERIAL_RULES}
        return sha256_text(canonical_json(payload))


def verify_state_identity(ref: TrajectoryRef, state: dict, folds: dict | None = None) -> None:
    t = state.get("task", {})
    expect = {"trajectory_id": ref.trajectory_id, "dsl": ref.dsl, "device": ref.device, "operator": ref.operator,
              "dtype": ref.dtype, "model": ref.model, "condition": ref.condition, "run_type": ref.run_type}
    actual = {"trajectory_id": state.get("trajectory_id"), "dsl": t.get("dsl"), "device": t.get("device"),
              "operator": t.get("operator"), "dtype": t.get("dtype"), "model": state.get("model"),
              "condition": state.get("condition"), "run_type": state.get("run_type", "formal")}
    diff = {k: (expect[k], actual[k]) for k in expect if expect[k] != actual[k]}
    if diff:
        raise EvidenceAccessError(f"{ref.trajectory_id}: state identity disagrees with the index: {diff}")
    if folds is not None:
        try:
            manifest_fold = fold_of(folds, t.get("operator"))
        except ManifestError as e:
            raise EvidenceAccessError(f"{ref.trajectory_id}: {e}")
        if manifest_fold != ref.fold:
            raise EvidenceAccessError(f"{ref.trajectory_id}: frozen fold manifest says {manifest_fold}, index says {ref.fold}")
        if t.get("fold") != manifest_fold:
            raise EvidenceAccessError(f"{ref.trajectory_id}: state fold label {t.get('fold')!r} disagrees with the frozen fold manifest ({manifest_fold})")
    if state.get("schema") not in ACCEPTED_STATE_SCHEMAS:
        raise EvidenceAccessError(f"{ref.trajectory_id}: state schema {state.get('schema')!r} not accepted")
    if state.get("status") != "complete":
        raise EvidenceAccessError(f"{ref.trajectory_id}: trajectory status {state.get('status')!r} is not complete")
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


def _clip(text: str | None, n: int) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[:n] + f"\n[... clipped to {n} chars by {MATERIAL_RULES['version']}]"


def rounds_block(state: dict, tdir: Path | None, sol: dict | None = None) -> tuple[str, dict]:
    """Model-visible materials for the extraction prompt (every attempt of
    every round, in order) plus the hashes of every source embedded."""
    lines = []
    hashes: dict = {}
    t_sol = (sol or {}).get("t_sol_ms")
    if sol and sol.get("case_t_emp_ms"):
        lines.append(f"Empirical targets (offline): T_emp for each of the {len(sol['case_t_emp_ms'])} cases, status {sol.get('status')}")
    elif sol:
        lines.append(f"SOL record (offline, declared mode {sol.get('arithmetic_mode')}, status {sol.get('status')}): "
                     f"T_SOL = {t_sol} ms; P_peak = {sol.get('p_peak_tflops')} TFLOP/s; BW_peak = {sol.get('bw_peak_gbs')} GB/s")
    prev_src = None
    for r in state.get("rounds", []):
        lines.append(f"### Round {r['round']} — status {r['status']}")
        for a in r.get("attempts", []):
            lines.append(f"#### Attempt {a.get('attempt')} ({a.get('kind')}): verdict {a.get('verdict')}, "
                         f"tokens {a.get('cost')} ({a.get('cost_status', 'known')})")
            diags = (a.get("compliance") or {}).get("diagnostics") or []
            if diags:
                lines.append("compliance: " + _clip("; ".join(diags), MATERIAL_RULES["max_diagnostic_chars"]))
            elif a.get("diagnostic") and a.get("verdict") != "clear":
                lines.append("diagnostic: " + _clip(a["diagnostic"], MATERIAL_RULES["max_diagnostic_chars"]))
            src, h = _source_text(a.get("source_path"), tdir)
            if src is not None:
                hashes[f"round_{r['round']}_attempt_{a.get('attempt')}"] = h
                if prev_src is not None:
                    d = list(difflib.unified_diff(prev_src.splitlines(), src.splitlines(), fromfile="previous", tofile="this",
                                                  lineterm="", n=MATERIAL_RULES["diff_context_lines"]))
                    lines.append("diff from the previous attempt:\n```diff\n" + _clip("\n".join(d), MATERIAL_RULES["max_source_chars"]) + "\n```")
                lines.append("```python\n" + _clip(src.rstrip(), MATERIAL_RULES["max_source_chars"]) + "\n```")
                prev_src = src
        if r.get("cases_total"):
            # revision 4: the round's 20-case evidence (offline, source device only; never shown to a generator)
            case_t = (sol or {}).get("case_t_emp_ms") or {}
            head = f"cases: {r.get('valid_cases')}/{r.get('cases_total')} valid"
            if r.get("latency_ms_geomean") is not None:
                head += f"; geometric-mean runtime {r['latency_ms_geomean']:.4f} ms (mode {r.get('timing_execution_mode')})"
                effs = [case_t[c['case_id']] / c['latency_ms_mean'] for c in r.get("case_results") or []
                        if c.get("status") == "valid" and case_t.get(c["case_id"]) and c.get("latency_ms_mean")]
                if effs and len(effs) == r.get("cases_total"):
                    import math
                    head += f"; geomean empirical efficiency {math.exp(sum(math.log(e) for e in effs) / len(effs)):.4f}"
            lines.append(head)
            for c in r.get("case_results") or []:
                lat = c.get("latency_ms_mean")
                eff_c = (case_t.get(c["case_id"]) / lat) if (lat and case_t.get(c["case_id"])) else None
                lines.append(f"  case {c.get('case_index')}: status {c.get('status')}"
                             + (f", runtime {lat:.4f} ms" if lat else "") + (f", efficiency {eff_c:.4f}" if eff_c else "")
                             + (f", config {json.dumps(c.get('config'))}" if c.get("config") else ""))
        elif r.get("latency_ms_mean") is not None:
            lines.append(f"runtime_ms: {r['latency_ms_mean']:.4f} (samples {r.get('latency_ms_samples')}; "
                         f"mode {r.get('timing_execution_mode')})")
            if t_sol:
                lines.append(f"sol_efficiency: {float(t_sol) / float(r['latency_ms_mean']):.4f}")
        if r.get("config"):
            lines.append(f"config: {json.dumps(r['config'])}")
        if r.get("diagnostic") and r.get("status") != "valid":
            lines.append("evaluation: " + _clip(str(r["diagnostic"]), MATERIAL_RULES["max_diagnostic_chars"]))
    return "\n".join(lines), hashes


def _accepted(res) -> bool:
    return bool(res.text and res.text.strip()) and not getattr(res, "truncated", False) \
        and getattr(res, "terminal_status", None) in COMPLETE_TERMINAL


def _archive_exchange(out_dir: Path, stem: str, system: str, user: str, res) -> None:
    (out_dir / f"{stem}.request.md").write_text(f"# system\n\n{system}\n\n# user\n\n{user}\n")
    (out_dir / f"{stem}.raw.json").write_text(json.dumps({"text": res.text, "raw_response": res.raw_response,
                                                          "usage": res.usage.to_dict(), "usage_raw": res.usage_raw,
                                                          "response_id": res.response_id, "model_id": res.model_id,
                                                          "terminal_status": res.terminal_status, "truncated": res.truncated,
                                                          "transport_attempts": res.transport_attempts, "elapsed_s": res.elapsed_s,
                                                          "error": res.error}, indent=1, default=str) + "\n")


def _rotate_partial(path: Path) -> None:
    if path.exists():
        n = 1
        while path.with_name(f"{path.stem}.partial_{n}{path.suffix}").exists():
            n += 1
        path.rename(path.with_name(f"{path.stem}.partial_{n}{path.suffix}"))


def extract_observations(scope: EvidenceScope, provider, cfg: DistillerConfig, *, read_state,
                         out_dir: Path | None = None, sol_info: dict | None = None, folds: dict | None = None,
                         log=lambda s: None) -> list[dict]:
    """Map step: one request per selected trajectory. read_state(path) -> dict.
    Raises DistillationIncomplete when a response is truncated/interrupted
    (its cost and partial text are persisted as a partial record)."""
    obs = []
    scope_hash = scope.sha256()
    cfg_hash = cfg.config_hash()
    for ref in scope.selected:
        path = scope.open(ref)                      # raises for anything outside the scope
        state = read_state(path)
        verify_state_identity(ref, state, folds)
        if state["task"]["fold"] not in scope.training_folds and scope.mode == "evaluation" or state["condition"] != "base":
            raise EvidenceAccessError(f"{ref.trajectory_id}: state disagrees with the scope")
        block, src_hashes = rounds_block(state, path.parent, (sol_info or {}).get(ref.trajectory_id))
        user = render_distill_extraction({
            "dsl": scope.dsl, "dsl_version": cfg.dsl_version, "source_device": scope.source_device,
            "training_folds": ", ".join(scope.training_folds), "held_out_fold": scope.held_out_fold or "(none: release mode)",
            "trajectory_id": ref.trajectory_id, "operator": ref.operator, "dtype": ref.dtype, "model": ref.model,
            "rounds_block": block,
        })
        req_hash = sha256_text(EXTRACTION_SYSTEM + "\0" + user)
        identity = {"scope_sha256": scope_hash, "state_sha256": ref.state_sha256, "distiller_config_hash": cfg_hash,
                    "request_sha256": req_hash, "source_sha256": src_hashes, "material_rules": MATERIAL_RULES["version"]}
        op = (out_dir / "observations" / f"{ref.trajectory_id}.json") if out_dir is not None else None
        if op is not None and op.exists():
            prev = json.loads(op.read_text())
            if prev.get("accepted") and all(prev.get(k) == v for k, v in identity.items()):
                obs.append(prev)
                log(f"{ref.trajectory_id}: observation reused (no request)")
                continue
            _rotate_partial(op)
        res = provider.generate(GenerationRequest(system=EXTRACTION_SYSTEM, user=user, model_id=cfg.model_id,
                                                  provider=cfg.provider_name, settings=cfg.settings,
                                                  metadata={"trajectory": ref.trajectory_id}))
        rec = {"trajectory_id": ref.trajectory_id, "operator": ref.operator, "dtype": ref.dtype, "model": ref.model,
               "text": res.text, "usage": res.usage.to_dict(), "response_id": res.response_id, "model_id": res.model_id,
               "terminal_status": res.terminal_status, "truncated": res.truncated, "transport_attempts": res.transport_attempts,
               "accepted": _accepted(res), "recorded_at": time.time(), **identity}
        if op is not None:
            op.parent.mkdir(parents=True, exist_ok=True)
            _archive_exchange(op.parent, ref.trajectory_id, EXTRACTION_SYSTEM, user, res)
            op.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
        if not rec["accepted"]:
            if op is not None:
                _rotate_partial(op)
            raise DistillationIncomplete(f"{ref.trajectory_id}: extraction response not complete "
                                         f"(terminal_status {res.terminal_status!r}, truncated {res.truncated}); cost recorded, not an observation")
        obs.append(rec)
        log(f"{ref.trajectory_id}: observation extracted ({res.usage.logical_total} tokens)")
    return obs


def synthesis_identity(scope: EvidenceScope, observations: list[dict], cfg: DistillerConfig, models: list[str]) -> str:
    payload = {"scope_sha256": scope.sha256(), "observations": sorted((o["trajectory_id"], sha256_text(o.get("text") or ""))
                                                                        for o in observations),
               "distiller_config_hash": cfg.config_hash(), "models": sorted(models)}
    return sha256_text(canonical_json(payload))


def synthesize(scope: EvidenceScope, observations: list[dict], provider, cfg: DistillerConfig, *,
               models: list[str], out_dir: Path | None = None, log=lambda s: None) -> dict:
    scope_hash = scope.sha256()
    seen = set()
    for o in observations:
        if o.get("scope_sha256") != scope_hash:
            raise EvidenceAccessError(f"observation {o.get('trajectory_id')} was extracted under another scope")
        if not o.get("accepted", True):
            raise EvidenceAccessError(f"observation {o.get('trajectory_id')} is a partial record, not an accepted observation")
        if o["trajectory_id"] in seen:
            raise EvidenceAccessError(f"duplicate observation for {o['trajectory_id']}")
        seen.add(o["trajectory_id"])
    selected = {r.trajectory_id for r in scope.selected}
    extra = seen - selected
    if extra:
        raise EvidenceAccessError(f"observations outside the scope: {sorted(extra)}")
    missing = selected - seen
    if missing:
        raise EvidenceAccessError(f"observations missing for selected trajectories: {sorted(missing)}")
    ident = synthesis_identity(scope, observations, cfg, models)
    spath = (out_dir / "synthesis.json") if out_dir is not None else None
    if spath is not None and spath.exists():
        prev = json.loads(spath.read_text())
        if prev.get("accepted") and prev.get("manifest", {}).get("synthesis_identity") == ident:
            log("synthesis reused (no request)")
            return prev
        _rotate_partial(spath)
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
    accepted = _accepted(res)
    text = (res.text or "").strip() + "\n"
    manifest = {
        "schema": "tilebench-optimization-skill/2",
        "dsl": scope.dsl, "compatible_versions": [cfg.dsl_version],
        **scope.manifest(),
        "scope_sha256": scope_hash,
        "synthesis_identity": ident,
        "evaluation_or_release": scope.mode,
        "distiller": {"model_id": cfg.model_id, "provider": cfg.provider_name, "settings": cfg.settings,
                      "response_id": res.response_id, "terminal_status": res.terminal_status, "truncated": res.truncated},
        "distiller_config_hash": cfg.config_hash(),
        "material_rules": MATERIAL_RULES,
        "observation_sha256": {o["trajectory_id"]: sha256_text(o["text"] or "") for o in observations},
        "content_sha256": sha256_text(text),
        "status": "draft" if accepted else "partial",   # frozen only by the study owner, before any Enhanced result is viewed
        "usage": {"extraction": [o["usage"] for o in observations], "synthesis": res.usage.to_dict()},
    }
    result = {"text": text, "manifest": manifest, "accepted": accepted, "request_sha256": sha256_text(SYNTHESIS_SYSTEM + "\0" + user)}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _archive_exchange(out_dir, "synthesis", SYNTHESIS_SYSTEM, user, res)
        spath.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        if not accepted:
            _rotate_partial(spath)
    if not accepted:
        raise DistillationIncomplete(f"synthesis response not complete (terminal_status {res.terminal_status!r}, "
                                     f"truncated {res.truncated}); cost recorded, no skill produced")
    return result


def write_skill(out_dir: Path, result: dict, *, test_only: bool) -> None:
    if not result.get("accepted", True):
        raise DistillationIncomplete("refusing to write a skill from a partial synthesis")
    out_dir.mkdir(parents=True, exist_ok=True)
    text = result["text"]
    manifest = dict(result["manifest"])
    if test_only:
        text = "<!-- TEST-ONLY synthetic Optimization Skill; never referenced by a production manifest -->\n" + text
        manifest["status"] = "test-only"
        manifest["content_sha256"] = sha256_text(text)
    (out_dir / "SKILL.md").write_text(text)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
