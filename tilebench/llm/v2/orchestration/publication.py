"""Publication export: the Git-tracked, shareable copy of a campaign.

Policy (docs/llm_v2/ARTIFACT_POLICY.md): every round's request (the exact
prompt), response (provider result, ids, usage, terminal status), generated
implementation, compliance evidence, evaluation result and archived worker
evidence, the usage/transport ledgers and the trajectory state are exported
for every trajectory of a campaign — not only the final round or the winner.
Nothing is rewritten; files are copied byte for byte and indexed by sha256.

Permission rule: a trajectory whose injected context includes a component
that is not `publishable` in skills/manifest.json is withheld entirely
(prompts contain the component text, and responses may quote it); it is
listed in REDACTIONS.json with its content hashes. Large caches never exist
in a trajectory directory (they live in the campaign's _sandbox, which is
not exported)."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from tilebench.llm.v2.orchestration.state import TrajectoryState
from tilebench.llm.v2.skills.loader import is_publishable

SKIP_NAMES = {"trajectory.tmp"}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def component_entries(manifest: dict, content_hashes: dict) -> dict:
    """{component key: manifest entry or None} for the skill components of a
    trajectory (keys look like 'reference:triton@3.6.0')."""
    out = {}
    for key in content_hashes:
        if key in ("contract", "templates") or ":" not in key:
            continue
        kind, rest = key.split(":", 1)
        name, _, version = rest.rpartition("@")
        out[key] = manifest.get(kind, {}).get(name, {}).get(version)
    return out


def withheld_reason(manifest: dict, state: TrajectoryState) -> str | None:
    for key, entry in component_entries(manifest, state.content_hashes).items():
        if entry is None:
            return f"component {key} is not registered in skills/manifest.json"
        if not is_publishable(entry):
            return f"component {key} is not publishable (permission {entry.get('permission')}, publishable={entry.get('publishable', False)})"
    return None


def export_publication(campaign_dir: Path, out_dir: Path, *, manifest: dict) -> dict:
    campaign_dir = Path(campaign_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"schema": "tilebench-llm-v2-publication/1", "campaign_dir": str(campaign_dir),
                   "exported": time.time(), "files": {}, "trajectories": [], "withheld": []}
    redactions: list[dict] = []
    for src in sorted(list(campaign_dir.glob("campaign*.json")) + list(campaign_dir.glob("campaign_runs.jsonl"))
                      + list(campaign_dir.glob("summary_*.json"))):
        dst = out_dir / src.name
        shutil.copyfile(src, dst)
        index["files"][src.name] = {"sha256": _sha256(dst), "bytes": dst.stat().st_size}
    rows = []
    for tpath in sorted(campaign_dir.rglob("trajectory.json")):
        tdir = tpath.parent
        if "_sandbox" in tdir.parts:
            continue
        state = TrajectoryState.load(tpath)
        rel = tdir.relative_to(campaign_dir)
        why = withheld_reason(manifest, state)
        if why:
            redactions.append({"trajectory_id": state.trajectory_id, "path": str(rel), "reason": why,
                               "content_hashes": state.content_hashes, "task": state.task, "model": state.model,
                               "condition": state.condition, "status": state.status})
            index["withheld"].append(state.trajectory_id)
            continue
        n = 0
        for f in sorted(tdir.rglob("*")):
            if not f.is_file() or f.name in SKIP_NAMES:
                continue
            frel = f.relative_to(campaign_dir)
            dst = out_dir / frel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            index["files"][str(frel)] = {"sha256": _sha256(dst), "bytes": dst.stat().st_size}
            n += 1
        costs = [a.cost for r in state.rounds for a in r.attempts]
        row = {"trajectory_id": state.trajectory_id, "path": str(rel), "task": state.task, "model": state.model,
               "condition": state.condition, "run_type": state.run_type, "status": state.status,
               "rounds_closed": sum(1 for r in state.rounds if r.status != "pending"),
               "valid_rounds": len(state.valid_rounds()), "attempts": sum(len(r.attempts) for r in state.rounds),
               "best_valid": state.best_valid, "cumulative_tokens": None if any(c is None for c in costs) else sum(costs),
               "files": n, "content_hashes": state.content_hashes, "generator": state.generator}
        index["trajectories"].append(row)
        rows.append(row)
    (out_dir / "REDACTIONS.json").write_text(json.dumps({"withheld_trajectories": redactions,
                                                         "rule": "a trajectory with any non-publishable component is withheld entirely"},
                                                        indent=1, sort_keys=True) + "\n")
    (out_dir / "INDEX.json").write_text(json.dumps(index, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "SUMMARY.md").write_text(summary_markdown(rows, redactions))
    return {"out": str(out_dir), "trajectories": len(rows), "withheld": len(redactions), "files": len(index["files"])}


def summary_markdown(rows: list[dict], redactions: list[dict]) -> str:
    lines = ["# Publication summary", "",
             "| trajectory | task | model | condition | run_type | status | rounds | valid | attempts | best ms | tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["task"]
        best = (r["best_valid"].get("latency_ms_geomean") or r["best_valid"].get("latency_ms_mean")) if r.get("best_valid") else None
        lines.append(f"| {r['trajectory_id']} | {t['device']}/{t['dsl']}/{t['operator']}/{t['dtype']} | {r['model']} | "
                     f"{r['condition']} | {r['run_type']} | {r['status']} | {r['rounds_closed']} | {r['valid_rounds']} | "
                     f"{r['attempts']} | {best if best is None else f'{best:.4f}'} | {r['cumulative_tokens']} |")
    if redactions:
        lines += ["", "## Withheld trajectories", ""]
        for x in redactions:
            lines.append(f"- {x['trajectory_id']} ({x['path']}): {x['reason']}")
    return "\n".join(lines) + "\n"
