"""Non-paid end-to-end validation of the Trn2 NKI chain (protocol revision 4).

    python scripts/llm_v2/trn2_e2e_validation.py --campaign validation_trn2_rev4_e2e_<date> [--rounds 1]

The REAL campaign chain (preflight, task context and prompt with the registered Reference Skill and Device Context,
strict parser, compliance checker, NeuronSubprocessEvaluator in the docker/bwrap sandbox with the device lock, state
machine, trajectory persistence, resume) runs for vector_add (fp16, all 20 frozen cases) with a SCRIPTED provider
(providers.mock.MockProvider) that answers every request with the independent acceptance candidate
artifacts/llm_v2/preflight/Trn2/acceptance_candidate_vector_add_impl_nki.py. No API request is sent; run type
`validation` (unscored, never a distillation source)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tilebench.llm.v2.orchestration.campaign import CampaignSpec, prepare_track, run_track_task  # noqa: E402
from tilebench.llm.v2.providers.mock import MockProvider, scripted_text  # noqa: E402

CANDIDATE = REPO / "artifacts/llm_v2/preflight/Trn2/acceptance_candidate_vector_add_impl_nki.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--model", default="gpt")
    ap.add_argument("--isolation", default="auto")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    body = CANDIDATE.read_text()
    provider = MockProvider(script=[{"text": scripted_text("impl_nki.py", body)} for _ in range(5)])
    spec = CampaignSpec(name=a.campaign, run_type="validation", device="Trn2", dsl="nki", condition="base", model=a.model,
                        operators=["vector_add"], dtypes=["fp16"], stop_after_rounds=a.rounds, isolation=a.isolation,
                        resume=a.resume, executor="trn2_e2e_validation (scripted provider)")
    track = prepare_track(spec, provider=provider)
    if isinstance(track, dict):
        print(json.dumps(track, indent=1, default=str)[:6000])
        return 2
    print("isolation:", track.evaluator.report.get("backend"), "| arch:", track.arch, "| tasks:", [e.key.as_str() for e in track.tasks])
    for e in track.tasks:
        status = run_track_task(track, e)
        print(e.key.as_str(), "->", status)
    print("requests sent to the scripted provider:", len(provider.requests))
    if provider.requests:
        r = provider.requests[0]
        out = REPO / "artifacts/llm_v2/preflight/Trn2" / f"{a.campaign}_prompt_round1.json"
        out.write_text(json.dumps({"system": r.system, "user": r.user}, indent=1) + "\n")
        print("rendered prompt archived:", out.relative_to(REPO), "chars", len(r.system) + len(r.user))
    print(json.dumps({k: track.summary.get(k) for k in ("campaign", "run_type", "config_hash", "isolation")}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
