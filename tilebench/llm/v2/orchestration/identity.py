"""Campaign / trajectory identity and output layout.

outputs/llm_v2/<campaign>/<condition>/<device>/<dsl>/<model>/<operator>/<dtype>/<case_id>/
    trajectory.json, usage.jsonl, transport.jsonl,
    round_NN/attempt_M/{request.json, response.json, impl_<dsl>.py, compliance.json}
    round_NN/evaluation.json
    best_valid/impl_<dsl>.py

The legacy root tilebench/benchmarks/llm_generated/ is never written."""
from __future__ import annotations

from pathlib import Path

from tilebench.paths import OUTPUT_ROOT

from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text

LLM_V2_OUTPUT_ROOT = OUTPUT_ROOT / "llm_v2"


def trajectory_id(task: dict, model: str, condition: str) -> str:
    key = {k: task[k] for k in ("operator", "dtype", "case_id", "device", "dsl")}
    key.update({"model": model, "condition": condition})
    return sha256_text(canonical_json(key))[:20]


def trajectory_dir(campaign: str, task: dict, model: str, condition: str, root: Path | None = None) -> Path:
    root = root or LLM_V2_OUTPUT_ROOT
    return (root / campaign / condition / task["device"] / task["dsl"] / model /
            task["operator"] / task["dtype"] / task["case_id"])


def attempt_dir(tdir: Path, round_index: int, attempt: int) -> Path:
    return tdir / f"round_{round_index:02d}" / f"attempt_{attempt}"


def request_hash(system: str, user: str, model_id: str, settings: dict) -> str:
    return sha256_text(canonical_json({"system": system, "user": user, "model": model_id, "settings": settings}))
