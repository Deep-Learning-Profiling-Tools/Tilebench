"""Synthetic trajectory fixtures for distillation tests (test-only)."""
from __future__ import annotations

import json
from pathlib import Path

from tilebench.llm.v2.distillation.access import TrajectoryRef
from tilebench.llm.v2.manifests.schema import fold_of


def synthetic_index(root: Path, folds: dict, *, dsl: str, devices: list[str], models: list[str],
                    operators: list[str], conditions=("base", "enhanced")) -> list[TrajectoryRef]:
    root.mkdir(parents=True, exist_ok=True)
    refs = []
    for dev in devices:
        for model in models:
            for cond in conditions:
                for op in operators:
                    tid = f"{dsl}-{dev}-{model}-{cond}-{op}"
                    path = root / f"{tid}.json"
                    state = {"trajectory_id": tid, "model": model, "condition": cond,
                             "task": {"operator": op, "dtype": "fp16", "dsl": dsl, "device": dev, "fold": fold_of(folds, op)},
                             "rounds": [{"round": 1, "status": "valid", "latency_ms_mean": 1.0, "latency_ms_samples": [1, 1, 1],
                                         "config": {"BLOCK": 64}, "source_path": None, "diagnostic": None}]}
                    path.write_text(json.dumps(state))
                    refs.append(TrajectoryRef(tid, dsl, dev, op, "fp16", model, cond, fold_of(folds, op), str(path)))
    return refs
