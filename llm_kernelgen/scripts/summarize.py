"""Summarise all evaluation results for an experiment.

Scans all sample directories under ``llm_kernelgen/generated/<experiment_id>/``
and aggregates per-operator, per-backend metrics into a paper-ready summary.

Metrics reported
----------------
Generation quality
  * syntax_pass_rate      - fraction of samples that passed syntax/import check
  * compile_pass_rate     - fraction that compiled (Triton JIT / cuTile JIT)
  * runtime_pass_rate     - fraction where run() executed without exception
  * correctness_pass_rate - fraction that passed correctness on all cases
  * repair_success_rate   - fraction of failed samples repaired within R rounds

Performance (correctness-passing kernels only)
  * mean_speedup_vs_torch
  * median_speedup_vs_torch
  * pass_at_1, pass_at_k   (k = num_samples per operator)

Effort / cost
  * mean_prompt_tokens
  * mean_completion_tokens
  * mean_reasoning_tokens
  * mean_latency_s
  * mean_loc
  * total_api_calls

Usage
-----
::

    PYTHONPATH=. python llm_kernelgen/scripts/summarize.py \\
        --experiment exp_main_triton_zero_shot \\
        --output results/llm_gen/exp_main_summary.json

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LLM_ROOT  = Path(__file__).resolve().parents[1]
_GEN_ROOT  = _LLM_ROOT / "generated"

sys.path.insert(0, str(_REPO_ROOT))

STAGE_ORDER = [
    "syntax_import",
    "run_case0",
    "correct_case0",
    "correct_all",
    "bench_complete",
    "perf_score",
]


def _safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def _safe_median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    n = len(vals)
    if not n:
        return None
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021).

    n = total samples, c = correct samples, k = pass@k target.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def _load_sample(sample_dir: Path) -> dict[str, Any]:
    """Load all artefact files from one sample directory."""
    record: dict[str, Any] = {"sample_dir": str(sample_dir)}

    meta_path = sample_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as fh:
            record["metadata"] = json.load(fh)

    status_path = sample_dir / "eval_status.json"
    if status_path.exists():
        with status_path.open() as fh:
            record["eval_status"] = json.load(fh)

    bench_path = sample_dir / "bench.json"
    if bench_path.exists():
        with bench_path.open() as fh:
            record["bench"] = json.load(fh)

    repair_path = sample_dir / "repair_summary.json"
    if repair_path.exists():
        with repair_path.open() as fh:
            record["repair"] = json.load(fh)

    return record


def summarize_experiment(
    experiment_id: str,
    gen_root: Path | None = None,
) -> dict[str, Any]:
    """Aggregate all sample results for *experiment_id*.

    Parameters
    ----------
    experiment_id:
        Subdirectory name under ``llm_kernelgen/generated/``.
    gen_root:
        Override path for the ``generated/`` root directory.

    Returns
    -------
    dict
        Nested summary dict ready to be serialised as JSON.
    """
    root = (gen_root or _GEN_ROOT) / experiment_id
    if not root.exists():
        raise FileNotFoundError(f"Experiment directory not found: {root}")

    # Discover all sample directories.
    # Structure: <root>/<operator>/<backend>/sample_<N>/
    all_samples: list[dict[str, Any]] = []
    for op_dir in sorted(root.iterdir()):
        if not op_dir.is_dir():
            continue
        for backend_dir in sorted(op_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            for sample_dir in sorted(backend_dir.glob("sample_*")):
                if not sample_dir.is_dir():
                    continue
                sample = _load_sample(sample_dir)
                sample["operator"] = op_dir.name
                sample["backend"]  = backend_dir.name
                all_samples.append(sample)

    if not all_samples:
        return {
            "experiment_id": experiment_id,
            "warning": "No sample directories found.",
            "per_operator": {},
            "aggregate": {},
        }

    # Per-operator, per-backend aggregation.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in all_samples:
        groups[(s["operator"], s["backend"])].append(s)

    per_operator: dict[str, dict] = {}
    agg_metrics: dict[str, list] = defaultdict(list)

    for (op, backend), samples in sorted(groups.items()):
        n = len(samples)
        meta_list = [s.get("metadata", {}) for s in samples]
        eval_list = [s.get("eval_status", {}) for s in samples]

        # Generation quality
        syntax_ok   = sum(1 for e in eval_list if e.get("stage_reached", -1) >= 0)
        runtime_ok  = sum(1 for e in eval_list if e.get("stage_reached", -1) >= 1)
        correct_ok  = sum(1 for e in eval_list if e.get("success", False))

        # Repair
        repaired_count = 0
        for s in samples:
            repair = s.get("repair", {})
            if not s.get("eval_status", {}).get("success") and repair.get("repaired"):
                repaired_count += 1
        failed_before_repair = n - correct_ok
        repair_success_rate  = (
            repaired_count / failed_before_repair
            if failed_before_repair > 0 else None
        )

        # pass@1 and pass@k
        pass1 = correct_ok / n if n else 0.0
        passk = _pass_at_k(n, correct_ok, min(n, 5))

        # Speedups (correctness-passing samples only).
        speedups = []
        for s in samples:
            if s.get("eval_status", {}).get("success"):
                bench = s.get("bench", {})
                if isinstance(bench, list):
                    for r in bench:
                        for bk in ("triton", "cutile"):
                            sp = r.get(f"speedup_{bk}")
                            if sp and sp > 0:
                                speedups.append(sp)

        # Token / cost metrics.
        prompt_toks     = [m.get("prompt_tokens", 0) for m in meta_list]
        completion_toks = [m.get("completion_tokens", 0) for m in meta_list]
        reasoning_toks  = [m.get("reasoning_tokens", 0) for m in meta_list]
        latency_s       = [m.get("gen_latency_s", 0.0) for m in meta_list]
        locs            = [m.get("loc", 0) for m in meta_list]

        entry = {
            "n_samples":            n,
            "syntax_pass_rate":     syntax_ok / n,
            "runtime_pass_rate":    runtime_ok / n,
            "correctness_pass_rate": correct_ok / n,
            "repair_success_rate":  repair_success_rate,
            "pass_at_1":            pass1,
            "pass_at_k":            passk,
            "mean_speedup_vs_torch":   _safe_mean(speedups),
            "median_speedup_vs_torch": _safe_median(speedups),
            "mean_prompt_tokens":      _safe_mean(prompt_toks),
            "mean_completion_tokens":  _safe_mean(completion_toks),
            "mean_reasoning_tokens":   _safe_mean(reasoning_toks),
            "mean_latency_s":          _safe_mean(latency_s),
            "mean_loc":                _safe_mean(locs),
        }
        per_operator.setdefault(op, {})[backend] = entry

        # Collect for aggregate.
        for key in ("syntax_pass_rate", "runtime_pass_rate",
                    "correctness_pass_rate", "pass_at_1"):
            agg_metrics[key].append(entry[key])
        if entry["mean_speedup_vs_torch"] is not None:
            agg_metrics["mean_speedup_vs_torch"].append(entry["mean_speedup_vs_torch"])
        agg_metrics["mean_prompt_tokens"].append(_safe_mean(prompt_toks) or 0)
        agg_metrics["mean_completion_tokens"].append(_safe_mean(completion_toks) or 0)

    aggregate = {k: _safe_mean(v) for k, v in agg_metrics.items()}
    aggregate["total_samples"] = len(all_samples)

    return {
        "experiment_id": experiment_id,
        "per_operator":  per_operator,
        "aggregate":     aggregate,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise LLM kernel generation experiment results.",
    )
    parser.add_argument("--experiment", required=True,
                        help="Experiment ID (subdirectory of llm_kernelgen/generated/).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <gen_root>/<exp>/summary.json).")
    parser.add_argument("--gen-root", default=None,
                        help="Override path for the generated/ root directory.")
    args = parser.parse_args()

    gen_root = Path(args.gen_root) if args.gen_root else None
    summary  = summarize_experiment(args.experiment, gen_root=gen_root)

    out_root = (gen_root or _GEN_ROOT) / args.experiment
    out_path = Path(args.output) if args.output else out_root / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Summary written to {out_path}")

    # Pretty-print aggregate metrics.
    agg = summary.get("aggregate", {})
    print("\n=== Aggregate metrics ===")
    for k, v in sorted(agg.items()):
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")


if __name__ == "__main__":
    main()
