"""Batch generate, evaluate, and (optionally) repair LLM kernels.

This is the top-level driver for a full experiment run.  It:

1. Loads an experiment config from ``llm_kernelgen/configs/experiments.yaml``.
2. Generates *num_samples* independent kernels per operator × backend.
3. Evaluates each generated kernel via the TileBench engine (subprocess).
4. Optionally runs up to *repair_rounds* repair iterations for failing samples.
5. Writes a summary JSON to ``llm_kernelgen/generated/<experiment_id>/summary.json``.

Usage
-----
::

    # Run a named experiment from experiments.yaml.
    PYTHONPATH=. python scripts/eval_llm_kernels.py \\
        --experiment smoke_triton \\
        --models-yaml llm_kernelgen/configs/models.yaml

    # Override specific parameters inline.
    PYTHONPATH=. python scripts/eval_llm_kernels.py \\
        --experiment smoke_triton \\
        --operators softmax rmsnorm \\
        --samples 3 \\
        --repair-rounds 2

    # Evaluate only (skip generation, re-evaluate existing files).
    PYTHONPATH=. python scripts/eval_llm_kernels.py \\
        --experiment smoke_triton \\
        --eval-only

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT  = _REPO_ROOT / "llm_kernelgen"
_GEN_ROOT  = _LLM_ROOT / "generated"
_CFG_DIR   = _LLM_ROOT / "configs"

sys.path.insert(0, str(_REPO_ROOT))

from llm_kernelgen.runtime.adapters import client_from_config, load_model_configs
from llm_kernelgen.scripts.evaluate import evaluate_via_subprocess
from llm_kernelgen.scripts.generate import generate_one
from llm_kernelgen.scripts.repair import repair_one
from llm_kernelgen.scripts.summarize import summarize_experiment
from llm_kernelgen.scripts.build_prompt import load_system_prompt


def _load_experiment_cfg(exp_id: str, exps_yaml: Path) -> dict[str, Any]:
    with exps_yaml.open() as fh:
        raw = yaml.safe_load(fh)
    exps = raw.get("experiments", raw)
    if exp_id not in exps:
        raise KeyError(
            f"Experiment '{exp_id}' not found in {exps_yaml}.  "
            f"Available: {list(exps.keys())}"
        )
    return exps[exp_id]


def _load_profile(profile_name: str) -> dict[str, Any]:
    profiles_yaml = _CFG_DIR / "prompt_profiles.yaml"
    with profiles_yaml.open() as fh:
        raw = yaml.safe_load(fh)
    profiles = raw.get("profiles", raw)
    if profile_name not in profiles:
        raise KeyError(
            f"Profile '{profile_name}' not found in prompt_profiles.yaml."
        )
    return profiles[profile_name]


def _load_operators_split(split_name: str) -> list[str]:
    """Load operator list from dataset split files."""
    split_map = {
        "train_examples": _LLM_ROOT / "dataset" / "train_examples.yaml",
        "dev_ops":        _LLM_ROOT / "dataset" / "dev_ops.yaml",
        "test_ops":       _LLM_ROOT / "dataset" / "test_ops.yaml",
    }
    if split_name not in split_map:
        raise ValueError(
            f"Unknown operators_split '{split_name}'.  "
            f"Expected one of: {list(split_map.keys())}"
        )
    path = split_map[split_name]
    with path.open() as fh:
        data = yaml.safe_load(fh)
    ops = data.get("operators", [])
    # Support list of dicts (train_examples.yaml) or plain list.
    if ops and isinstance(ops[0], dict):
        return list(ops[0].keys()) if isinstance(ops[0], dict) else []
    if isinstance(ops, dict):
        return list(ops.keys())
    return list(ops)


def run_experiment(
    *,
    experiment_id: str,
    exp_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    client: Any,
    profile: dict[str, Any],
    operators: list[str],
    backends: list[str],
    num_samples: int,
    repair_rounds: int,
    eval_only: bool = False,
) -> dict[str, Any]:
    """Execute the full generation → evaluation → repair pipeline."""
    results: list[dict] = []

    for operator in operators:
        for backend in backends:
            for sample_idx in range(num_samples):
                sample_dir = (
                    _GEN_ROOT
                    / experiment_id
                    / operator
                    / backend
                    / f"sample_{sample_idx:02d}"
                )
                impl_file = sample_dir / f"impl_{backend}.py"

                # ---------- Generation ----------
                if not eval_only:
                    print(
                        f"\n[{operator}/{backend}/sample_{sample_idx:02d}] "
                        f"Generating ..."
                    )
                    gen_result = generate_one(
                        client=client,
                        model_cfg=model_cfg,
                        operator=operator,
                        backend=backend,
                        profile=profile,
                        out_dir=sample_dir,
                        experiment_id=experiment_id,
                        sample_index=sample_idx,
                    )
                    if not gen_result.get("success"):
                        results.append({
                            "operator": operator,
                            "backend": backend,
                            "sample_index": sample_idx,
                            "phase": "generation",
                            "success": False,
                            "error": gen_result.get("error"),
                        })
                        continue
                else:
                    if not impl_file.exists():
                        print(
                            f"  Skipping {operator}/{backend}/sample_{sample_idx:02d}: "
                            f"impl file not found."
                        )
                        continue

                # ---------- Evaluation ----------
                print(
                    f"[{operator}/{backend}/sample_{sample_idx:02d}] "
                    f"Evaluating ..."
                )
                bench_path = sample_dir / "bench.json"
                eval_result = evaluate_via_subprocess(
                    operator=operator,
                    backend=backend,
                    impl_path=impl_file,
                    output_path=bench_path,
                )
                print(
                    f"  stage={eval_result.get('stage_name', '?')}  "
                    f"success={eval_result.get('success', False)}"
                )

                # ---------- Repair ----------
                repaired = False
                if not eval_result.get("success") and repair_rounds > 0:
                    print(
                        f"[{operator}/{backend}/sample_{sample_idx:02d}] "
                        f"Repairing (max {repair_rounds} rounds) ..."
                    )
                    repair_summary = repair_one(
                        client=client,
                        model_cfg=model_cfg,
                        operator=operator,
                        backend=backend,
                        sample_dir=sample_dir,
                        max_rounds=repair_rounds,
                    )
                    repaired = repair_summary.get("repaired", False)

                results.append({
                    "operator":     operator,
                    "backend":      backend,
                    "sample_index": sample_idx,
                    "phase":        "eval",
                    "success":      eval_result.get("success", False) or repaired,
                    "repaired":     repaired,
                    "stage":        eval_result.get("stage_name", "?"),
                })

    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch LLM kernel generation + evaluation for TileBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment ID from configs/experiments.yaml.",
    )
    parser.add_argument(
        "--operators", nargs="*", default=None,
        help="Override operator list (default: from experiment config).",
    )
    parser.add_argument(
        "--backends", nargs="*", default=None,
        help="Override backend list (default: from experiment config).",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Override number of samples per operator (default: from experiment config).",
    )
    parser.add_argument(
        "--repair-rounds", type=int, default=None,
        help="Override max repair rounds (default: from experiment config).",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip generation; evaluate existing impl files only.",
    )
    parser.add_argument(
        "--experiments-yaml",
        default=str(_CFG_DIR / "experiments.yaml"),
    )
    parser.add_argument(
        "--models-yaml",
        default=str(_CFG_DIR / "models.yaml"),
    )
    args = parser.parse_args()

    # Load configs
    exp_cfg      = _load_experiment_cfg(args.experiment, Path(args.experiments_yaml))
    model_configs = load_model_configs(args.models_yaml)
    model_alias  = exp_cfg["model_alias"]
    if model_alias not in model_configs:
        print(f"ERROR: model alias '{model_alias}' not found.", file=sys.stderr)
        sys.exit(1)
    model_cfg = model_configs[model_alias]
    client    = client_from_config(model_cfg)

    profile_name = exp_cfg.get("prompt_profile", f"{exp_cfg.get('backends', ['triton'])[0]}_zero_shot")
    profile      = _load_profile(profile_name)

    # Resolve operators
    if args.operators:
        operators = args.operators
    else:
        split = exp_cfg.get("operators_split", "dev_ops")
        operators = _load_operators_split(split)

    backends      = args.backends      or exp_cfg.get("backends", ["triton"])
    num_samples   = args.samples       or exp_cfg.get("num_samples", 1)
    repair_rounds = args.repair_rounds if args.repair_rounds is not None else exp_cfg.get("repair_rounds", 0)

    print(f"Experiment:    {args.experiment}")
    print(f"Model:         {model_alias} ({model_cfg.get('model')})")
    print(f"Profile:       {profile_name}")
    print(f"Operators:     {operators}")
    print(f"Backends:      {backends}")
    print(f"Samples:       {num_samples}")
    print(f"Repair rounds: {repair_rounds}")
    print(f"Eval only:     {args.eval_only}")
    print()

    run_result = run_experiment(
        experiment_id=args.experiment,
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        client=client,
        profile=profile,
        operators=operators,
        backends=backends,
        num_samples=num_samples,
        repair_rounds=repair_rounds,
        eval_only=args.eval_only,
    )

    # Write batch results
    batch_json = _GEN_ROOT / args.experiment / "batch_results.json"
    batch_json.parent.mkdir(parents=True, exist_ok=True)
    batch_json.write_text(
        json.dumps(run_result, indent=2), encoding="utf-8"
    )
    print(f"\nBatch results written to {batch_json}")

    # Generate summary
    try:
        summary = summarize_experiment(args.experiment)
        summary_path = _GEN_ROOT / args.experiment / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        agg = summary.get("aggregate", {})
        print("\n=== Experiment summary ===")
        for k, v in sorted(agg.items()):
            if isinstance(v, float):
                print(f"  {k:<35} {v:.4f}")
            else:
                print(f"  {k:<35} {v}")
    except Exception as exc:
        print(f"Warning: could not generate summary: {exc}")


if __name__ == "__main__":
    main()
