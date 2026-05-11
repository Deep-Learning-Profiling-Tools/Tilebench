"""Main generation script for LLM kernel synthesis.

Generates one or more independent kernel samples for a given operator,
backend, experiment configuration, and model.  Each sample is written to
a self-contained directory under ``llm_kernelgen/generated/``.

Usage (CLI)
-----------
::

    # Single operator, single sample
    PYTHONPATH=. python llm_kernelgen/scripts/generate.py \\
        --operator softmax \\
        --backend triton \\
        --model gpt4o \\
        --experiment exp_smoke \\
        --samples 1

    # Batch over multiple operators
    PYTHONPATH=. python llm_kernelgen/scripts/generate.py \\
        --operator softmax rmsnorm rope \\
        --backend triton \\
        --model o3_medium \\
        --experiment exp_main \\
        --samples 5

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT   = Path(__file__).resolve().parents[2]
_LLM_ROOT    = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = _LLM_ROOT / "configs"
_GEN_ROOT    = _LLM_ROOT / "generated"

sys.path.insert(0, str(_REPO_ROOT))

from llm_kernelgen.runtime.adapters import client_from_config, load_model_configs
from llm_kernelgen.runtime.sandbox import validate_code
from llm_kernelgen.scripts.build_prompt import build_prompt, load_system_prompt
from llm_kernelgen.scripts.extract_code import count_lines, extract_single_python_block


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_one(
    *,
    client: Any,
    model_cfg: dict[str, Any],
    operator: str,
    backend: str,
    profile: dict[str, Any],
    out_dir: Path,
    experiment_id: str,
    sample_index: int,
) -> dict[str, Any]:
    """Generate a single kernel sample and write all artefacts to *out_dir*.

    Returns
    -------
    dict
        Summary record including success/failure status and metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt
    try:
        prompt_text, ctx_meta = build_prompt(
            operator_name=operator,
            backend=backend,
            profile=profile,
        )
    except Exception as exc:
        return {
            "success": False,
            "stage": "prompt_build",
            "error": str(exc),
            "out_dir": str(out_dir),
        }

    instructions = load_system_prompt()
    (out_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")

    # Generate via LLM
    t0 = time.monotonic()
    try:
        response = client.generate(
            model=model_cfg["model"],
            instructions=instructions,
            prompt=prompt_text,
            temperature=model_cfg.get("temperature", 0.2),
            max_output_tokens=model_cfg.get("max_output_tokens", 12000),
            reasoning_effort=model_cfg.get("reasoning_effort"),
        )
    except Exception as exc:
        return {
            "success": False,
            "stage": "api_call",
            "error": str(exc),
            "out_dir": str(out_dir),
        }
    gen_latency = time.monotonic() - t0

    (out_dir / "response.raw.txt").write_text(response.text, encoding="utf-8")

    # Extract code
    try:
        code = extract_single_python_block(response.text)
    except ValueError as exc:
        _write_metadata(
            out_dir, experiment_id, operator, backend, model_cfg,
            profile, sample_index, ctx_meta, response,
            stage="extract_code", error=str(exc),
        )
        return {
            "success": False,
            "stage": "extract_code",
            "error": str(exc),
            "out_dir": str(out_dir),
        }

    # Static validation
    try:
        validate_code(code, strict_file_io=True)
    except (SyntaxError, ValueError) as exc:
        (out_dir / f"impl_{backend}.py").write_text(code, encoding="utf-8")
        _write_metadata(
            out_dir, experiment_id, operator, backend, model_cfg,
            profile, sample_index, ctx_meta, response,
            stage="static_check", error=str(exc),
        )
        return {
            "success": False,
            "stage": "static_check",
            "error": str(exc),
            "out_dir": str(out_dir),
        }

    impl_file = out_dir / f"impl_{backend}.py"
    impl_file.write_text(code, encoding="utf-8")

    meta = _write_metadata(
        out_dir, experiment_id, operator, backend, model_cfg,
        profile, sample_index, ctx_meta, response,
        stage="generated",
        gen_latency_s=gen_latency,
        loc=count_lines(code),
    )

    print(
        f"  [{operator}/{backend}/sample_{sample_index:02d}] "
        f"generated OK  "
        f"({response.prompt_tokens}+{response.completion_tokens} tokens, "
        f"{gen_latency:.1f}s)"
    )

    return {
        "success": True,
        "stage": "generated",
        "impl_path": str(impl_file),
        "out_dir": str(out_dir),
        "meta": meta,
    }


def _write_metadata(
    out_dir: Path,
    experiment_id: str,
    operator: str,
    backend: str,
    model_cfg: dict,
    profile: dict,
    sample_index: int,
    ctx_meta: dict,
    response: Any | None,
    *,
    stage: str,
    error: str = "",
    gen_latency_s: float = 0.0,
    loc: int = 0,
) -> dict:
    meta: dict[str, Any] = {
        "experiment_id":              experiment_id,
        "operator":                   operator,
        "backend":                    backend,
        "sample_index":               sample_index,
        "model":                      model_cfg.get("model", ""),
        "api_type":                   model_cfg.get("api_type", ""),
        "base_url":                   model_cfg.get("base_url", ""),
        "temperature":                model_cfg.get("temperature", 0.2),
        "reasoning_effort":           model_cfg.get("reasoning_effort"),
        "max_output_tokens":          model_cfg.get("max_output_tokens", 12000),
        "prompt_profile":             profile.get("description", ""),
        "allowed_context":            ctx_meta,
        "forbidden_context_enforced": True,
        "timestamp":                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit":                 _get_git_commit(),
        "stage":                      stage,
        "error":                      error,
        "gen_latency_s":              gen_latency_s,
        "loc":                        loc,
        "prompt_tokens":              getattr(response, "prompt_tokens", 0) if response else 0,
        "completion_tokens":          getattr(response, "completion_tokens", 0) if response else 0,
        "reasoning_tokens":           getattr(response, "reasoning_tokens", 0) if response else 0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _load_profile(profile_name: str) -> dict[str, Any]:
    profiles_yaml = _CONFIGS_DIR / "prompt_profiles.yaml"
    with profiles_yaml.open() as fh:
        all_profiles = yaml.safe_load(fh)
    profiles = all_profiles.get("profiles", all_profiles)
    if profile_name not in profiles:
        raise KeyError(
            f"Profile '{profile_name}' not found in prompt_profiles.yaml.  "
            f"Available: {list(profiles.keys())}"
        )
    return profiles[profile_name]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM kernels for TileBench operators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--operator", nargs="+", required=True,
        help="Operator name(s) to generate kernels for.",
    )
    parser.add_argument(
        "--backend", choices=["triton", "cutile"], required=True,
        help="Target backend.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model alias from configs/models.yaml.",
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment ID string (used as the output subdirectory name).",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="Number of independent samples to generate per operator.",
    )
    parser.add_argument(
        "--profile", default=None,
        help=(
            "Prompt profile name from configs/prompt_profiles.yaml.  "
            "Defaults to '<backend>_zero_shot'."
        ),
    )
    parser.add_argument(
        "--models-yaml", default=str(_CONFIGS_DIR / "models.yaml"),
        help="Path to models.yaml.",
    )
    args = parser.parse_args()

    # Load config
    model_configs = load_model_configs(args.models_yaml)
    if args.model not in model_configs:
        print(f"ERROR: model alias '{args.model}' not found in {args.models_yaml}", file=sys.stderr)
        sys.exit(1)
    model_cfg = model_configs[args.model]

    profile_name = args.profile or f"{args.backend}_zero_shot"
    profile      = _load_profile(profile_name)

    client = client_from_config(model_cfg)

    results: list[dict] = []
    for operator in args.operator:
        for sample_idx in range(args.samples):
            out_dir = (
                _GEN_ROOT
                / args.experiment
                / operator
                / args.backend
                / f"sample_{sample_idx:02d}"
            )
            print(f"Generating {operator}/{args.backend}/sample_{sample_idx:02d} ...")
            result = generate_one(
                client=client,
                model_cfg=model_cfg,
                operator=operator,
                backend=args.backend,
                profile=profile,
                out_dir=out_dir,
                experiment_id=args.experiment,
                sample_index=sample_idx,
            )
            results.append(result)

    ok  = sum(1 for r in results if r.get("success"))
    tot = len(results)
    print(f"\nDone: {ok}/{tot} samples generated successfully.")

    # Write batch summary
    summary_path = _GEN_ROOT / args.experiment / "generate_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Batch summary written to {summary_path}")


if __name__ == "__main__":
    main()
