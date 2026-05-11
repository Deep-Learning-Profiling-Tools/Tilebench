"""Repair loop for LLM-generated kernels.

Given a failed kernel sample directory, reads the error log and previously
generated code, constructs a repair prompt, calls the LLM, and writes the
fixed code back to the same directory (with round-specific filenames so
nothing is lost).

Repair protocol (paper-compliant):
  - Maximum R rounds (configurable, default 2).
  - Each round sees ONLY:
      * The error log from the previous attempt.
      * The previously generated code.
  - Human ground-truth implementation is NEVER shown.
  - All repair artefacts are archived as
      ``impl_<backend>_repair_r<N>.py``, ``response_repair_r<N>.raw.txt``, etc.

Usage (CLI)
-----------
::

    PYTHONPATH=. python llm_kernelgen/scripts/repair.py \\
        --operator softmax \\
        --backend triton \\
        --sample-dir llm_kernelgen/generated/exp1/softmax/triton/sample_00 \\
        --model o3_medium \\
        --max-rounds 2

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LLM_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from llm_kernelgen.runtime.adapters import client_from_config, load_model_configs
from llm_kernelgen.runtime.sandbox import validate_code
from llm_kernelgen.scripts.extract_code import count_lines, extract_single_python_block
from llm_kernelgen.scripts.evaluate import evaluate_via_subprocess

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    _HAS_JINJA = True
except ImportError:
    _HAS_JINJA = False


_PROMPTS_DIR = _LLM_ROOT / "prompts"
_CONFIGS_DIR = _LLM_ROOT / "configs"


def _load_repair_template() -> Any:
    env = Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("repair.md.j2")


def _load_system_prompt() -> str:
    path = _PROMPTS_DIR / "system.md"
    return path.read_text(encoding="utf-8") if path.exists() else (
        "You are an expert GPU kernel engineer."
    )


def _read_error_log(sample_dir: Path) -> str:
    for candidate in ("eval.log", "compile.log", "verify.log"):
        p = sample_dir / candidate
        if p.exists():
            return p.read_text(encoding="utf-8")[-4000:]
    return "No error log found."


def _read_previous_code(sample_dir: Path, backend: str, round_num: int) -> str:
    """Return the most recent generated code before this repair round."""
    if round_num == 1:
        p = sample_dir / f"impl_{backend}.py"
    else:
        p = sample_dir / f"impl_{backend}_repair_r{round_num - 1}.py"

    if p.exists():
        return p.read_text(encoding="utf-8")

    # Fallback: find the latest repair file.
    candidates = sorted(sample_dir.glob(f"impl_{backend}*.py"))
    if candidates:
        return candidates[-1].read_text(encoding="utf-8")
    return "# No previous code found.\n"


def _detect_failure_stage(sample_dir: Path) -> tuple[str, str]:
    """Return (failure_stage, error_type) by reading eval_status.json."""
    status_path = sample_dir / "eval_status.json"
    if status_path.exists():
        with status_path.open() as fh:
            status = json.load(fh)
        stage = status.get("stage_name", "unknown")
        error = status.get("error", "")
        return stage, type(error).__name__ if error else "unknown"
    return "unknown", "unknown"


def repair_one(
    *,
    client: Any,
    model_cfg: dict[str, Any],
    operator: str,
    backend: str,
    sample_dir: Path,
    max_rounds: int = 2,
) -> dict[str, Any]:
    """Run the repair loop for one sample directory.

    Parameters
    ----------
    client:
        Instantiated LLM client.
    model_cfg:
        Model configuration dict.
    operator:
        Operator name.
    backend:
        ``"triton"`` or ``"cutile"``.
    sample_dir:
        Path to the sample directory (contains ``impl_<backend>.py``).
    max_rounds:
        Maximum number of repair iterations.

    Returns
    -------
    dict
        Repair summary including ``repaired`` (bool) and round details.
    """
    if not _HAS_JINJA:
        raise ImportError("Jinja2 is required.  pip install jinja2")

    template      = _load_repair_template()
    instructions  = _load_system_prompt()
    repair_log: list[dict] = []

    for round_num in range(1, max_rounds + 1):
        error_log    = _read_error_log(sample_dir)
        prev_code    = _read_previous_code(sample_dir, backend, round_num)
        stage, etype = _detect_failure_stage(sample_dir)

        prompt_text = template.render(
            backend=backend,
            operator_name=operator,
            repair_round=round_num,
            failure_stage=stage,
            error_type=etype,
            error_log=error_log,
            failing_cases=[],
            previous_code=prev_code,
        )

        repair_prompt_path = sample_dir / f"prompt_repair_r{round_num}.md"
        repair_prompt_path.write_text(prompt_text, encoding="utf-8")

        # Call LLM
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
            repair_log.append({
                "round": round_num,
                "success": False,
                "stage": "api_call",
                "error": str(exc),
            })
            continue
        latency = time.monotonic() - t0

        raw_path = sample_dir / f"response_repair_r{round_num}.raw.txt"
        raw_path.write_text(response.text, encoding="utf-8")

        # Extract and validate code
        try:
            code = extract_single_python_block(response.text)
            validate_code(code, strict_file_io=True)
        except (SyntaxError, ValueError) as exc:
            repair_log.append({
                "round": round_num,
                "success": False,
                "stage": "extract_or_validate",
                "error": str(exc),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_s": latency,
            })
            continue

        # Write repaired code
        repair_impl_path = sample_dir / f"impl_{backend}_repair_r{round_num}.py"
        repair_impl_path.write_text(code, encoding="utf-8")

        # Also overwrite the canonical impl file so evaluation uses the latest.
        (sample_dir / f"impl_{backend}.py").write_text(code, encoding="utf-8")

        # Evaluate repaired code
        bench_path = sample_dir / f"bench_repair_r{round_num}.json"
        eval_result = evaluate_via_subprocess(
            operator=operator,
            backend=backend,
            impl_path=sample_dir / f"impl_{backend}.py",
            output_path=bench_path,
        )

        round_summary = {
            "round":             round_num,
            "success":           eval_result.get("success", False),
            "stage_reached":     eval_result.get("stage_name", "?"),
            "error":             eval_result.get("error", ""),
            "prompt_tokens":     response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "reasoning_tokens":  response.reasoning_tokens,
            "latency_s":         latency,
            "loc":               count_lines(code),
        }
        repair_log.append(round_summary)

        print(
            f"  Repair round {round_num}/{max_rounds}: "
            f"success={round_summary['success']}  "
            f"stage={round_summary['stage_reached']}"
        )

        if round_summary["success"]:
            break

    final_success = any(r.get("success") for r in repair_log)
    summary = {
        "repaired":    final_success,
        "rounds_used": len(repair_log),
        "max_rounds":  max_rounds,
        "rounds":      repair_log,
    }

    (sample_dir / "repair_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM repair loop on a failed kernel sample.",
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--backend", choices=["triton", "cutile"], required=True)
    parser.add_argument("--sample-dir", required=True,
                        help="Path to the sample directory.")
    parser.add_argument("--model", required=True,
                        help="Model alias from configs/models.yaml.")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--models-yaml",
                        default=str(_CONFIGS_DIR / "models.yaml"))
    args = parser.parse_args()

    model_configs = load_model_configs(args.models_yaml)
    if args.model not in model_configs:
        print(f"ERROR: model alias '{args.model}' not found.", file=sys.stderr)
        sys.exit(1)
    model_cfg = model_configs[args.model]
    client    = client_from_config(model_cfg)

    sample_dir = Path(args.sample_dir).resolve()
    print(f"Repairing {args.operator}/{args.backend}: {sample_dir}")

    summary = repair_one(
        client=client,
        model_cfg=model_cfg,
        operator=args.operator,
        backend=args.backend,
        sample_dir=sample_dir,
        max_rounds=args.max_rounds,
    )

    print(
        f"Repair done: repaired={summary['repaired']}  "
        f"rounds_used={summary['rounds_used']}/{summary['max_rounds']}"
    )


if __name__ == "__main__":
    main()
