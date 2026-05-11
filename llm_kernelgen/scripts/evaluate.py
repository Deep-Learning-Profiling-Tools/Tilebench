"""Evaluation script for LLM-generated kernels.

Runs the TileBench engine against one generated ``impl_<backend>.py`` file
and records multi-stage results:

  Stage 0 – syntax / import success
  Stage 1 – run() executes on case 0
  Stage 2 – correctness pass on case 0
  Stage 3 – correctness pass on all configured cases
  Stage 4 – full benchmark completes
  Stage 5 – performance score computed

Usage (CLI)
-----------
::

    PYTHONPATH=. python llm_kernelgen/scripts/evaluate.py \\
        --operator softmax \\
        --backend triton \\
        --impl llm_kernelgen/generated/exp1/softmax/triton/sample_00/impl_triton.py \\
        --output llm_kernelgen/generated/exp1/softmax/triton/sample_00/bench.json

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_SYNTAX_IMPORT  = 0
STAGE_RUN_CASE0      = 1
STAGE_CORRECT_CASE0  = 2
STAGE_CORRECT_ALL    = 3
STAGE_BENCH_COMPLETE = 4
STAGE_PERF_SCORE     = 5


def _stage_name(s: int) -> str:
    return {
        0: "syntax_import",
        1: "run_case0",
        2: "correct_case0",
        3: "correct_all",
        4: "bench_complete",
        5: "perf_score",
    }.get(s, f"stage_{s}")


# ---------------------------------------------------------------------------
# Subprocess-based evaluation (isolates GPU state between samples)
# ---------------------------------------------------------------------------

def evaluate_via_subprocess(
    *,
    operator: str,
    backend: str,
    impl_path: Path,
    output_path: Path,
    case_indices: str | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run evaluation in a subprocess and return the parsed result dict.

    Using a subprocess ensures that a kernel that crashes or leaks GPU memory
    does not corrupt the evaluation of subsequent samples.
    """
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_generated.py"),
        "--operator", operator,
        "--backend", backend,
        "--impl", str(impl_path),
        "--output", str(output_path),
    ]
    if case_indices:
        cmd += ["--case-indices", case_indices]

    log_path = output_path.with_name("eval.log")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=str(_REPO_ROOT),
        )
        log = proc.stdout or ""
        log_path.write_text(log, encoding="utf-8")

        if proc.returncode == 0 and output_path.exists():
            with output_path.open() as fh:
                bench_results: list[dict] = json.load(fh)
            return _parse_bench_results(bench_results, log)
        else:
            return {
                "stage_reached": STAGE_SYNTAX_IMPORT,
                "stage_name": _stage_name(STAGE_SYNTAX_IMPORT),
                "success": False,
                "error": f"subprocess exited with code {proc.returncode}",
                "log_snippet": log[-2000:],
            }
    except subprocess.TimeoutExpired:
        return {
            "stage_reached": STAGE_SYNTAX_IMPORT,
            "stage_name": _stage_name(STAGE_SYNTAX_IMPORT),
            "success": False,
            "error": f"evaluation timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "stage_reached": STAGE_SYNTAX_IMPORT,
            "stage_name": _stage_name(STAGE_SYNTAX_IMPORT),
            "success": False,
            "error": str(exc),
        }


def _parse_bench_results(results: list[dict], log: str) -> dict[str, Any]:
    """Parse the list of per-case result dicts from the engine."""
    if not results:
        return {
            "stage_reached": STAGE_RUN_CASE0,
            "stage_name": _stage_name(STAGE_RUN_CASE0),
            "success": False,
            "error": "empty results list",
        }

    backend_key_ok  = None  # e.g. "triton_ok"
    backend_key_ms  = None  # e.g. "triton_ms"
    backend_key_err = None  # e.g. "triton_err"

    # Detect backend from keys.
    sample = results[0]
    for bk in ("triton", "cutile"):
        if f"{bk}_ok" in sample:
            backend_key_ok  = f"{bk}_ok"
            backend_key_ms  = f"{bk}_ms"
            backend_key_err = f"{bk}_err"
            break

    if backend_key_ok is None:
        return {
            "stage_reached": STAGE_SYNTAX_IMPORT,
            "stage_name": _stage_name(STAGE_SYNTAX_IMPORT),
            "success": False,
            "error": "cannot detect backend key in bench results",
        }

    total         = len(results)
    correct_count = sum(1 for r in results if r.get(backend_key_ok))
    all_correct   = correct_count == total

    first_ok    = results[0].get(backend_key_ok, False)
    first_err   = results[0].get(backend_key_err, "")
    torch_ms    = [r.get("torch_ms", float("nan")) for r in results]
    backend_ms  = [r.get(backend_key_ms, float("nan")) for r in results]
    speedups    = [
        t / b if b and b == b and b > 0 else None
        for t, b in zip(torch_ms, backend_ms)
    ]

    # Determine highest stage reached.
    if not first_ok:
        stage = STAGE_RUN_CASE0  # ran but failed on case 0
    elif all_correct:
        stage = STAGE_BENCH_COMPLETE
    else:
        stage = STAGE_CORRECT_CASE0 if first_ok else STAGE_RUN_CASE0

    return {
        "stage_reached": stage,
        "stage_name": _stage_name(stage),
        "success": all_correct,
        "total_cases": total,
        "correct_cases": correct_count,
        "first_case_correct": first_ok,
        "first_case_error": first_err,
        "torch_ms": torch_ms,
        "backend_ms": backend_ms,
        "speedups_vs_torch": speedups,
        "mean_speedup_vs_torch": (
            sum(s for s in speedups if s is not None) / max(1, sum(1 for s in speedups if s is not None))
        ),
        "log_snippet": log[-2000:],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one LLM-generated kernel file using TileBench.",
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--backend", choices=["triton", "cutile"], required=True)
    parser.add_argument("--impl", required=True, help="Path to impl_<backend>.py")
    parser.add_argument("--output", default=None,
                        help="Output path for bench.json (default: next to impl file)")
    parser.add_argument("--case-indices", default=None,
                        help="Comma-separated case indices to restrict evaluation.")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    impl_path   = Path(args.impl).resolve()
    output_path = Path(args.output) if args.output else impl_path.with_name("bench.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {args.operator}/{args.backend}: {impl_path}")
    result = evaluate_via_subprocess(
        operator=args.operator,
        backend=args.backend,
        impl_path=impl_path,
        output_path=output_path,
        case_indices=args.case_indices,
        timeout=args.timeout,
    )

    status_path = impl_path.with_name("eval_status.json")
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    stage = result.get("stage_name", "?")
    ok    = result.get("success", False)
    print(f"  Stage reached: {stage}  success={ok}")
    if not ok:
        print(f"  Error: {result.get('error', '')}")


if __name__ == "__main__":
    main()
