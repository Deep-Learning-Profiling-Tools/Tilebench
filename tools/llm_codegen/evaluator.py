"""Orchestrator that launches evaluator_runner.py as a subprocess.

Why a subprocess: LLM-generated code can segfault / hang / leak CUDA memory.
Isolating each iteration's evaluation in its own process keeps the main
generator robust across iterations.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_RUNNER = _THIS_DIR / "evaluator_runner.py"

# Overall subprocess timeout (catches the case where the runner itself hangs).
# Set so that an op with many cases × 2 backends still fits comfortably.
DEFAULT_OVERALL_TIMEOUT_S = 3600 * 2   # 2 hours
DEFAULT_PER_CASE_CAP_S = 900           # 15 minutes per (backend, case) autotune


def evaluate(
    op: str,
    iter_dir: Path,
    overall_timeout_s: int = DEFAULT_OVERALL_TIMEOUT_S,
    per_case_cap_s: int = DEFAULT_PER_CASE_CAP_S,
    skip_backends: list[str] | None = None,
) -> dict:
    """Run the evaluation subprocess. Returns the feedback dict.

    `iter_dir` must contain: impl_torch.py, config.yaml, and impl_<b>.py for
    every backend `b` not in `skip_backends`. Skipped backends are not
    loaded or timed; their per-backend score fields remain 0.0.
    """
    output_json = iter_dir / "feedback.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["LLMGEN_ITER_DIR"] = str(iter_dir)
    env["LLMGEN_OP"] = op
    env["LLMGEN_OUTPUT_JSON"] = str(output_json)
    env["LLMGEN_PER_CASE_CAP_S"] = str(per_case_cap_s)
    env["LLMGEN_SKIP_BACKENDS"] = ",".join(skip_backends or [])

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(_RUNNER)],
            env=env,
            cwd=str(_REPO_ROOT),
            timeout=overall_timeout_s,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if not output_json.exists():
            # Runner crashed before writing output. Synthesize feedback.
            return {
                "op": op,
                "iter_dir": str(iter_dir),
                "fatal": "evaluator_runner.py did not produce feedback.json",
                "elapsed_s": elapsed,
                "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
                "rc": proc.returncode,
            }
        feedback = json.loads(output_json.read_text())
        feedback["elapsed_s"] = elapsed
        # Surface stderr-tail if it's non-empty — helps debugging runner-internal warnings.
        if proc.stderr:
            feedback["stderr_tail"] = proc.stderr[-2000:]
        return feedback
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        return {
            "op": op,
            "iter_dir": str(iter_dir),
            "fatal": f"evaluator_runner.py exceeded overall {overall_timeout_s}s timeout",
            "elapsed_s": elapsed,
            "stdout_tail": (e.stdout or b"").decode(errors="ignore")[-2000:] if e.stdout else "",
            "stderr_tail": (e.stderr or b"").decode(errors="ignore")[-2000:] if e.stderr else "",
        }


def is_verify_clean(feedback: dict) -> bool:
    """True iff this iter is safe to promote to final/: no fatal, no compile
    errors, no per-case timeouts, no verify failures.
    """
    if feedback.get("fatal"):
        return False
    if any(feedback.get("compile_errors", {}).values()):
        return False
    if feedback.get("case_timeout_errors"):
        return False
    if feedback.get("verify_failures"):
        return False
    return True


def is_backend_verify_clean(feedback: dict, backend: str) -> bool:
    """True iff this backend in this iter is safe to promote: no compile / per-case-timeout
    failure for this backend, no verify failures for this backend.
    """
    if feedback.get("fatal"):
        return False
    if feedback.get("compile_errors", {}).get(backend):
        return False
    if feedback.get("case_timeout_errors", {}).get(backend):
        return False
    if feedback.get(f"verify_failures_{backend}"):
        return False
    return True
