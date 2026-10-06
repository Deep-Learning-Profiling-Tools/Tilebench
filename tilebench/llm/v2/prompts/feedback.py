"""Whitelisted feedback construction.

Only these facts reach the generator: the attempt status, the measured
runtime (mean and the three raw samples) of VALID rounds, sanitized
diagnostics for failed rounds, and the candidate's own reported config.
Scoring fields (T_SOL, efficiency, roofline, speedup, human timing, peak
percentages, profiler counters) never pass through here, and the raw
evaluator JSON is never concatenated into a prompt."""
from __future__ import annotations

import re

STATUS_TEXT = {
    "valid": "compiled, passed numerical verification and was timed",
    "format_error": "response format error: the file could not be extracted",
    "interface_error": "interface error: run()/get_last_config() missing, failing, or not returning one fixed dict",
    "compile_error": "compilation/import failed",
    "runtime_error": "execution raised an error",
    "numerical_error": "output did not match the reference within tolerance",
    "timing_error": "timing failed",
    "contract_violation": "rejected for a contract violation (no repair succeeded in that round)",
    "review_required": "held for compliance review",
    "infrastructure_incomplete": "evaluation could not be completed (infrastructure)",
}

# Words whose presence in a diagnostic line would leak evaluator scoring.
_SCRUB = re.compile(r"(roofline|t_sol|t_emp|p_emp|bw_emp|calibration|empirical|ceiling|sol[_ -]?efficiency|speedup|stop_score|pct_peak|human|torch_ms|baseline_ms|score)",
                    re.I)


def sanitize_diagnostic(text: str | None, *, max_lines: int, max_chars: int) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.replace("\r", "").split("\n"):
        if _SCRUB.search(ln):
            lines.append("[line withheld]")
        else:
            lines.append(ln.rstrip())
        if len(lines) >= max_lines:
            lines.append(f"[... truncated to {max_lines} lines by a fixed rule]")
            break
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n[... truncated to {max_chars} characters by a fixed rule]"
    return out


def outcome_text(round_result: dict) -> str:
    status = round_result.get("status", "unknown")
    base = STATUS_TEXT.get(status, status)
    if status == "valid":
        s = round_result.get("latency_ms_samples") or []
        samples = ", ".join(f"{x:.4f}" for x in s)
        return f"{base}: {round_result['latency_ms_mean']:.4f} ms (samples: {samples})"
    return base


def diagnostics_block(round_result: dict, limits: dict) -> str:
    diag = round_result.get("diagnostic")
    if not diag:
        return ""
    text = sanitize_diagnostic(diag, max_lines=limits["max_diagnostic_lines"], max_chars=limits["max_diagnostic_chars"])
    return f"\nDiagnostics:\n```\n{text}\n```\n"


def runtime_history(rounds: list[dict]) -> str:
    rows = [r for r in rounds if r.get("status") == "valid"]
    if not rows:
        return "No valid candidate yet."
    lines = ["| round | mean ms | samples ms |", "|---|---|---|"]
    for r in rows:
        s = ", ".join(f"{x:.4f}" for x in (r.get("latency_ms_samples") or []))
        lines.append(f"| {r['round']} | {r['latency_ms_mean']:.4f} | {s} |")
    return "\n".join(lines)


FORBIDDEN_FEEDBACK_KEYS = ("t_sol", "t_emp", "p_emp", "bw_emp", "calibration", "sol_efficiency", "efficiency", "roofline",
                           "stop_score", "speedup", "human_ms", "torch_ms", "pct_peak", "profiler", "score", "tflops", "bandwidth")


def assert_feedback_clean(text: str) -> None:
    hit = _SCRUB.search(text)
    if hit:
        raise AssertionError(f"feedback leaks scoring term {hit.group(0)!r}")
