"""Whitelisted feedback construction.

Only these facts reach the generator: the round status, the number of valid
cases out of the suite's cases, the geometric-mean runtime over the cases of
VALID rounds (never a per-case runtime), sanitized diagnostics and, for an
invalid round, the semantic parameters of its first failing case, and the
configurations the candidate itself reported (without their case mapping).
Scoring fields (T_SOL, efficiency, roofline, speedup, human timing, peak
percentages, profiler counters) never pass through here, and the raw
evaluator JSON is never concatenated into a prompt."""
from __future__ import annotations

import json
import re

STATUS_TEXT = {
    "valid": "compiled, passed numerical verification and was timed",
    "format_error": "response format error: the file could not be extracted",
    "interface_error": "interface error: run()/get_last_config() missing, failing, or not returning one fixed dict",
    "compile_error": "compilation/import failed",
    "runtime_error": "execution raised an error",
    "numerical_error": "output did not match the reference within tolerance",
    "timing_error": "timing failed",
    "contract_violation": "rejected for a contract violation (not evaluated)",
    "review_required": "held for compliance review",
    "infrastructure_incomplete": "evaluation could not be completed (infrastructure)",
}

# Words whose presence in a diagnostic line would leak evaluator scoring.
_SCRUB = re.compile(r"(roofline|t_sol|t_emp|p_emp|bw_emp|calibration|empirical|ceiling|sol[_ -]?efficiency|speedup|stop_score|pct_peak|human|torch_ms|baseline_ms|score"
                    r"|x-ratelimit|anthropic-ratelimit|estimated_cost|pricing_snapshot|api_pricing|usd_cost)", re.I)


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
    total = round_result.get("cases_total")
    if status == "valid":
        return f"{base} on all {total} cases: geometric-mean runtime {round_result['latency_ms_geomean']:.4f} ms"
    if total:
        v, ev = round_result.get("valid_cases") or 0, round_result.get("cases_evaluated")
        out = f"{base}; valid on {v} of {total} cases" + (f" ({ev} evaluated)" if ev is not None and ev != total else "")
        fb = round_result.get("first_failing_case")
        if fb and fb.get("params") is not None:
            params = ", ".join(f"{k}={val}" for k, val in fb["params"].items())
            out += f"; first failing case: {params} ({STATUS_TEXT.get(fb.get('status'), fb.get('status'))})"
        return out
    return base


def configs_line(configs: list | None) -> str:
    """The configurations the candidate reported (distinct values, no case mapping)."""
    if not configs:
        return "Configurations reported by `get_last_config()`: none recorded."
    shown = ", ".join(f"`{json.dumps(c, sort_keys=True)}`" for c in configs[:6])
    more = f" (+{len(configs) - 6} more)" if len(configs) > 6 else ""
    return f"Configurations reported by `get_last_config()` across the cases ({len(configs)} distinct): {shown}{more}"


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
    lines = ["| round | valid cases | geometric-mean ms |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['round']} | {r.get('valid_cases')}/{r.get('cases_total')} | {r['latency_ms_geomean']:.4f} |")
    return "\n".join(lines)


FORBIDDEN_FEEDBACK_KEYS = ("t_sol", "t_emp", "p_emp", "bw_emp", "calibration", "sol_efficiency", "efficiency", "roofline",
                           "stop_score", "speedup", "human_ms", "torch_ms", "pct_peak", "profiler", "score", "tflops", "bandwidth",
                           "usd", "estimated_cost", "pricing", "rate_limit")


def assert_feedback_clean(text: str) -> None:
    hit = _SCRUB.search(text)
    if hit:
        raise AssertionError(f"feedback leaks scoring term {hit.group(0)!r}")
