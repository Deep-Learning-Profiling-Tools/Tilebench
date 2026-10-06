"""LLM code-generation main loop.

Usage:
    python -m tilebench.llm.generate \
        --operator vector_add --model gpt-5.5 --max-iters 10

For each iteration:
  1. Build a prompt (initial vs feedback).
  2. Call the LLM (cached: skip if iter_N/response.md already exists).
  3. Parse out impl_triton.py + impl_cutile.py.
  4. Copy impl_torch.py + config.yaml from benchmarks/operators/<op>/.
  5. Run evaluator (subprocess; 15-min per-case autotune cap).
  6. Save feedback.json.

The loop runs the full max_iters budget for both backends every iter;
there is no early-stopping based on roofline / stop_score thresholds.
After the budget is exhausted, the best verify-clean iter for each
backend is promoted to benchmarks/llm_generated/<op>/<model>/final/.

Each iteration is saved verbatim to benchmarks/llm_generated/<op>/<model>/iter_N/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
from tilebench.paths import LLM_GENERATED_ROOT, REPO_ROOT, operator_dir
_REPO_ROOT = REPO_ROOT
sys.path.insert(0, str(_REPO_ROOT))

from tilebench.llm.evaluator import (
    evaluate,
    is_backend_verify_clean,
)
from tilebench.llm.llm_client import LLMClient
from tilebench.llm.prompt_builder import (
    SYSTEM, build_feedback_prompt, build_initial_prompt, parse_response,
)

BACKENDS: tuple[str, ...] = ("triton", "cutile")


def _iter_dir(op: str, model: str, effort: str, iter_idx: int) -> Path:
    return (
        LLM_GENERATED_ROOT
        / op / model / effort / f"iter_{iter_idx}"
    )


def _final_dir(op: str, model: str, effort: str) -> Path:
    return LLM_GENERATED_ROOT / op / model / effort / "final"


def _trim_case_grid_to_largest(cfg_text: str) -> str:
    """Trim each `case_grid` swept variable down to its largest single value.

    The main benchmark engine sweeps ~20 sizes per op to characterise scaling,
    but the LLM-codegen pipeline picks ONE hard-coded cfg per iter — there's
    no per-size autotune, so the small cases just re-run the same kernel with
    the cfg tuned for the largest case. Evaluating them adds eval time
    without giving the LLM new information. Trimming to the largest matches
    the NCU sweep's `default_params_per_dtype` (so LLM-codegen and NCU
    results are directly comparable) and cuts evaluator time ~7-20x.

    Handles two YAML shapes:

      var:
        expr: "[N * i for i in range(1, 21)]"     →  var: [N*20]
      var: [a, b, c, d]                            →  var: [max(...)]

    All other lines (comments, case_defaults, metrics, verify, etc.) are
    preserved verbatim. The original `benchmarks/operators/<op>/config.yaml`
    is NOT modified; only the copy under `benchmarks/llm_generated/...`.
    """
    import re

    out_lines: list[str] = []
    lines = cfg_text.splitlines(keepends=True)
    i = 0
    in_case_grid = False
    case_grid_indent = -1
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        # Track case_grid block boundaries: starts at unindented `case_grid:`,
        # ends when we hit another top-level key.
        if re.match(r"^case_grid:\s*$", stripped):
            in_case_grid = True
            case_grid_indent = 0
            out_lines.append(line)
            i += 1
            continue
        if in_case_grid:
            # End of case_grid block: top-level key or end of file
            if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                in_case_grid = False

        if in_case_grid:
            # Pattern A: explicit list  `  <var>: [a, b, c]`
            m = re.match(r"^(\s+)(\w+):\s*\[([^\]]+)\]\s*(#.*)?$", line)
            if m and m.group(2) != "dtype":
                indent, var, body, comment = m.groups()
                items = [s.strip() for s in body.split(",") if s.strip()]
                # Try to parse as ints; if any item isn't an int, leave alone.
                try:
                    ints = [int(s) for s in items]
                    largest = max(ints)
                    comment = (" " + comment.strip()) if comment else ""
                    out_lines.append(f"{indent}{var}: [{largest}]{comment}\n")
                    i += 1
                    continue
                except ValueError:
                    pass

            # Pattern B: `  <var>:\n    expr: "[...]"`
            m = re.match(r"^(\s+)(\w+):\s*$", line)
            if m and m.group(2) != "dtype" and i + 1 < len(lines):
                nxt = lines[i + 1]
                mexpr = re.match(r"^(\s+)expr:\s*\"([^\"]+)\"\s*(#.*)?$", nxt)
                if mexpr:
                    indent_outer, var = m.group(1), m.group(2)
                    _indent_inner, expr_body, comment = mexpr.groups()
                    try:
                        values = eval(expr_body, {"range": range})
                        largest = max(values)
                        comment = (" " + comment.strip()) if comment else ""
                        out_lines.append(
                            f"{indent_outer}{var}: [{largest}]{comment}\n"
                        )
                        i += 2
                        continue
                    except Exception:
                        pass

        out_lines.append(line)
        i += 1
    return "".join(out_lines)


def _copy_framework_files(op: str, dest: Path) -> None:
    """Bring impl_torch.py and config.yaml into the iter dir so the evaluator
    has everything it needs in one place. The copied config.yaml has its
    case_grid trimmed to a single largest-case per swept variable (matches
    the NCU sweep's default_params_per_dtype). The source file under
    benchmarks/operators/<op>/ is left untouched."""
    src = operator_dir(op)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "impl_torch.py", dest / "impl_torch.py")
    cfg_text = (src / "config.yaml").read_text()
    (dest / "config.yaml").write_text(_trim_case_grid_to_largest(cfg_text))


def _read_prev_impls(iter_dir: Path) -> tuple[str, str]:
    triton = (iter_dir / "impl_triton.py").read_text() if (iter_dir / "impl_triton.py").exists() else ""
    cutile = (iter_dir / "impl_cutile.py").read_text() if (iter_dir / "impl_cutile.py").exists() else ""
    return triton, cutile


def _short_feedback_summary(feedback: dict, active_backends: tuple[str, ...]) -> str:
    parts = []
    if feedback.get("fatal"):
        parts.append(f"FATAL: {feedback['fatal']}")
        return " | ".join(parts)
    ce = feedback.get("compile_errors", {})
    bad_compile = [b for b, e in ce.items() if e and b in active_backends]
    if bad_compile:
        parts.append(f"compile_fail={bad_compile}")
    te = feedback.get("case_timeout_errors", {})
    bad_timeout = [b for b in te if b in active_backends]
    if bad_timeout:
        parts.append(f"case_timeout={bad_timeout}")
    for b in active_backends:
        score = feedback.get(f"stop_score_{b}", 0.0)
        vf = len(feedback.get(f"verify_failures_{b}", []))
        verify = "✓" if vf == 0 else f"✗{vf}"
        parts.append(f"{b}={score*100:.1f}%/{verify}")
    return " | ".join(parts)


def run_one_iter(
    op: str,
    model: str,
    effort: str,
    iter_idx: int,
    client: LLMClient,
    prev_feedback: dict | None,
    active_backends: tuple[str, ...],
    history: list[dict] | None = None,
    best_so_far: dict | None = None,
) -> dict:
    """Run one iteration: build prompt → LLM → parse → eval.

    Returns feedback, augmented with `llm_usage` and `timing_breakdown`.
    """
    iter_dir = _iter_dir(op, model, effort, iter_idx)
    iter_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = iter_dir / "prompt.md"
    response_path = iter_dir / "response.md"
    impl_paths = {b: iter_dir / f"impl_{b}.py" for b in active_backends}

    # ---- Build prompt ----
    if iter_idx == 0:
        prompt = build_initial_prompt(op, backends=active_backends)
    else:
        prev_dir = _iter_dir(op, model, effort, iter_idx - 1)
        prev_triton, prev_cutile = _read_prev_impls(prev_dir)
        prompt = build_feedback_prompt(
            op=op, iter_idx=iter_idx,
            prev_impl_triton=prev_triton,
            prev_impl_cutile=prev_cutile,
            feedback=prev_feedback or {},
            history=history or [],
            best_so_far=best_so_far,
            backends=active_backends,
        )
    prompt_path.write_text(prompt)

    # ---- Call LLM (cached) ----
    llm_usage = {}
    cache_complete = response_path.exists() and all(p.exists() for p in impl_paths.values())
    if cache_complete:
        print(f"  [iter {iter_idx}] cache hit — skipping LLM call", flush=True)
        response_text = response_path.read_text()
    else:
        print(f"  [iter {iter_idx}] calling {model} for backends={list(active_backends)} ...", flush=True)
        t0 = time.time()
        resp = client.generate(prompt, system=SYSTEM)
        elapsed = time.time() - t0
        response_text = resp.text
        response_path.write_text(response_text)
        llm_usage = {
            "elapsed_s": elapsed,
            "ttft_s": resp.ttft_s,
            "input_tokens": resp.input_tokens,
            "cached_input_tokens": resp.cached_input_tokens,
            "output_tokens": resp.output_tokens,
            "reasoning_tokens": resp.reasoning_tokens,
            "response_chars": len(response_text),
        }
        (iter_dir / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2))
        ttft_str = f"{resp.ttft_s:.1f}s" if resp.ttft_s is not None else "—"
        cached_str = resp.cached_input_tokens if resp.cached_input_tokens is not None else "—"
        print(
            f"  [iter {iter_idx}] got {len(response_text)} chars in {elapsed:.1f}s "
            f"(ttft={ttft_str} in_tok={resp.input_tokens} cached={cached_str} "
            f"out_tok={resp.output_tokens} reasoning_tok={resp.reasoning_tokens})",
            flush=True,
        )

        # Parse the response — only require active-backend files.
        files = parse_response(response_text)
        missing = [f"impl_{b}.py" for b in active_backends if f"impl_{b}.py" not in files]
        if missing:
            return {
                "fatal": f"LLM response missing required files: {missing}",
                "response_length": len(response_text),
                "llm_usage": llm_usage,
            }
        for b in active_backends:
            impl_paths[b].write_text(files[f"impl_{b}.py"])

    # ---- Copy reference + config ----
    _copy_framework_files(op, iter_dir)

    # ---- Evaluate ----
    skip_backends = [b for b in BACKENDS if b not in active_backends]
    print(f"  [iter {iter_idx}] evaluating (skip={skip_backends}) ...", flush=True)
    feedback = evaluate(op=op, iter_dir=iter_dir, skip_backends=skip_backends)
    # Surface llm_usage + active backends into feedback for downstream tracking.
    if not llm_usage and (iter_dir / "llm_usage.json").exists():
        try:
            llm_usage = json.loads((iter_dir / "llm_usage.json").read_text())
        except Exception:
            pass
    feedback["llm_usage"] = llm_usage
    feedback["active_backends"] = list(active_backends)
    (iter_dir / "feedback.json").write_text(json.dumps(feedback, indent=2, default=str))
    print(f"  [iter {iter_idx}] {_short_feedback_summary(feedback, active_backends)}", flush=True)
    return feedback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", required=True, help="Operator name (e.g. vector_add)")
    ap.add_argument(
        "--model", default="gpt-5.5",
        help="LLM to use (gpt-5.5 / claude-opus-4-7)",
    )
    ap.add_argument(
        "--effort", default="xhigh",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level (default xhigh). 'high' is ~3-5× faster.",
    )
    ap.add_argument("--max-iters", type=int, default=10)
    args = ap.parse_args()

    client = LLMClient(model=args.model, effort=args.effort)

    print(
        f"=== LLM codegen: op={args.operator} model={args.model} "
        f"effort={args.effort} max_iters={args.max_iters} ===",
        flush=True,
    )

    prev_feedback = None
    history: list[dict] = []
    # Per-backend state.
    #   best_clean[b]: highest stop_score_<b> among verify-clean iters for `b`
    #                  (carries impl source so the feedback prompt can show it
    #                  for regression-recovery), plus iter index.
    #   best_any[b]:   highest stop_score_<b> regardless of verify cleanliness
    #                  (fallback for promotion).
    # There is no roofline-based early stopping; the loop runs the full
    # max_iters budget. Both backends are regenerated every iter.
    best_clean: dict[str, dict] = {}
    best_any: dict[str, dict] = {}

    for i in range(args.max_iters):
        active_backends = BACKENDS  # both backends generated every iter

        print(f"\n=== Iteration {i} ===", flush=True)
        iter_t0 = time.time()
        try:
            feedback = run_one_iter(
                op=args.operator, model=args.model, effort=args.effort,
                iter_idx=i, client=client, prev_feedback=prev_feedback,
                active_backends=active_backends,
                history=history,
                best_so_far=best_clean or None,
            )
        except Exception as e:  # noqa: BLE001 — keep iterating despite transient API / runner failures
            import traceback
            print(f"  [iter {i}] FAILED with {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            iter_dir = _iter_dir(args.operator, args.model, args.effort, i)
            iter_dir.mkdir(parents=True, exist_ok=True)
            feedback = {
                "fatal": f"{type(e).__name__}: {e}",
                "iter_dir": str(iter_dir),
                "active_backends": list(active_backends),
            }
            (iter_dir / "feedback.json").write_text(json.dumps(feedback, indent=2))
        iter_elapsed_s = time.time() - iter_t0
        feedback.setdefault("timing_breakdown", {})
        feedback["timing_breakdown"]["iter_total_s"] = iter_elapsed_s
        iter_dir = _iter_dir(args.operator, args.model, args.effort, i)
        (iter_dir / "feedback.json").write_text(json.dumps(feedback, indent=2, default=str))
        prev_feedback = feedback

        # ---- Track per-backend history + bests ----
        per_backend_h: dict[str, dict] = {}
        for b in BACKENDS:
            score = feedback.get(f"stop_score_{b}", 0.0)
            rep = feedback.get(f"report_arith_mean_{b}", 0.0)
            vf_count = len(feedback.get(f"verify_failures_{b}", []))
            clean = is_backend_verify_clean(feedback, b)
            # Pick a representative cfg + speedup for this backend in this iter:
            # use the LARGEST verify-clean case (last in problem-size sort).
            # If all combos failed verify, leave cfg/speedup as None.
            backend_combos = [
                r for r in feedback.get("roofline_per_combo", [])
                if r.get("backend") == b
            ]
            backend_combos.sort(key=lambda r: -(r.get("problem_size") or 0))
            rep_combo = backend_combos[0] if backend_combos else {}
            per_backend_h[b] = {
                "skipped": False,
                "stop_score": score,
                "report_mean": rep,
                "verify_clean": clean,
                "verify_fail_count": vf_count,
                "cfg": rep_combo.get("cfg"),
                "speedup_vs_torch": rep_combo.get("speedup_vs_torch"),
            }
            # best_any: highest score regardless of verify
            if b not in best_any or score > best_any[b]["stop_score"]:
                best_any[b] = {"iter": i, "stop_score": score, "report_mean": rep}
            # best_clean: highest score among verify-clean iters; carry source
            if clean and (b not in best_clean or score > best_clean[b]["stop_score"]):
                src_path = iter_dir / f"impl_{b}.py"
                best_clean[b] = {
                    "iter": i, "stop_score": score, "report_mean": rep,
                    f"impl_{b}": src_path.read_text() if src_path.exists() else "",
                }

        history_entry = {
            "iter": i,
            "active_backends": list(active_backends),
            "per_backend": per_backend_h,
            "iter_total_s": iter_elapsed_s,
            "llm_usage": feedback.get("llm_usage", {}),
        }
        history.append(history_entry)

    print(
        f"\n⏹ Iteration budget exhausted ({args.max_iters} iters). "
        f"best_clean={ {b: best_clean[b]['iter'] for b in best_clean} }",
        flush=True,
    )

    # ---- Promote per-backend best ----
    final_dir = _final_dir(args.operator, args.model, args.effort)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    # Carry impl_torch.py + config.yaml from the operator dir as canonical.
    _copy_framework_files(args.operator, final_dir)

    promotion: dict[str, dict] = {}
    for b in BACKENDS:
        # Prefer best_clean; fall back to best_any (with a warning).
        src = best_clean.get(b) or best_any.get(b)
        if src is None:
            print(f"\n⚠ No iter produced a working `{b}` impl — final/ will lack impl_{b}.py.", flush=True)
            continue
        if best_clean.get(b) is None:
            print(f"\n⚠ No verify-clean `{b}` iter — promoting buggy best_any (iter {src['iter']}).", flush=True)
        src_iter_dir = _iter_dir(args.operator, args.model, args.effort, src["iter"])
        src_file = src_iter_dir / f"impl_{b}.py"
        if src_file.exists():
            shutil.copy2(src_file, final_dir / f"impl_{b}.py")
            promotion[b] = {
                "iter": src["iter"],
                "stop_score": src["stop_score"],
                "verify_clean": b in best_clean,
            }
            print(f"📦 Promoted `{b}` from iter {src['iter']} "
                  f"(stop_score={src['stop_score']*100:.1f}%) → {final_dir}", flush=True)

    # Write a top-level run summary alongside iter dirs.
    run_summary = {
        "op": args.operator,
        "model": args.model,
        "effort": args.effort,
        "max_iters": args.max_iters,
        "backends": list(BACKENDS),
        "history": history,
        "best_clean": {
            b: {k: v for k, v in best_clean[b].items() if not k.startswith("impl_")}
            for b in best_clean
        },
        "best_any": best_any,
        "promotion": promotion,
    }
    run_path = LLM_GENERATED_ROOT / args.operator / args.model / args.effort / "run_summary.json"
    run_path.write_text(json.dumps(run_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
