"""LLM code-generation main loop.

Usage:
    PYTHONPATH=. python tools/llm_codegen/generate.py \
        --operator vector_add --model gpt-5.5 --max-iters 10 --threshold 0.8

For each iteration:
  1. Build a prompt (initial vs feedback).
  2. Call the LLM (cached: skip if iter_N/response.md already exists).
  3. Parse out impl_triton.py + impl_cutile.py.
  4. Copy impl_torch.py + config.yaml from benchmarks/operators/<op>/.
  5. Run evaluator (subprocess; 15-min per-case autotune cap).
  6. Save feedback.json. Check stopping condition.
  7. If stopped: promote to benchmarks/llm_generated/<op>/<model>/final/.

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
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.llm_codegen.evaluator import (
    evaluate,
    is_backend_stopping_met,
    is_backend_verify_clean,
)
from tools.llm_codegen.llm_client import LLMClient
from tools.llm_codegen.prompt_builder import (
    SYSTEM, build_feedback_prompt, build_initial_prompt, parse_response,
)

BACKENDS: tuple[str, ...] = ("triton", "cutile")


def _iter_dir(op: str, model: str, effort: str, iter_idx: int) -> Path:
    return (
        _REPO_ROOT / "benchmarks" / "llm_generated"
        / op / model / effort / f"iter_{iter_idx}"
    )


def _final_dir(op: str, model: str, effort: str) -> Path:
    return _REPO_ROOT / "benchmarks" / "llm_generated" / op / model / effort / "final"


def _copy_framework_files(op: str, dest: Path) -> None:
    """Bring impl_torch.py and config.yaml into the iter dir so the evaluator
    has everything it needs in one place."""
    src = _REPO_ROOT / "benchmarks" / "operators" / op
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("impl_torch.py", "config.yaml"):
        shutil.copy2(src / fname, dest / fname)


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
    ae = feedback.get("autotune_errors", {})
    bad_autotune = [b for b in ae if b in active_backends]
    if bad_autotune:
        parts.append(f"autotune_timeout={bad_autotune}")
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
    frozen_info: dict | None = None,
    history: list[dict] | None = None,
    best_so_far: dict | None = None,
) -> dict:
    """Run one iteration: build prompt → LLM → parse → eval. Only generates
    impl files for `active_backends`; frozen backends are skipped end-to-end.

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
            frozen_info=frozen_info,
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

    # ---- Evaluate (skip frozen backends entirely) ----
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
    ap.add_argument(
        "--threshold", type=float, default=0.80,
        help="Geo-mean roofline_pct to stop (default 0.80 = 80%%)",
    )
    args = ap.parse_args()

    client = LLMClient(model=args.model, effort=args.effort)

    print(
        f"=== LLM codegen: op={args.operator} model={args.model} "
        f"effort={args.effort} threshold={args.threshold} ===",
        flush=True,
    )

    prev_feedback = None
    history: list[dict] = []
    # Per-backend state.
    #   frozen[b]: iter index where backend `b` first met the stop condition,
    #              or None if not yet frozen. Once frozen, `b` is skipped end-
    #              to-end in subsequent iters (no LLM, no eval).
    #   best_clean[b]: highest stop_score_<b> among verify-clean iters for `b`
    #                  (carries impl source so the feedback prompt can show it
    #                  for regression-recovery), plus iter index.
    #   best_any[b]:   highest stop_score_<b> regardless of verify cleanliness
    #                  (fallback for promotion).
    frozen: dict[str, int | None] = {b: None for b in BACKENDS}
    frozen_score: dict[str, float] = {b: 0.0 for b in BACKENDS}
    best_clean: dict[str, dict] = {}
    best_any: dict[str, dict] = {}

    for i in range(args.max_iters):
        active_backends = tuple(b for b in BACKENDS if frozen[b] is None)
        if not active_backends:
            print(f"\n✅ All backends frozen by iter {i-1}; stopping pipeline.", flush=True)
            break

        frozen_info = {
            b: {"iter": frozen[b], "stop_score": frozen_score[b]}
            for b in BACKENDS if frozen[b] is not None
        }
        print(f"\n=== Iteration {i} (active={list(active_backends)}, "
              f"frozen={list(frozen_info.keys())}) ===", flush=True)
        iter_t0 = time.time()
        try:
            feedback = run_one_iter(
                op=args.operator, model=args.model, effort=args.effort,
                iter_idx=i, client=client, prev_feedback=prev_feedback,
                active_backends=active_backends,
                frozen_info=frozen_info or None,
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

        # ---- Track per-backend history + bests + freezing ----
        per_backend_h: dict[str, dict] = {}
        for b in BACKENDS:
            if b not in active_backends:
                per_backend_h[b] = {"skipped": True, "frozen_at": frozen[b]}
                continue
            score = feedback.get(f"stop_score_{b}", 0.0)
            rep = feedback.get(f"report_arith_mean_{b}", 0.0)
            vf_count = len(feedback.get(f"verify_failures_{b}", []))
            clean = is_backend_verify_clean(feedback, b)
            per_backend_h[b] = {
                "skipped": False,
                "stop_score": score,
                "report_mean": rep,
                "verify_clean": clean,
                "verify_fail_count": vf_count,
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
            # Freeze if this iter hit threshold AND is verify-clean for `b`.
            if is_backend_stopping_met(feedback, b, threshold=args.threshold):
                frozen[b] = i
                frozen_score[b] = score
                print(f"  🥶 `{b}` froze at iter {i} (stop_score={score*100:.1f}%)", flush=True)

        history_entry = {
            "iter": i,
            "active_backends": list(active_backends),
            "per_backend": per_backend_h,
            "iter_total_s": iter_elapsed_s,
            "llm_usage": feedback.get("llm_usage", {}),
        }
        history.append(history_entry)

        if all(frozen[b] is not None for b in BACKENDS):
            print(f"\n✅ All backends frozen at iter {i}; stopping pipeline.", flush=True)
            break
    else:
        print(
            f"\n⏹ Max iterations ({args.max_iters}) reached. "
            f"frozen={ {b: frozen[b] for b in BACKENDS if frozen[b] is not None} } | "
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
        "threshold": args.threshold,
        "max_iters": args.max_iters,
        "backends": list(BACKENDS),
        "history": history,
        "frozen": {b: frozen[b] for b in BACKENDS},
        "best_clean": {
            b: {k: v for k, v in best_clean[b].items() if not k.startswith("impl_")}
            for b in best_clean
        },
        "best_any": best_any,
        "promotion": promotion,
    }
    run_path = _REPO_ROOT / "benchmarks" / "llm_generated" / args.operator / args.model / args.effort / "run_summary.json"
    run_path.write_text(json.dumps(run_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
