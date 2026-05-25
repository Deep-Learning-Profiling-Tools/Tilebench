"""Worker script that imports LLM-generated impls and runs the eval loop.

Invoked as a subprocess by evaluator.py with a wall-clock timeout. Writes the
result dict (one big JSON) to a path passed in via env var. Stays as a
separate script so that:
  - GPU crashes / hangs / segfaults are isolated from the orchestrator
  - autotune timeouts can be enforced as a subprocess SIGTERM
  - the orchestrator never holds any imports from the generated code

Env vars expected:
    LLMGEN_ITER_DIR        directory containing impl_triton.py / impl_cutile.py /
                           impl_torch.py
    LLMGEN_OP              operator name (e.g. "vector_add")
    LLMGEN_OUTPUT_JSON     where to write the result JSON
    LLMGEN_PER_CASE_CAP_S  per-(backend,case) autotune wall-clock cap, default 900
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml


_REPO_ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
sys.path.insert(0, str(_REPO_ROOT))

from core.dtypes import resolve_dtype
from core.verifier import verify
from core.timer import (
    _build_profile_base, _find_scope_mean_ns, _flush_l2_cache, _load_profile_data,
)
try:
    import triton.profiler as proton
except ImportError:
    proton = None
from data.tensors import GENERATORS, expand_cases, infer_problem_size
from tools.llm_codegen.roofline import roofline_pct, load_peak


def _proton_time_rotating(
    impl_run,
    generator,
    case: dict,
    *,
    warmup: int = 5,
    repeat: int = 20,
    flush_l2: bool = True,
    scope_name: str = "launch",
) -> float:
    """Proton-based mean kernel latency (ms), rotating fresh inputs each
    iteration to defeat any output cache the impl might keep. Mirrors
    core/timer.py's report_benchmark structure but takes a sequence of
    independently-generated input tuples instead of a single fixed tuple.
    """
    if proton is None:
        raise RuntimeError("triton.profiler (proton) is not available in this environment.")

    inputs_list = []
    for _ in range(warmup + repeat):
        inp = generator(**case)
        if not isinstance(inp, tuple):
            inp = (inp,)
        inputs_list.append(inp)

    # Warmup outside Proton session.
    for inp in inputs_list[:warmup]:
        if flush_l2:
            _flush_l2_cache()
        impl_run(*inp)
    torch.cuda.synchronize()

    # Measurement session.
    profile_base = _build_profile_base("tree", None, None)
    session_id = proton.start(name=profile_base, context="shadow", data="tree", backend=None)
    try:
        for inp in inputs_list[warmup:]:
            if flush_l2:
                _flush_l2_cache()
            with proton.scope(scope_name):
                impl_run(*inp)
        torch.cuda.synchronize()
    finally:
        proton.finalize(session=session_id)

    # Parse hatchet.
    hatchet_data, path = _load_profile_data(profile_base)
    try:
        os.remove(path)
    except OSError:
        pass
    roots = hatchet_data if isinstance(hatchet_data, list) else [hatchet_data]
    mean_ns = 0.0
    for root in roots:
        t = _find_scope_mean_ns(root, scope_name, repeat)
        if t > 0:
            mean_ns = t
            break
    if mean_ns <= 0:
        raise RuntimeError(
            f"Failed to extract GPU timing from Proton hatchet data: "
            f"scope {scope_name!r} not found or reported zero time."
        )
    return mean_ns / 1e6  # ms


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Forbidden patterns we statically scan impl_triton.py / impl_cutile.py for.
# A match in the source file is treated as a "delegated computation" cheat:
# the LLM bypassed the Triton/cuTile kernel and routed the work through a
# higher-level PyTorch / cuDNN / cuBLAS path.
_FORBIDDEN_PATTERNS = [
    # PyTorch high-level ops (substring match against module-qualified names).
    r"\btorch\.nn\.functional\b",
    r"\bF\.scaled_dot_product_attention\b",
    r"\bF\.softmax\b", r"\bF\.log_softmax\b",
    r"\bF\.layer_norm\b", r"\bF\.batch_norm\b", r"\bF\.rms_norm\b",
    r"\bF\.linear\b",
    r"\bF\.conv1d\b", r"\bF\.conv2d\b", r"\bF\.conv3d\b",
    r"\bF\.max_pool1d\b", r"\bF\.max_pool2d\b", r"\bF\.max_pool3d\b",
    r"\bF\.cross_entropy\b", r"\bF\.kl_div\b", r"\bF\.dropout\b",
    r"\bF\.normalize\b", r"\bF\.silu\b", r"\bF\.relu\b", r"\bF\.sigmoid\b",
    # Top-level torch math ops that are the *whole* operator on their own.
    r"\btorch\.matmul\b", r"\btorch\.mm\b", r"\btorch\.bmm\b",
    r"\btorch\.einsum\b",
    r"\btorch\.softmax\b", r"\btorch\.log_softmax\b",
    r"\btorch\.argmax\b", r"\btorch\.argmin\b",
    r"\btorch\.sort\b", r"\btorch\.topk\b",
    r"\btorch\.bincount\b", r"\btorch\.histc\b",
    r"\btorch\.flip\b",
    r"\btorch\.linalg\b",
    # Importing the reference back-door.
    r"\bimpl_torch\b",
    r"\bfrom\s+\.\s*impl_torch\b",
    # Autotune machinery — the pipeline does its own iterative refinement
    # and the LLM must pick ONE configuration per iteration. Letting the
    # LLM defer to Triton's / cuTile's built-in autotuners would conflate
    # "LLM picked a good cfg" with "autotuner found a good cfg".
    r"\btriton\.autotune\b",
    r"\bCutileAutotuner\b",
    r"\bcore\.cutile_autotune\b",
    r"\bct\.tune\.exhaustive_search\b",
    r"\bct_experimental\.autotune_launch\b",
]


def _scan_for_forbidden(src: str) -> list[str]:
    """Return the list of forbidden-pattern hits found in the given source."""
    hits = []
    for pat in _FORBIDDEN_PATTERNS:
        m = re.search(pat, src)
        if m:
            hits.append(m.group(0))
    return hits


def _try_load_impl(iter_dir: Path, backend: str) -> tuple[object | None, str | None]:
    path = iter_dir / f"impl_{backend}.py"
    if not path.exists():
        return None, f"{path.name} does not exist"
    # Reject delegated-computation cheats before importing.
    if backend in ("triton", "cutile"):
        src = path.read_text()
        hits = _scan_for_forbidden(src)
        if hits:
            return None, (
                f"FORBIDDEN PATTERN DETECTED in {path.name}: "
                f"the source contains {sorted(set(hits))}. "
                "Either (a) your run() delegates the operator's computation "
                "to torch.nn.functional / torch.matmul / cuDNN instead of "
                "your @triton.jit / @ct.kernel — see framework_guide.md "
                "'⛔️ FORBIDDEN: delegating the actual computation'; or "
                "(b) you used triton.autotune / CutileAutotuner — see "
                "'⛔️ No autotune — you pick one configuration per iteration'."
            )
    try:
        return _load_module(f"llmgen_impl_{backend}", path), None
    except Exception:
        return None, traceback.format_exc()


def _resolve_dtype_in_params(params: dict) -> dict:
    if isinstance(params.get("dtype"), str):
        params = dict(params)
        params["dtype"] = resolve_dtype(params["dtype"])
    return params


def _run_one_case(impl, op: str, case: dict, impl_torch_run, per_case_cap_s: int):
    """Run one case for one backend. Returns dict with keys:
        compile_ok, verify_ok, latency_s, torch_latency_s, error, cfg
    """
    case = _resolve_dtype_in_params(case)
    generator = GENERATORS[op]
    inputs = generator(**case)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    torch.cuda.synchronize()
    ref = impl_torch_run(*inputs)
    torch.cuda.synchronize()

    # Bound verify+timing via a SIGALRM so a pathological kernel cfg
    # (infinite loop, OOM compile) cannot stall the whole sweep.
    prev_handler = signal.signal(signal.SIGALRM, _alarm_raise)
    signal.alarm(int(per_case_cap_s))
    try:
        # ---- Verify on the first call ----
        # No autotune in this pipeline: the LLM-chosen cfg is hard-coded in
        # impl.run, and the first call IS the cfg's actual execution.
        output = impl.run(*inputs)
        torch.cuda.synchronize()
        cfg = getattr(impl, "get_last_config", lambda: None)()
        ok, err = verify(output, ref)
        if not ok:
            signal.alarm(0)
            return {
                "compile_ok": True, "verify_ok": False, "latency_s": None,
                "torch_latency_s": None, "error": err, "cfg": cfg,
            }

        # ---- Anti-cache check ----
        # Some LLM-generated impls cache outputs across run() calls
        # (e.g. via an "_OUTPUT_CACHE" dict keyed by tensor identity + version).
        # That would make timing return ~0ms and inflate roofline_pct past 100%.
        # Defense: regenerate fresh inputs (different identity, different
        # storage, fresh version) and verify the impl returns a CORRECT
        # output for the new inputs — a cached impl would return the OLD
        # output and fail verify against the NEW reference.
        fresh_inputs = generator(**case)
        if not isinstance(fresh_inputs, tuple):
            fresh_inputs = (fresh_inputs,)
        torch.cuda.synchronize()
        fresh_ref = impl_torch_run(*fresh_inputs)
        fresh_out = impl.run(*fresh_inputs)
        torch.cuda.synchronize()
        ok2, err2 = verify(fresh_out, fresh_ref)
        if not ok2:
            signal.alarm(0)
            return {
                "compile_ok": True, "verify_ok": False, "latency_s": None,
                "torch_latency_s": None,
                "error": f"verify failed on fresh-input check (likely output-caching cheat): {err2}",
                "cfg": cfg,
            }

        # ---- Timing: Proton with rotating fresh inputs ----
        # Use the same Proton-based GPU-kernel timer as the production engine
        # (core/timer.py), but rotate inputs each iter to defeat output caching.
        latency_ms = _proton_time_rotating(
            impl.run, generator, case,
            warmup=5, repeat=20, flush_l2=True, scope_name="launch",
        )
        # Time the torch reference under the same methodology so we can
        # report `speedup_vs_torch = torch_latency / kernel_latency`.
        torch_latency_ms = _proton_time_rotating(
            impl_torch_run, generator, case,
            warmup=5, repeat=20, flush_l2=True, scope_name="launch",
        )

        signal.alarm(0)
        return {
            "compile_ok": True, "verify_ok": True,
            "latency_s": latency_ms / 1000.0,
            "torch_latency_s": torch_latency_ms / 1000.0,
            "error": None, "cfg": cfg,
        }
    except TimeoutError as e:
        signal.alarm(0)
        return {
            "compile_ok": True, "verify_ok": False, "latency_s": None,
            "torch_latency_s": None,
            "error": f"timeout ({per_case_cap_s}s) — {e}",
            "cfg": None,
        }
    except Exception:
        signal.alarm(0)
        return {
            "compile_ok": True, "verify_ok": False, "latency_s": None,
            "torch_latency_s": None,
            "error": traceback.format_exc(),
            "cfg": None,
        }
    finally:
        signal.signal(signal.SIGALRM, prev_handler)


def _alarm_raise(signum, frame):
    raise TimeoutError("per-case wall-clock cap exceeded")


def _params_repr(params: dict) -> dict:
    """JSON-safe dict (dtypes → strings, drop tensors etc.)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, torch.dtype):
            out[k] = str(v).replace("torch.", "")
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
    return out


def main():
    iter_dir = Path(os.environ["LLMGEN_ITER_DIR"])
    op = os.environ["LLMGEN_OP"]
    output_json = Path(os.environ["LLMGEN_OUTPUT_JSON"])
    per_case_cap_s = int(os.environ.get("LLMGEN_PER_CASE_CAP_S", "900"))
    # Backends to skip this iter (currently unused by the main loop, but
    # kept as a hook for callers that want to evaluate only a subset).
    skip_backends = {
        b for b in os.environ.get("LLMGEN_SKIP_BACKENDS", "").split(",") if b
    }

    # Load config + expand cases.
    config = yaml.safe_load((iter_dir / "config.yaml").read_text())
    cases = expand_cases(op, config)

    # Load impls (skip backends listed in LLMGEN_SKIP_BACKENDS; their
    # impl files may not be present).
    triton_mod, triton_compile_err = (None, None) if "triton" in skip_backends \
        else _try_load_impl(iter_dir, "triton")
    cutile_mod, cutile_compile_err = (None, None) if "cutile" in skip_backends \
        else _try_load_impl(iter_dir, "cutile")
    torch_mod, torch_compile_err = _try_load_impl(iter_dir, "torch")
    if torch_compile_err:
        # Reference can't be loaded → catastrophic.
        result = {
            "fatal": f"impl_torch.py failed to load: {torch_compile_err}",
            "compile_errors": {"torch": torch_compile_err},
        }
        output_json.write_text(json.dumps(result, indent=2))
        return

    peak = load_peak("B200")
    metrics_cfg = config.get("metrics", {})
    flops_expr = metrics_cfg.get("flops_expr")
    bytes_expr = metrics_cfg.get("bytes_expr")
    compile_errors = {"triton": triton_compile_err, "cutile": cutile_compile_err}
    # Per-backend wall-clock-timeout bucket. Set when a single (backend,
    # case) exceeds per_case_cap_s — usually means a pathological cfg
    # (too-small tile, too-deep pipeline) made the kernel run forever.
    case_timeout_errors: dict[str, str] = {}
    verify_failures = []
    roofline_per_combo = []
    eval_total_t0 = time.time()

    for backend, mod in [("triton", triton_mod), ("cutile", cutile_mod)]:
        if mod is None:
            continue

        for case in cases:
            case = dict(case)
            dtype_str = case.get("dtype", "fp32")
            if not isinstance(dtype_str, str):
                dtype_str = str(dtype_str).replace("torch.", "")

            params_for_eval = dict(case)
            params_for_eval["problem_size"] = infer_problem_size(op, case)

            res = _run_one_case(mod, op, case, torch_mod.run, per_case_cap_s)

            params_repr = _params_repr(case)

            if not res["compile_ok"]:
                # Shouldn't reach here — handled at module-load time. Defensive.
                continue

            if not res["verify_ok"]:
                # Distinguish timeout vs verification failure.
                err = res.get("error", "")
                if "timeout" in err.lower() or "exceeded" in err.lower():
                    case_timeout_errors[backend] = err
                else:
                    verify_failures.append({
                        "backend": backend,
                        "dtype": dtype_str,
                        "params": params_repr,
                        "error": str(err)[:1000],
                    })
                continue

            # Compute roofline pct.
            r = roofline_pct(
                flops_expr=flops_expr, bytes_expr=bytes_expr,
                params=params_for_eval, dtype_str=dtype_str,
                latency_s=res["latency_s"], peak=peak,
            )
            kernel_ms = res["latency_s"] * 1000
            torch_ms = (res.get("torch_latency_s") or 0) * 1000
            speedup = (torch_ms / kernel_ms) if (torch_ms > 0 and kernel_ms > 0) else None
            roofline_per_combo.append({
                "backend": backend, "dtype": dtype_str,
                "params": params_repr,
                "problem_size": params_for_eval["problem_size"],
                "latency_ms": kernel_ms,
                "torch_latency_ms": torch_ms if torch_ms > 0 else None,
                "speedup_vs_torch": speedup,
                "cfg": res.get("cfg"),
                **{k: v for k, v in r.items() if k != "error"},
            })

    eval_elapsed_s = time.time() - eval_total_t0

    # ---- Aggregate metrics (per backend) ----
    # The LLM-codegen pipeline trims case_grid to a single largest case per
    # dtype (see _copy_framework_files), so each (backend, dtype) contributes
    # exactly one roofline_pct. report_mean and stop_score are therefore
    # numerically identical here — both are the arithmetic mean of capped
    # roofline_pct across the per-dtype largest case for that backend. We
    # keep both field names for backward compat with downstream readers; the
    # top-K-per-dtype logic that used to differentiate them collapses to a
    # plain mean since K (=1 case/dtype) ≥ 1.
    def _mean_roofline(combos: list[dict]) -> float:
        pcts = [min(r["roofline_pct"], 1.0) for r in combos
                if r.get("roofline_pct", 0) > 0]
        return sum(pcts) / len(pcts) if pcts else 0.0

    combos_triton = [r for r in roofline_per_combo if r["backend"] == "triton"]
    combos_cutile = [r for r in roofline_per_combo if r["backend"] == "cutile"]

    stop_score_triton = report_mean_triton = _mean_roofline(combos_triton)
    stop_score_cutile = report_mean_cutile = _mean_roofline(combos_cutile)

    verify_failures_triton = [v for v in verify_failures if v["backend"] == "triton"]
    verify_failures_cutile = [v for v in verify_failures if v["backend"] == "cutile"]

    n_combos_attempted = len(cases) * (2 - len(skip_backends))

    result = {
        "op": op,
        "iter_dir": str(iter_dir),
        "n_cases": len(cases),
        "n_combos_attempted": n_combos_attempted,
        "n_combos_succeeded": len(roofline_per_combo),
        "skipped_backends": sorted(skip_backends),
        "compile_errors": compile_errors,
        "case_timeout_errors": case_timeout_errors,
        "verify_failures":         verify_failures,
        "verify_failures_triton":  verify_failures_triton,
        "verify_failures_cutile":  verify_failures_cutile,
        "roofline_per_combo": roofline_per_combo,
        # Per-backend score: arithmetic mean of min(roofline_pct, 1.0) over
        # one case per dtype (the largest, as trimmed by
        # generate._trim_case_grid_to_largest). Reported as feedback to the
        # next iteration; no longer used for early stopping.
        # `report_arith_mean_<b>` is kept as an alias of `stop_score_<b>`
        # because downstream readers (run_summary.json post-hoc analysis)
        # already reference both names; they are numerically identical now.
        "report_arith_mean_triton":     report_mean_triton,
        "report_arith_mean_cutile":     report_mean_cutile,
        "stop_score_triton":            stop_score_triton,
        "stop_score_cutile":            stop_score_cutile,
        "timing_breakdown": {
            "evaluator_total_s": eval_elapsed_s,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
