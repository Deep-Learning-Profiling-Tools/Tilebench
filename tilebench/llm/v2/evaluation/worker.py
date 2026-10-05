"""Isolated evaluation worker (subprocess entry point).

    python -m tilebench.llm.v2.evaluation.worker --job job.json --out result.json

job.json: {"operator", "dtype", "params", "dsl", "source_path", "sandbox_dir",
           "atol", "rtol", "rules": evaluator_rules, "timing": {"warmup", "repeat",
           "use_cuda_graph", "flush"}, "seed": int}

Isolation actually enforced here (and nothing more):
- a fresh process per candidate, so a fatal device error cannot poison the
  orchestrator or later candidates;
- the generated file is copied into sandbox_dir, cwd is sandbox_dir, and the
  DSL compile caches are redirected into it (TRITON_CACHE_DIR,
  CUDA_TILE_CACHE_DIR, TILELANG_CACHE_DIR), so generated code cannot poison
  shared caches;
- every environment variable whose name contains KEY, TOKEN, SECRET or
  PASSWORD is removed before the worker starts (done by the launcher);
- the worker never imports tilebench.llm.* evaluator state; it only imports
  the generated module, tilebench.core.verifier and tilebench.data.tensors.
NOT enforced (listed honestly): no network namespace, no seccomp, no
read-only filesystem, no memory/time cgroup beyond the launcher's wall-clock
timeout. Generated code could in principle read the repository.

Stage records distinguish import/compile (first run on fresh inputs),
verification runs (3 checks), graph preparation runs and the formal
1 warmup + 3 timed launches; nothing is folded into another count."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

RESULT_STATUSES = ("valid", "compile_error", "runtime_error", "numerical_error", "timing_error",
                   "infrastructure_incomplete")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run_job(job: dict) -> dict:
    t0 = time.time()
    result: dict = {"schema": "tilebench-llm-v2-eval/1", "status": "infrastructure_incomplete",
                    "stages": {}, "diagnostic": None, "config": None, "latency_ms_mean": None,
                    "latency_ms_samples": None, "timing": None}
    try:
        import torch
        from tilebench.core.dtypes import resolve_dtype
        from tilebench.data.tensors import get_generator
        from tilebench.llm.v2.evaluation.anticache import run_numerical_checks
        from tilebench.llm.v2.evaluation.timing import measure
    except Exception:  # noqa: BLE001
        result["diagnostic"] = "worker import failure:\n" + traceback.format_exc()
        return result
    if not torch.cuda.is_available():
        result["diagnostic"] = "no CUDA/HIP device visible to the worker"
        return result
    sync = torch.cuda.synchronize
    op, dtype_str, params = job["operator"], job["dtype"], job["params"]
    dtype = resolve_dtype(dtype_str)
    gen = get_generator(op)
    seed = int(job.get("seed", 0))

    def make_inputs():
        nonlocal seed
        torch.manual_seed(seed)
        seed += 1
        inp = gen(**params, dtype=dtype)
        return inp if isinstance(inp, tuple) else (inp,)

    # reference (functional semantics) comes from the manual impl_torch of the operator
    try:
        from tilebench.paths import operator_dir
        ref_mod = _load_module(operator_dir(op) / "impl_torch.py")
    except Exception:  # noqa: BLE001
        result["diagnostic"] = "reference import failure:\n" + traceback.format_exc()
        return result

    # 1. import / compile (first execution on fresh inputs)
    stage = {"started": time.time()}
    try:
        cand = _load_module(Path(job["source_path"]))
        if not hasattr(cand, "run"):
            raise AttributeError("generated module defines no run()")
        inputs = make_inputs()
        out = cand.run(*inputs)
        sync()
        stage["ok"] = True
        stage["executions"] = 1
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        result["status"] = "compile_error" if "import" in tb.lower() or "compil" in tb.lower() else "runtime_error"
        result["diagnostic"] = tb[-8000:]
        result["stages"]["import_compile"] = {**stage, "ok": False}
        return result
    result["stages"]["import_compile"] = stage
    try:
        cfg = cand.get_last_config() if hasattr(cand, "get_last_config") else None
        result["config"] = cfg if isinstance(cfg, dict) else None
    except Exception:  # noqa: BLE001
        result["config"] = None

    # 2. numerical checks on fresh inputs / same-address refill / repeat
    rules = job.get("rules", {})
    restore_required = bool(rules.get("mutation", {}).get("restore_required", False))
    try:
        checks = run_numerical_checks(cand.run, ref_mod.run, make_inputs, atol=job["atol"], rtol=job["rtol"],
                                      restore_required=restore_required, sync=sync)
    except Exception:  # noqa: BLE001
        result["status"] = "runtime_error"
        result["diagnostic"] = traceback.format_exc()[-8000:]
        return result
    result["stages"]["numerical_checks"] = {"executions_candidate": len(checks), "executions_reference": len(checks),
                                            "checks": [c.to_dict() for c in checks]}
    failed = [c for c in checks if not c.ok]
    if failed:
        result["status"] = "numerical_error"
        result["diagnostic"] = "\n".join(f"{c.name}: {c.message}" for c in failed)[-8000:]
        return result

    # 3. timing on fixed inputs: 1 warmup + 3 timed; restore mutated inputs outside the scope
    try:
        timed_inputs = make_inputs()
        snap = [x.clone() if isinstance(x, torch.Tensor) else x for x in timed_inputs]

        def before_launch():
            if restore_required:
                for x, s in zip(timed_inputs, snap):
                    if isinstance(x, torch.Tensor):
                        x.copy_(s)

        t = job.get("timing", {})
        rec = measure(lambda: cand.run(*timed_inputs), warmup=int(t.get("warmup", 1)), repeat=int(t.get("repeat", 3)),
                      use_cuda_graph=bool(t.get("use_cuda_graph", True)), flush=bool(t.get("flush", True)),
                      before_launch=before_launch if restore_required else None,
                      proton_output_dir=job.get("sandbox_dir"))
        result["timing"] = rec.to_dict()
        if rec.mean_ms is None or rec.mean_ms <= 0:
            result["status"] = "timing_error"
            result["diagnostic"] = rec.note or "timing produced no positive sample"
            return result
        result["latency_ms_mean"] = rec.mean_ms
        result["latency_ms_samples"] = rec.samples_ms
        result["status"] = "valid"
    except Exception:  # noqa: BLE001
        result["status"] = "timing_error"
        result["diagnostic"] = traceback.format_exc()[-8000:]
    result["elapsed_s"] = time.time() - t0
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    job = json.loads(Path(a.job).read_text())
    if os.environ.get("TILEBENCH_V2_SANDBOX") != "1":
        print("worker must be started by the launcher (TILEBENCH_V2_SANDBOX=1)", file=sys.stderr)
        return 2
    res = run_job(job)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
