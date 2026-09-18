"""NCU sweep driver — iterate ncu_catalogue.json, run NCU on each
(op, dtype, backend) triple at sweep-max input.

For each op/dtype:
  - Use autotune-winner cfg from catalogue if present; else use default
    (no _DEFAULT_CONFIG override).
  - Run NCU on both backends.
  - Report under outputs/ncu/<op>/{backend}_{dtype}.ncu-rep.
  - On failure: log and continue.

Progress is appended to outputs/ncu/sweep_log.json after each pair.
Run with: PYTHONPATH=. python tilebench/profiling/ncu_driver.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tilebench.profiling import ncu_kernel_select as ks
from tilebench.paths import NCU_CATALOGUE, NCU_OUTPUT_ROOT, PROFILING_ROOT, REPO_ROOT

ROOT = REPO_ROOT
NCU = "/usr/local/cuda/bin/ncu"
HARNESS = PROFILING_ROOT / "ncu_generic_harness.py"
CATALOGUE = NCU_CATALOGUE
NCU_DIR = NCU_OUTPUT_ROOT
LOG_PATH = Path(os.environ.get("NCU_SWEEP_LOG", NCU_DIR / "sweep_log.json"))
FAIL_PATH = NCU_DIR / "sweep_failures.md"

KERNEL_REGEX_BY_BACKEND_DEFAULT = ".*"


def out_path(op: str, backend: str, dtype: str) -> Path:
    return NCU_DIR / op / f"{backend}_{dtype}.ncu-rep"


def run_one(op: str, dtype: str, backend: str, params: dict, cfg: dict | None,
            n_kernels_per_call: int = 1, timeout_s: int = 1800,
            kernel_names: list[str] | None = None) -> dict:
    out = out_path(op, backend, dtype)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["NCU_OP"] = op
    env["NCU_BACKEND"] = backend
    env["NCU_DTYPE"] = dtype
    env["NCU_PARAMS_JSON"] = json.dumps(params)
    if cfg is not None:
        env["NCU_CFG_JSON"] = json.dumps(cfg)
    env["PYTHONPATH"] = str(ROOT)

    # Harness structure: 3 warmups + 256 MB L2 eviction OUTSIDE the profiler
    # range, then exactly ONE measured impl.run() inside cudaProfilerStart/
    # Stop. The range therefore contains only the measured run's launches:
    # nothing to skip, and `count` = matched launches of one complete run.
    rgx = ks.kernel_regex(kernel_names)
    if rgx:
        count = ks.real_kernel_count(kernel_names) or n_kernels_per_call
    else:
        print(f"  WARNING {op}/{dtype}/{backend}: no kernel names — FRAGILE "
              f"launch-order capture (cannot validate names)", flush=True)
        count = n_kernels_per_call
    skip = 0

    cmd = [
        NCU, "--set", "full", "--import-source", "on",
        # Unified methodology: application replay + no NCU cache control, so
        # each replay pass re-executes the whole harness (warmups + manual
        # eviction + one measured run) and intra-operator producer-consumer
        # L2 state is preserved; the manual eviction supplies the cold entry.
        "--replay-mode", "application", "--cache-control", "none",
        # strict: every filtered kernel must match across all replay passes in
        # the exact order, so a pass that diverges fails loudly instead of
        # silently mixing kernels from different passes.
        "--app-replay-mode", "strict",
        # Profile only kernels inside the harness's cudaProfilerStart/Stop
        # region (excludes input-generator launches, warmups, and the
        # eviction kernel from NCU's launch counter).
        "--profile-from-start", "off",
    ]
    # Select the op's compute kernel(s) by NAME when known (robust against
    # variable aux launch counts). With --kernel-name, ncu's launch counter
    # counts only matched kernels inside the range.
    if rgx:
        cmd += ["--kernel-name", f"regex:{rgx}"]
    cmd += [
        "--launch-skip", str(skip), "--launch-count", str(count),
        "--force-overwrite",
        "-o", str(out).removesuffix(".ncu-rep"),
        sys.executable, str(HARNESS),
    ]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, env=env, cwd=str(ROOT),
            capture_output=True, text=True, timeout=timeout_s,
        )
        elapsed = time.time() - t0
        ok = (r.returncode == 0) and out.exists() and out.stat().st_size > 0
        # Hardening: the report must contain EXACTLY the expected matched
        # launch sequence — every name must belong to the expected stems AND
        # the launch count must equal `count` (repeated launches of the same
        # kernel are counted, not deduplicated). The rep file is authoritative
        # (application replay may print ==PROF== lines once per pass);
        # stdout parsing is the fallback when the rep can't be read.
        captured = ks.captured_from_report(out)
        if captured is None:
            captured = ks.captured_kernels((r.stdout or "") + (r.stderr or ""))
        validated, problems = ks.validate_capture(
            captured, kernel_names if rgx else None, expected_count=count)
        if ok and not validated:
            ok = False
        unexpected = problems
        return {
            "op": op, "dtype": dtype, "backend": backend,
            "ok": ok, "rc": r.returncode, "elapsed_s": round(elapsed, 2),
            "captured": captured, "validated": validated, "unexpected": unexpected,
            "stderr_tail": r.stderr[-600:] if not ok else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "op": op, "dtype": dtype, "backend": backend,
            "ok": False, "rc": -1, "elapsed_s": timeout_s,
            "stderr_tail": "TIMEOUT",
        }


def main() -> None:
    catalogue = json.loads(CATALOGUE.read_text())
    NCU_DIR.mkdir(parents=True, exist_ok=True)

    kc_path = NCU_DIR / "kernel_counts.json"
    kernel_counts: dict[tuple[str, str, str], int] = {}
    kernel_names_map: dict[tuple[str, str, str], list] = {}
    if kc_path.exists():
        for r in json.loads(kc_path.read_text()):
            key = (r["op"], r["dtype"], r["backend"])
            if r.get("count") is not None:
                kernel_counts[key] = r["count"]
            if r.get("names"):
                kernel_names_map[key] = r["names"]

    ops_filter = {o.strip() for o in os.environ.get("NCU_OPS", "").split(",") if o.strip()}
    pairs = []
    for c in catalogue:
        op = c["op"]
        if ops_filter and op not in ops_filter:
            continue
        for dt in c["dtypes"]:
            params = c["default_params_per_dtype"][dt]
            winner = c["autotune_winner_per_dtype"].get(dt)
            for backend in ("triton", "cutile"):
                cfg = None
                if winner is not None:
                    cfg = winner.get(backend)
                n = kernel_counts.get((op, dt, backend), 1)
                names = kernel_names_map.get((op, dt, backend))
                pairs.append((op, dt, backend, params, cfg, n, names))

    total = len(pairs)
    log = []
    if LOG_PATH.exists():
        try:
            log = json.loads(LOG_PATH.read_text())
        except Exception:
            log = []
    done_keys = {(e["op"], e["dtype"], e["backend"]) for e in log if e.get("ok")}

    fails: list[dict] = [e for e in log if not e.get("ok")]
    t_start = time.time()

    for i, (op, dt, backend, params, cfg, n, names) in enumerate(pairs):
        key = (op, dt, backend)
        if key in done_keys:
            continue
        out = out_path(op, backend, dt)
        if out.exists() and out.stat().st_size > 0:
            log.append({"op": op, "dtype": dt, "backend": backend, "ok": True,
                        "rc": 0, "elapsed_s": 0, "stderr_tail": "(pre-existing)"})
            done_keys.add(key)
            continue
        res = run_one(op, dt, backend, params, cfg, n_kernels_per_call=n,
                      kernel_names=names)
        res["n_kernels"] = n
        log.append(res)
        if not res["ok"]:
            fails.append(res)
        LOG_PATH.write_text(json.dumps(log, indent=2))

        done = len(log)
        if done % 20 == 0 or done == total:
            ok_n = sum(1 for e in log if e["ok"])
            elapsed = time.time() - t_start
            print(f"[progress] {done}/{total} done, ok={ok_n}, "
                  f"fail={done-ok_n}, elapsed={elapsed/60:.1f} min",
                  flush=True)

    ok_n = sum(1 for e in log if e["ok"])
    print(f"\n=== DONE === {ok_n}/{len(log)} ok, {len(log)-ok_n} failed",
          flush=True)
    if fails:
        lines = ["# NCU sweep failures\n"]
        for f in fails:
            lines.append(
                f"- **{f['op']}/{f['dtype']}/{f['backend']}** "
                f"(rc={f['rc']}, {f['elapsed_s']}s)\n"
                f"  ```\n  {f['stderr_tail'].strip()}\n  ```\n"
            )
        FAIL_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
