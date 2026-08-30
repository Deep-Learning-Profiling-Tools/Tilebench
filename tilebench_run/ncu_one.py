"""Run NCU for one operator.

Usage:
  python tilebench_run/ncu_one.py <op> [<dtype>] [--backend triton|cutile|tilelang|all|both]

Reads autotune winners / kernel counts from the catalogue + kernel_counts.json
and runs NCU at the sweep-max input case (the same case used by the global
sweep). Outputs to tilebench_run/ncu/<op>/<backend>_<dtype>.ncu-rep.

Examples:
  # All profiling backends, all dtypes:
  python tilebench_run/ncu_one.py matmul_int8

  # Single dtype, both backends:
  python tilebench_run/ncu_one.py matmul_fp32_fp16_fp8 fp32

  # Single backend, single dtype:
  python tilebench_run/ncu_one.py softmax fp16 --backend cutile
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ncu_common import (
    BACKENDS,
    LEGACY_BACKENDS,
    ncu_bin,
    ncu_profile_source_args,
    prepare_profile_env,
    repo_root,
)
import ncu_kernel_select as ks  # sibling module in tilebench_run/ (on sys.path as a script)

ROOT = repo_root()
NCU = ncu_bin()
HARNESS = ROOT / "tilebench_run" / "ncu_generic_harness.py"
CATALOGUE = ROOT / "tilebench_run" / "ncu_catalogue.json"
KERNEL_COUNTS = ROOT / "tilebench_run" / "ncu" / "kernel_counts.json"
OUT_DIR = ROOT / "tilebench_run" / "ncu"


def run_one(op: str, backend: str, dtype: str, params: dict,
            cfg: dict | None, n_kernels: int,
            kernel_names: list[str] | None = None) -> int:
    out_path = OUT_DIR / op / f"{backend}_{dtype}.ncu-rep"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["NCU_OP"] = op
    env["NCU_BACKEND"] = backend
    env["NCU_DTYPE"] = dtype
    env["NCU_PARAMS_JSON"] = json.dumps(params)
    if cfg is not None:
        env["NCU_CFG_JSON"] = json.dumps(cfg)
    env = prepare_profile_env(env, op=op, backend=backend, dtype=dtype)
    rgx = ks.kernel_regex(kernel_names)
    if rgx:
        count = ks.real_kernel_count(kernel_names) or n_kernels
    else:
        print(f"  WARNING {op}/{backend}/{dtype}: no kernel names known — "
              f"FRAGILE launch-order capture (count {n_kernels}); cannot "
              f"validate kernel names.")
        count = n_kernels
    # Harness runs warmups + 256 MB L2 eviction OUTSIDE the profiler range and
    # exactly ONE measured impl.run() inside it — nothing to skip.
    skip = 0
    cmd = [
        NCU, "--set", "full", *ncu_profile_source_args(),
        # Unified methodology (see ncu_driver.py): application replay + no NCU
        # cache control; the harness's manual eviction supplies the cold entry.
        "--replay-mode", "application", "--cache-control", "none",
        # strict: all filtered kernels must match across replay passes in the
        # exact order (see ncu_driver.py).
        "--app-replay-mode", "strict",
        # Profile only inside the harness's cudaProfilerStart/Stop region
        # (excludes generator launches, warmups, and the eviction kernel).
        "--profile-from-start", "off",
    ]
    # Prefer selecting the op's compute kernel(s) by NAME (robust against
    # variable aux launch counts).
    if rgx:
        cmd += ["--kernel-name", f"regex:{rgx}"]
    cmd += [
        "--launch-skip", str(skip), "--launch-count", str(count),
        "--force-overwrite",
        "-o", str(out_path).removesuffix(".ncu-rep"),
        sys.executable, str(HARNESS),
    ]
    print(f"→ {op}/{backend}/{dtype}  N={n_kernels}  cfg={cfg}")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if r.stdout:
        print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="")
    rc = r.returncode
    # Hardening: confirm NCU profiled ONLY the op's own compute kernel(s) — not an
    # aux/wrong kernel a fragile launch-order capture might have grabbed. NCU writes
    # its `==PROF== Profiling "<name>"` progress lines to stdout.
    captured = ks.captured_from_report(out_path)
    if captured is None:
        captured = ks.captured_kernels((r.stdout or "") + (r.stderr or ""))
    ok, problems = ks.validate_capture(
        captured, kernel_names if rgx else None, expected_count=count)
    if not ok:
        rc = rc or 3
        print(f"  CAPTURE VALIDATION FAILED: captured({len(captured)})={captured} "
              f"problems={problems}")
    else:
        print(f"  validated: {len(captured)}/{count} launches, "
              f"names ⊆ {ks.kernel_stems(kernel_names)}")
    print(f"  rc={rc}, {dt:.1f}s, out={out_path}")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("op")
    ap.add_argument("dtype", nargs="?", default=None,
                    help="dtype (e.g. fp16, fp32). Omit for all dtypes.")
    ap.add_argument("--backend", choices=[*BACKENDS, "all", "both"],
                    default="all")
    args = ap.parse_args()

    catalogue = json.loads(CATALOGUE.read_text())
    op_entry = next((c for c in catalogue if c["op"] == args.op), None)
    if op_entry is None:
        sys.exit(f"error: op {args.op!r} not in catalogue")

    kc, kcn = {}, {}
    if KERNEL_COUNTS.exists():
        for r in json.loads(KERNEL_COUNTS.read_text()):
            key = (r["op"], r["dtype"], r["backend"])
            if r.get("count") is not None:
                kc[key] = r["count"]
            if r.get("names"):
                kcn[key] = r["names"]

    dtypes = [args.dtype] if args.dtype else op_entry["dtypes"]
    if args.backend == "all":
        backends = list(BACKENDS)
    elif args.backend == "both":
        backends = list(LEGACY_BACKENDS)
    else:
        backends = [args.backend]

    rc_total = 0
    for dt in dtypes:
        if dt not in op_entry["dtypes"]:
            print(f"warn: dtype {dt!r} not in {args.op}'s catalogue; "
                  f"available: {op_entry['dtypes']}", file=sys.stderr)
            continue
        params = op_entry["default_params_per_dtype"][dt]
        winner = op_entry["autotune_winner_per_dtype"].get(dt) or {}
        for be in backends:
            cfg = winner.get(be)
            n = kc.get((args.op, dt, be), 1)
            names = kcn.get((args.op, dt, be))
            rc_total |= run_one(args.op, be, dt, dict(params), cfg, n, names)
    sys.exit(rc_total)


if __name__ == "__main__":
    main()
