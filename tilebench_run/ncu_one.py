"""Run NCU for one operator.

Usage:
  python tilebench_run/ncu_one.py <op> [<dtype>] [--backend triton|cutile|both]

Reads autotune winners / kernel counts from the catalogue + kernel_counts.json
and runs NCU at the sweep-max input case (the same case used by the global
sweep). Outputs to tilebench_run/ncu/<op>/<backend>_<dtype>.ncu-rep.

Examples:
  # Both backends, all dtypes:
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

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
NCU = "/usr/local/cuda/bin/ncu"
HARNESS = ROOT / "tilebench_run" / "ncu_generic_harness.py"
CATALOGUE = ROOT / "tilebench_run" / "ncu_catalogue.json"
KERNEL_COUNTS = ROOT / "tilebench_run" / "ncu" / "kernel_counts.json"
OUT_DIR = ROOT / "tilebench_run" / "ncu"


def run_one(op: str, backend: str, dtype: str, params: dict,
            cfg: dict | None, n_kernels: int) -> int:
    out_path = OUT_DIR / op / f"{backend}_{dtype}.ncu-rep"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["NCU_OP"] = op
    env["NCU_BACKEND"] = backend
    env["NCU_DTYPE"] = dtype
    env["NCU_PARAMS_JSON"] = json.dumps(params)
    if cfg is not None:
        env["NCU_CFG_JSON"] = json.dumps(cfg)
    env["PYTHONPATH"] = str(ROOT)
    skip, count = 3 * n_kernels, n_kernels
    cmd = [
        NCU, "--set", "full", "--import-source", "on",
        "--launch-skip", str(skip), "--launch-count", str(count),
        "--force-overwrite",
        "-o", str(out_path).removesuffix(".ncu-rep"),
        sys.executable, str(HARNESS),
    ]
    print(f"→ {op}/{backend}/{dtype}  N={n_kernels}  cfg={cfg}")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, cwd=str(ROOT)).returncode
    dt = time.time() - t0
    print(f"  rc={rc}, {dt:.1f}s, out={out_path}")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("op")
    ap.add_argument("dtype", nargs="?", default=None,
                    help="dtype (e.g. fp16, fp32). Omit for all dtypes.")
    ap.add_argument("--backend", choices=["triton", "cutile", "both"],
                    default="both")
    args = ap.parse_args()

    catalogue = json.loads(CATALOGUE.read_text())
    op_entry = next((c for c in catalogue if c["op"] == args.op), None)
    if op_entry is None:
        sys.exit(f"error: op {args.op!r} not in catalogue")

    kc = {}
    if KERNEL_COUNTS.exists():
        for r in json.loads(KERNEL_COUNTS.read_text()):
            if r.get("count") is not None:
                kc[(r["op"], r["dtype"], r["backend"])] = r["count"]

    dtypes = [args.dtype] if args.dtype else op_entry["dtypes"]
    backends = ["triton", "cutile"] if args.backend == "both" else [args.backend]

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
            rc_total |= run_one(args.op, be, dt, dict(params), cfg, n)
    sys.exit(rc_total)


if __name__ == "__main__":
    main()
