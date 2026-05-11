"""Run TileBench evaluation for one LLM-generated kernel file.

Loads the specified ``impl_<backend>.py`` via an override path so that the
hand-written canonical implementations are never touched, then runs the full
TileBench correctness + timing suite.

Usage
-----
::

    # Evaluate an LLM-generated Triton kernel.
    PYTHONPATH=. python scripts/run_generated.py \\
        --operator softmax \\
        --backend triton \\
        --impl llm_kernelgen/generated/exp1/softmax/triton/sample_00/impl_triton.py

    # Restrict to a subset of cases.
    PYTHONPATH=. python scripts/run_generated.py \\
        --operator softmax \\
        --backend triton \\
        --impl llm_kernelgen/generated/exp1/softmax/triton/sample_00/impl_triton.py \\
        --case-indices 0,1,2

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from core.engine import run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one LLM-generated kernel with the TileBench engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--operator", required=True,
        help="Operator name (must match a subdirectory in benchmarks/operators/).",
    )
    parser.add_argument(
        "--backend", choices=["triton", "cutile"], required=True,
        help="Target backend.",
    )
    parser.add_argument(
        "--impl", required=True,
        help="Path to the LLM-generated impl_<backend>.py file.",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Path to write the bench.json results file.  "
            "Defaults to bench.json in the same directory as --impl."
        ),
    )
    parser.add_argument(
        "--case-indices", default=None,
        help=(
            "Comma-separated list of case indices to run "
            "(e.g. '0,1,2').  Runs all cases when omitted."
        ),
    )
    parser.add_argument(
        "--autotune", action="store_true",
        help="Pass autotune=True to the kernel's run() function.",
    )
    args = parser.parse_args()

    impl_path = Path(args.impl).resolve()
    if not impl_path.exists():
        print(f"ERROR: impl file not found: {impl_path}", file=sys.stderr)
        sys.exit(1)

    # Build benchmark overrides.
    benchmark_overrides: dict = {}
    if args.case_indices is not None:
        indices = [
            int(x.strip())
            for x in args.case_indices.split(",")
            if x.strip()
        ]
        benchmark_overrides["case_indices"] = indices
    if args.autotune:
        benchmark_overrides["autotune"] = True

    # Build impl override for the selected backend only.
    impl_overrides = {args.backend: str(impl_path)}

    print(f"Running benchmark: operator={args.operator} backend={args.backend}")
    print(f"  impl override: {impl_path}")

    results = run_benchmark_suite(
        args.operator,
        benchmark_overrides=benchmark_overrides or None,
        impl_overrides=impl_overrides,
    )

    # Write results JSON.
    out_path = (
        Path(args.output)
        if args.output
        else impl_path.with_name("bench.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Wrote {out_path}  ({len(results)} cases)")

    # Quick summary to stdout.
    for r in results:
        ok_key = f"{args.backend}_ok"
        ms_key = f"{args.backend}_ms"
        ok  = r.get(ok_key, False)
        ms  = r.get(ms_key, float("nan"))
        params = r.get("params", {})
        dtype  = r.get("dtype", "?")
        print(f"  {params} dtype={dtype}  ok={ok}  ms={ms:.4f}")


if __name__ == "__main__":
    main()
