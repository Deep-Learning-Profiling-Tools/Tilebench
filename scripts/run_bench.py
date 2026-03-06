import argparse
import json
from core.engine import run_benchmark_suite

_TIMING_KEYS = {
    "params", "problem_size", "dtype",
    "torch_ms", "torch_stats",
    "triton_ms", "triton_stats", "triton_ok", "triton_err",
    "cutile_ms", "cutile_stats", "cutile_ok", "cutile_err",
    "speedup_triton", "speedup_cutile",
}
_AUTOTUNE_KEYS = {
    "params", "problem_size", "dtype",
    "triton_autotune_cfg", "cutile_autotune_cfg",
}


def _split(results: list[dict]) -> tuple[list[dict], list[dict]]:
    timing = [{k: v for k, v in r.items() if k in _TIMING_KEYS} for r in results]
    autotune = [{k: v for k, v in r.items() if k in _AUTOTUNE_KEYS} for r in results]
    return timing, autotune


def main():
    parser = argparse.ArgumentParser(description="Run TileBench benchmarks")
    parser.add_argument("--operator", type=str, default="vector_add",
                        help="Operator to benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for timing results "
                             "(default: results/logs/time_measurement_logs/<operator>_results.json)")
    parser.add_argument("--autotune-log", type=str, default=None,
                        help="Output path for autotune config log "
                             "(default: results/logs/autotune_logs/<operator>_autotune.json)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Warmup iterations (default: 20)")
    parser.add_argument("--repeat", type=int, default=None,
                        help="Measured iterations inside Proton scope (default: 100)")
    parser.add_argument("--use-cuda-graph", action="store_true",
                        help="Capture CUDA graph; replay inside Proton scope")
    parser.add_argument("--proton-scope-name", type=str, default=None,
                        help="Proton scope name (default: launch)")
    parser.add_argument("--proton-backend", type=str, default=None,
                        help="Proton backend, e.g. cupti")
    parser.add_argument("--proton-context", type=str, default=None,
                        help="Proton context: shadow or python")
    parser.add_argument("--flush-l2", action="store_true",
                        help="Best-effort L2 flush before each iteration")
    parser.add_argument("--case-indices", type=str, default=None,
                        help="Comma-separated case indices to run, e.g. 0,1,3")
    parser.add_argument("--keep-proton-files", action="store_true",
                        help="Keep intermediate Proton .hatchet files for inspection")
    parser.add_argument("--proton-output-dir", type=str, default=None,
                        help="Directory to store kept Proton files (default: system temp dir)")
    args = parser.parse_args()

    overrides: dict = {}
    if args.warmup is not None:
        overrides["warmup"] = args.warmup
    if args.repeat is not None:
        overrides["repeat"] = args.repeat
    if args.use_cuda_graph:
        overrides["use_cuda_graph"] = True
    if args.proton_scope_name is not None:
        overrides["proton_scope_name"] = args.proton_scope_name
    if args.proton_backend is not None:
        overrides["proton_backend"] = args.proton_backend
    if args.proton_context is not None:
        overrides["proton_context"] = args.proton_context
    if args.flush_l2:
        overrides["flush_l2"] = True
    if args.case_indices is not None:
        overrides["case_indices"] = [
            int(x.strip()) for x in args.case_indices.split(",") if x.strip()
        ]
    if args.keep_proton_files:
        overrides["keep_proton_files"] = True
    if args.proton_output_dir is not None:
        overrides["proton_output_dir"] = args.proton_output_dir

    # Resolve output paths (operator-bound defaults)
    output_path = args.output or (
        f"results/logs/time_measurement_logs/{args.operator}_results.json"
    )
    autotune_path = args.autotune_log or (
        f"results/logs/autotune_logs/{args.operator}_autotune.json"
    )

    print(f"Starting benchmark for operator: {args.operator}")
    results = run_benchmark_suite(args.operator, benchmark_overrides=overrides)

    timing_results, autotune_results = _split(results)

    with open(output_path, "w") as f:
        json.dump(timing_results, f, indent=4)
    print(f"Timing results  → {output_path}")

    with open(autotune_path, "w") as f:
        json.dump(autotune_results, f, indent=4)
    print(f"Autotune log    → {autotune_path}")

    print("\nSummary:")
    print(
        f"{'Params':>20} | {'Dtype':>8} | {'Torch(ms)':>10} | "
        f"{'Triton(ms)':>10} | {'cuTile(ms)':>10} | {'Speedup(T)':>10} | {'Speedup(C)':>10}"
    )
    print("-" * 95)
    for r in timing_results:
        print(
            f"{str(r['params']):>20} | {r['dtype']:8s} | {r['torch_ms']:10.4f} | "
            f"{r['triton_ms']:10.4f} | {r['cutile_ms']:10.4f} | "
            f"{r['speedup_triton']:10.2f} | {r['speedup_cutile']:10.2f}"
        )


if __name__ == "__main__":
    main()
