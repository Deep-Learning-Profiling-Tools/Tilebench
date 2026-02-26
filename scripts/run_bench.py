import argparse
import json
import os

from tabulate import tabulate

from core.engine import run_benchmark_suite

def main():
    parser = argparse.ArgumentParser(description="Run TileBench benchmarks")
    parser.add_argument("--operator", type=str, default="vector_add", help="Operator to benchmark")
    parser.add_argument("--output", type=str, default="results/logs/results.json", help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Starting benchmark for operator: {args.operator}")
    results = run_benchmark_suite(args.operator)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output}")
    
    # Print a summary table
    print("\nSummary:")
    rows = []
    for r in results:
        rows.append([
            str(r["params"]),
            r["dtype"],
            f"{r['torch_ms']:.4f}",
            f"{r['triton_ms']:.4f}",
            f"{r['cutile_ms']:.4f}",
            f"{r['speedup_triton']:.2f}",
            f"{r['speedup_cutile']:.2f}",
        ])
    print(
        tabulate(
            rows,
            headers=["Params", "Dtype", "Torch(ms)", "Triton(ms)", "cuTile(ms)", "Speedup(T)", "Speedup(C)"],
            tablefmt="simple",
        )
    )

if __name__ == "__main__":
    main()
