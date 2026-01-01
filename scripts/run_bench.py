import argparse
import json
from core.engine import run_benchmark_suite

def main():
    parser = argparse.ArgumentParser(description="Run TileBench benchmarks")
    parser.add_argument("--operator", type=str, default="vector_add", help="Operator to benchmark")
    parser.add_argument("--output", type=str, default="results/logs/results.json", help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Starting benchmark for operator: {args.operator}")
    results = run_benchmark_suite(args.operator)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output}")
    
    # Print a summary table
    print("\nSummary:")
    print(f"{'Params':>20} | {'Dtype':>8} | {'Torch(ms)':>10} | {'Triton(ms)':>10} | {'cuTile(ms)':>10} | {'Speedup(T)':>10} | {'Speedup(C)':>10}")
    print("-" * 95)
    for r in results:
        param_str = str(r['params'])
        print(f"{param_str:>20} | {r['dtype']:8s} | {r['torch_ms']:10.4f} | {r['triton_ms']:10.4f} | {r['cutile_ms']:10.4f} | {r['speedup_triton']:10.2f} | {r['speedup_cutile']:10.2f}")

if __name__ == "__main__":
    main()
