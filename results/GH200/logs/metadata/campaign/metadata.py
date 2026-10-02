"""Write results/GH200/logs/metadata/{environment,campaign_summary}.json (no hand-entered values)."""
import collections, csv, json, subprocess, sys
from importlib import metadata as md
from pathlib import Path
import torch
from tilebench import hardware, provenance
from tilebench.core import timer
R = Path("results/GH200"); M = R / "logs/metadata"
def sh(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"unavailable: {e}"
prov = provenance.collect("GH200")
env = {
    "captured_by": "tilebench.provenance.collect + tilebench.hardware + torch + nvidia-smi + importlib.metadata (no hand-entered values)",
    "note": "captured after the campaign on the same host and environment; prov.source is the checkout at capture time, "
            "the benchmark source of each result is in campaign_summary.json and the per-run provenance sidecars",
    "provenance_at_capture": prov,
    "packages": {p: (md.version(p) if p in {d.metadata["Name"] for d in md.distributions()} else None) for p in
                 ["torch", "triton", "cuda-tile", "tilelang", "apache-tvm-ffi", "nvidia-cuda-tileiras", "nvidia-cuda-nvcc",
                  "nvidia-nvvm", "cuda-bindings", "numpy"]},
    "cuda": {"torch_cuda": torch.version.cuda, "driver_and_gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
             "cuTile_tileiras": sh([sys.executable, "-c", "from cuda.tile._compile import _find_compiler_bin as f; print(f().path)"]),
             "tilelang_CUDA_HOME": sh([sys.executable, "-c", "import tilelang.env as e; print(e.CUDA_HOME)"])},
    "hardware": {"detect_arch": hardware.detect_arch(), "supports_tma": hardware.supports_tma(), "supports_tmem": hardware.supports_tmem(),
                 "last_level_cache_bytes": hardware.last_level_cache_bytes(),
                 "torch_L2_cache_size_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
                 "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                 "l2_flush_buffer_mb": timer._flush_l2_buffer_mb()},
    "cutile_crash_isolation_timeout_sec_at_capture": __import__("tilebench.core.cutile_autotune", fromlist=["x"]).CRASH_ISOLATION_TIMEOUT_SEC,
}
summary = {"protocol": {"runner": "scripts/run_bench.py --gpu GH200 --tile-language triton,cutile,tilelang --warmup 1 --repeat 3 [--autotune]",
                        "config_yaml_benchmark": "use_cuda_graph=true, flush_l2=true (unchanged)",
                        "autotune_candidate_timing": "Triton/TileLang warmup=1 rep=3; cuTile native exhaustive_search"},
           "csv": {}, "source_sha_counts": collections.Counter(), "rows": collections.Counter()}
for mode in ("default", "autotune"):
    for p in sorted((R / "logs/provenance").glob(f"*_{mode}_triton-cutile-tilelang.json")):
        op = p.name.split(f"_{mode}_")[0]
        src = json.load(open(p))["source"]
        rows = list(csv.DictReader(open(R / "csv" / f"{op}_{mode}.csv")))
        summary["csv"][f"{op}_{mode}.csv"] = {"rows": len(rows), "benchmark_source_sha": src["git_sha"], "dirty": src["dirty"]}
        summary["source_sha_counts"][src["git_sha"]] += 1; summary["rows"][mode] += len(rows)
summary["rerun_evidence"] = sorted(str(p.relative_to(R)) for p in (R / "logs/rerun_evidence").iterdir())
summary["result_commit_note"] = "results/GH200/csv is committed on exp/gh200 as a results-only commit; it is not the benchmark source"
M.mkdir(parents=True, exist_ok=True)
json.dump(env, open(M / "environment.json", "w"), indent=1, default=str)
json.dump(summary, open(M / "campaign_summary.json", "w"), indent=1, default=str)
print("csv files:", len(summary["csv"]), "rows:", dict(summary["rows"]), "source SHAs:", dict(summary["source_sha_counts"]))
print("packages:", env["packages"]); print("cuda:", env["cuda"]); print("hardware:", env["hardware"])
