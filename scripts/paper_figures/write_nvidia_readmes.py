"""Write artifacts/paper_figures/nvidia/<device>/README.md from the built tables (numbers are read, not typed).

Usage: python scripts/paper_figures/write_nvidia_readmes.py --root artifacts/paper_figures/nvidia --cache <cache>
"""
import argparse
import collections
import csv
import gzip
import json
import os
from pathlib import Path

csv.field_size_limit(1 << 30)

LIMITS = {
    "B200": [
        "Formal latency: PyTorch/Triton/cuTile columns come from the frozen paper campaign (config 20/100; fixed 64 MB L2 eviction for 41 operators, below the 126.5 MB L2). The TileLang column is a direct runtime measured in a later campaign (PR #319), so cross-DSL ratios involving TileLang also carry run-to-run and environment differences.",
        "Profiling source commit is not recorded for any B200 report (`code_match_status = not_recorded`). Triton/cuTile reports were captured 2026-08 with NCU 2026.1.1; the cuda-tile version at capture time is not recorded (paper campaign: cuda-tile 1.3.0).",
        "TileLang reports were collected on a different host/environment (HF PR #2: CUDA compiler 13.0.88, cuda-tile 1.3.0, TileLang 0.1.11). 8 are reduced kernel-replay collections and 5 are targeted metric lists: they have no per-opcode or PC-sampling data, and kernel-replay cache metrics are not methodologically identical to application replay.",
        "Triton/cuTile profiles capture only the operator kernels; PyTorch helper launches inside `run()` (fill/copy) are not in the report (see `profile_index.notes`, 9 profiles), although the formal timing covers the whole `run()`.",
        "TileLang source correlation points to the generated `tvm_kernels.cu`, which is not embedded in the reports; source-line columns therefore identify generated-code lines only.",
        "Triton `matmul_fp32_fp16_fp8`, `matmul_int8`, `batched_matmul`, `streamk_matmul` pre-transpose B on the host and cache it (`_bt_cache`): the transpose is outside the timed region and the NCU range.",
    ],
    "GH200": [
        "Formal latency: all four backends measured in one run with warmup 1 / repeat 3 and a 120 MiB eviction; this differs from the B200 protocol (see environment.json).",
        "Reports: NCU 2025.4.0, `--set full`, application replay, collected on GH200 and downloaded from Hugging Face for this offline extraction. No GH200 GPU work was performed for this package.",
        "Profiling sources (638ea849 Triton/cuTile, d6ddb622 TileLang, 5610f18f Triton bitonic_sort) differ from the benchmark sources (c882fe50, 3c5eccbf) only in profiling infrastructure, plus the documented TileLang destindex default-config split (PROVENANCE.md on archive/tilebenchpp-2026-10).",
        "cuTile uses cuda-tile 1.5.0 with tileiras 13.4.92 on GH200 (B200 paper campaign: 1.3.0), so cross-device cuTile differences mix architecture and compiler version.",
        "TileLang on sm_90 builds a fragment-accumulator body for the 9 operators whose Blackwell path uses TMEM (supports_tmem dispatch); those kernels are not the same source path as on B200.",
        "Triton `matmul_fp32_fp16_fp8`, `matmul_int8`, `batched_matmul`, `streamk_matmul` pre-transpose B on the host (cached, outside timing and the NCU range); Hopper WGMMA operand-layout effects must not be read as pure compiler quality.",
        "No measured peak metadata exists for GH200, so roofline percentages are not provided.",
    ],
}


def size(p):
    b = os.path.getsize(p)
    return f"{b / 1e6:.2f} MB" if b >= 1e5 else f"{b / 1e3:.1f} kB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    for dev in ("B200", "GH200"):
        d = root / dev
        env = json.load(open(d / "environment.json"))
        qa = json.load(open(d / "qa_summary.json"))
        prof = list(csv.DictReader(open(d / "profile_index.csv")))
        nc = collections.Counter()
        with gzip.open(d / "kernel_metrics_long.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                if r["status"] == "not_collected":
                    nc[r["raw_metric_name"]] += 1
        lv = collections.Counter((p["dsl"], p["collection_level"], p["replay_mode"]) for p in prof)
        L = [f"# NVIDIA {dev} profiling data package (schema v1)", "",
             "Offline extraction of existing Nsight Compute reports plus the formal benchmark CSVs. No benchmark, autotuner or profiler was run.",
             "See `../../schema_v1.md` for column definitions and identifier rules.", "",
             "## Provenance", "",
             f"* Hugging Face dataset `{env['hf_dataset']['repo_id']}` (dataset), revision `{env['hf_dataset']['revision']}`, folder `{env['hf_dataset']['folder']}`: "
             f"{env['hf_dataset']['n_reports']} reports, {env['hf_dataset']['n_sha256_match']} sha256-identical to the local files used.",
             f"* NCU versions in the reports: {env['report_sessions']['ncu_versions']}; report creation range: {env['report_sessions']['report_created_range']}.",
             f"* Benchmark campaign: {json.dumps(env['benchmark_campaign'])}",
             f"* Profiling: {json.dumps(env['profiling'])}",
             f"* Operator configs and CSVs read from the checkout at `{env['repo_head_used_for_configs']}`; archive refs {env['archive_refs']}.", "",
             "## Coverage", "",
             "| DSL | collection | replay | profiles |", "|---|---|---|---|"]
        L += [f"| {k[0]} | {k[1]} | {k[2]} | {v} |" for k, v in sorted(lv.items())]
        L += ["", f"QA status: **{qa['status']}** ({sum(c['status'] == 'pass' for c in qa['checks'])}/{len(qa['checks'])} checks pass). Counts: {qa['counts']}.", "",
              "## Files", "", "| file | size |", "|---|---|"]
        for fn in sorted(os.listdir(d)):
            if fn != "README.md":
                L.append(f"| `{fn}` | {size(d / fn)} |")
        L += ["", "## Metrics not collected", "",
              "CORE metrics reported as `status = not_collected` (absent from the report; never inferred), count of shape-representative launches:", ""]
        L += [f"* `{k}`: {v}" for k, v in nc.most_common()] or ["* none"]
        L += ["", "## Comparison limitations", ""] + [f"* {x}" for x in LIMITS[dev]]
        L += ["", "## Reproduce", "", "```bash",
              "# 1. inventory + sha256 (no download for B200; GH200 reports downloaded from HF into an external cache)",
              f"python scripts/paper_figures/nvidia_inventory.py --device {dev} --cache $CACHE --revision {env['hf_dataset']['revision']} --report-root <report dir> [...]",
              "# 2. offline import of every report (ncu --import / ncu_report only)",
              f"/usr/bin/python3 scripts/paper_figures/ncu_extract.py --ncu /opt/nvidia/nsight-compute/2026.1.1/ncu --out-dir $CACHE/extract/{dev} <reports>",
              "# 3. tables, evidence, QA, READMEs",
              f"PYTHONPATH=.:scripts/paper_figures CUDA_VISIBLE_DEVICES= python scripts/paper_figures/build_nvidia_tables.py --device {dev} --repo . --extract-dir $CACHE/extract/{dev} --inventory $CACHE/inventory_{dev}.json --out artifacts/paper_figures/nvidia/{dev}",
              "python scripts/paper_figures/build_nvidia_evidence.py --repo . --root artifacts/paper_figures/nvidia --inventory-dir $CACHE",
              "python scripts/paper_figures/qa_nvidia.py --repo . --root artifacts/paper_figures/nvidia --cache $CACHE",
              "python scripts/paper_figures/write_nvidia_readmes.py --root artifacts/paper_figures/nvidia --cache $CACHE",
              "```", ""]
        (d / "README.md").write_text("\n".join(L))
        print("wrote", d / "README.md")


if __name__ == "__main__":
    main()
