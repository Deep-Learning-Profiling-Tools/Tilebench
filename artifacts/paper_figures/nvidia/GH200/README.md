# NVIDIA GH200 profiling data package (schema v1)

Offline extraction of existing Nsight Compute reports plus the formal benchmark CSVs. No benchmark, autotuner or profiler was run.
See `../../schema_v1.md` for column definitions and identifier rules.

## Provenance

* Hugging Face dataset `bcui2/NCU_report` (dataset), revision `21037737b7e371d38f3d029367dc3967d5b31a23`, folder `NVIDIA_GH200`: 330 reports, 330 sha256-identical to the local files used.
* NCU versions in the reports: {'2025.4.0.0 (build 36690805) (public-release)': 330}; report creation range: ['2026-Oct-03 07:00:17', '2026-Oct-04 23:21:42'].
* Benchmark campaign: {"csv_dir": "results/GH200/csv", "timing": "Proton, CUDA graph", "final_measurement_warmup_repeat": "1/3 (--warmup 1 --repeat 3)", "l2_eviction": "120 MiB (2x 60 MiB L2)", "benchmark_source": "c882fe50 (89 CSVs), 3c5eccbf (batched_matmul_autotune.csv)", "software": "torch 2.10.0+cu130, Triton 3.6.0, cuda-tile 1.5.0 (pip tileiras 13.4.92), TileLang 0.1.11", "tilelang_column": "measured in the same run as the other backends", "raw_logs": "archive/tilebenchpp-2026-10 66046918"}
* Profiling: {"all": "Nsight Compute 2025.4.0, --set full --import-source on --replay-mode application --cache-control none --app-replay-mode strict; 3 warmups + 120 MiB eviction outside the range, one impl.run() inside (PROVENANCE.md)", "sources": "Triton 108 + cuTile 110: 638ea849; Triton bitonic_sort fp16/fp32: 5610f18f; TileLang 110: d6ddb622", "metadata_archive": "archive/tilebenchpp-2026-10 outputs/profiling/GH200/{ncu_catalogue,kernel_counts}.json, ncu_sweep/{PROVENANCE.md,report_manifest.json}", "tilelang_hopper_path": "9 operators with a Blackwell TMEM path (1d/2d/3d_conv, batched_matmul, block_sparse_attention, flash_attention, matmul_fp32_fp16_fp8, matmul_int8, streamk_matmul) build a fragment-accumulator body on sm_90 (supports_tmem() dispatch)"}
* Operator configs and CSVs read from the checkout at `04d7e45526add29f567930ccbedaa6ac5b7f96ea`; archive refs {'origin/archive/raw-logs-2026-09-18': '9455c0bd76a0b09febee3080ce95f438f600896f', 'origin/archive/tilebenchpp-2026-10': '4c7dc1f08b91e39dbc2b2cf4c2ec591dba0e06f9'}.

## Coverage

| DSL | collection | replay | profiles |
|---|---|---|---|
| cutile | full | application | 110 |
| tilelang | full | application | 110 |
| triton | full | application | 110 |

QA status: **pass** (22/22 checks pass). Counts: {'benchmark_cases': 13200, 'profiles': 330, 'kernel_metric_rows': 357961, 'instruction_mix_rows': 59815, 'pc_hotspot_rows': 66774, 'execution_path_rows': 388, 'diagnosis_rows': 12}.

## Files

| file | size |
|---|---|
| `_build_log.json` | 0.3 kB |
| `benchmark_cases.csv` | 3.63 MB |
| `diagnosis_evidence.csv` | 15.2 kB |
| `environment.json` | 2.5 kB |
| `execution_paths.csv` | 0.16 MB |
| `instruction_mix.csv` | 9.11 MB |
| `kernel_metrics_long.csv.gz` | 4.90 MB |
| `pc_hotspots.csv.gz` | 0.65 MB |
| `profile_index.csv` | 0.46 MB |
| `qa_summary.json` | 4.4 kB |

## Metrics not collected

CORE metrics reported as `status = not_collected` (absent from the report; never inferred), count of shape-representative launches:

* `smsp__thread_inst_executed.sum`: 418

## Comparison limitations

* Formal latency: all four backends measured in one run with warmup 1 / repeat 3 and a 120 MiB eviction; this differs from the B200 protocol (see environment.json).
* Reports: NCU 2025.4.0, `--set full`, application replay, collected on GH200 and downloaded from Hugging Face for this offline extraction. No GH200 GPU work was performed for this package.
* Profiling sources (638ea849 Triton/cuTile, d6ddb622 TileLang, 5610f18f Triton bitonic_sort) differ from the benchmark sources (c882fe50, 3c5eccbf) only in profiling infrastructure, plus the documented TileLang destindex default-config split (PROVENANCE.md on archive/tilebenchpp-2026-10).
* cuTile uses cuda-tile 1.5.0 with tileiras 13.4.92 on GH200 (B200 paper campaign: 1.3.0), so cross-device cuTile differences mix architecture and compiler version.
* TileLang on sm_90 builds a fragment-accumulator body for the 9 operators whose Blackwell path uses TMEM (supports_tmem dispatch); those kernels are not the same source path as on B200.
* Triton `matmul_fp32_fp16_fp8`, `matmul_int8`, `batched_matmul`, `streamk_matmul` pre-transpose B on the host (cached, outside timing and the NCU range); Hopper WGMMA operand-layout effects must not be read as pure compiler quality.
* No measured peak metadata exists for GH200, so roofline percentages are not provided.

## Reproduce

```bash
# 1. inventory + sha256 (no download for B200; GH200 reports downloaded from HF into an external cache)
python scripts/paper_figures/nvidia_inventory.py --device GH200 --cache $CACHE --revision 21037737b7e371d38f3d029367dc3967d5b31a23 --report-root <report dir> [...]
# 2. offline import of every report (ncu --import / ncu_report only)
/usr/bin/python3 scripts/paper_figures/ncu_extract.py --ncu /opt/nvidia/nsight-compute/2026.1.1/ncu --out-dir $CACHE/extract/GH200 <reports>
# 3. tables, evidence, QA, READMEs
PYTHONPATH=.:scripts/paper_figures CUDA_VISIBLE_DEVICES= python scripts/paper_figures/build_nvidia_tables.py --device GH200 --repo . --extract-dir $CACHE/extract/GH200 --inventory $CACHE/inventory_GH200.json --out artifacts/paper_figures/nvidia/GH200
python scripts/paper_figures/build_nvidia_evidence.py --repo . --root artifacts/paper_figures/nvidia --inventory-dir $CACHE
python scripts/paper_figures/qa_nvidia.py --repo . --root artifacts/paper_figures/nvidia --cache $CACHE
python scripts/paper_figures/write_nvidia_readmes.py --root artifacts/paper_figures/nvidia --cache $CACHE
```
