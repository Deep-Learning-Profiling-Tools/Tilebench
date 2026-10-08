# NVIDIA B200 profiling data package (schema v1)

Offline extraction of existing Nsight Compute reports plus the formal benchmark CSVs. No benchmark, autotuner or profiler was run.
See `../../schema_v1.md` for column definitions and identifier rules.

## Provenance

* Hugging Face dataset `bcui2/NCU_report` (dataset), revision `21037737b7e371d38f3d029367dc3967d5b31a23`, folder `NVIDIA_B200`: 330 reports, 330 sha256-identical to the local files used.
* NCU versions in the reports: {'2026.1.1.0 (build 37634170) (public-release)': 330}; report creation range: ['2026-Aug-13 16:08:06', '2026-Oct-04 20:39:58'].
* Benchmark campaign: {"csv_dir": "results/B200/csv", "timing": "Proton, CUDA graph", "final_measurement_warmup_repeat": "not recorded; config.yaml at every result commit has 20/100 (developer_guide.md, Multi-Architecture Status)", "l2_eviction": "PyTorch/Triton/cuTile columns: fixed 64 MB buffer for 41 operators; 2x L2 (~253 MB) for cross_entropy, flash_decode, moe_topk_gating, linear_self_attention; TileLang columns: depends on run date (developer_guide.md)", "benchmark_source": "not recorded (CSVs predate tilebench/provenance.py)", "software": "PyTorch/Triton/cuTile columns from the paper campaign (cuda-tile 1.3.0 per developer_guide.md); TileLang 0.1.11", "tilelang_column": "direct TileLang runtime (PR #319), measured in a later campaign than the frozen PyTorch/Triton/cuTile columns", "raw_logs": "archive/raw-logs-2026-09-18"}
* Profiling: {"triton_cutile": "Nsight Compute 2026.1.1 (report session), --set full, application replay, --cache-control none; captured 2026-08 on dgx003 from tilebench_env; profiling source commit and cuda-tile version not recorded in the reports", "tilelang": "HF bcui2/NCU_report PR #2 (commit 21037737): driver 595.58.03, PyTorch 2.10.0+cu130, Triton 3.6.0, cuda-tile 1.3.0, TileLang 0.1.11, CUDA compiler 13.0.88, Nsight Compute 2026.1.1; 97 full, 5 targeted, 8 reduced (kernel replay)", "metadata_archive": "archive/raw-logs-2026-09-18 outputs/profiling/B200/{ncu_catalogue,kernel_counts}.json"}
* Operator configs and CSVs read from the checkout at `04d7e45526add29f567930ccbedaa6ac5b7f96ea`; archive refs {'origin/archive/raw-logs-2026-09-18': '9455c0bd76a0b09febee3080ce95f438f600896f', 'origin/archive/tilebenchpp-2026-10': '4c7dc1f08b91e39dbc2b2cf4c2ec591dba0e06f9'}.

## Coverage

| DSL | collection | replay | profiles |
|---|---|---|---|
| cutile | full | application | 110 |
| tilelang | full | application | 97 |
| tilelang | reduced | kernel | 8 |
| tilelang | targeted | application | 5 |
| triton | full | application | 110 |

QA status: **pass** (22/22 checks pass). Counts: {'benchmark_cases': 13200, 'profiles': 330, 'kernel_metric_rows': 336760, 'instruction_mix_rows': 62458, 'pc_hotspot_rows': 54832, 'execution_path_rows': 390, 'diagnosis_rows': 20}.

## Files

| file | size |
|---|---|
| `_build_log.json` | 0.3 kB |
| `benchmark_cases.csv` | 3.65 MB |
| `diagnosis_evidence.csv` | 26.7 kB |
| `environment.json` | 2.7 kB |
| `execution_paths.csv` | 0.16 MB |
| `instruction_mix.csv` | 9.42 MB |
| `kernel_metrics_long.csv.gz` | 4.60 MB |
| `pc_hotspots.csv.gz` | 0.53 MB |
| `profile_index.csv` | 0.47 MB |
| `qa_summary.json` | 5.7 kB |

## Metrics not collected

CORE metrics reported as `status = not_collected` (absent from the report; never inferred), count of shape-representative launches:

* `smsp__thread_inst_executed.sum`: 430
* `sm__cycles_elapsed.avg`: 33
* `smsp__cycles_active.avg`: 33
* `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`: 33
* `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: 33
* `lts__t_sectors.sum`: 33
* `sass__inst_executed_shared_loads`: 33
* `sass__inst_executed_shared_stores`: 33
* `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum`: 33
* `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`: 33
* `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`: 33
* `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`: 33
* `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`: 33
* `smsp__issue_active.avg.pct_of_peak_sustained_active`: 10
* `sass__inst_executed_global_loads`: 10
* `sass__inst_executed_global_stores`: 10
* `sass__inst_executed_local_loads`: 10
* `sass__inst_executed_local_stores`: 10
* `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`: 10
* `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: 10
* `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`: 10
* `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`: 10

## Comparison limitations

* Formal latency: PyTorch/Triton/cuTile columns come from the frozen paper campaign (config 20/100; fixed 64 MB L2 eviction for 41 operators, below the 126.5 MB L2). The TileLang column is a direct runtime measured in a later campaign (PR #319), so cross-DSL ratios involving TileLang also carry run-to-run and environment differences.
* Profiling source commit is not recorded for any B200 report (`code_match_status = not_recorded`). Triton/cuTile reports were captured 2026-08 with NCU 2026.1.1; the cuda-tile version at capture time is not recorded (paper campaign: cuda-tile 1.3.0).
* TileLang reports were collected on a different host/environment (HF PR #2: CUDA compiler 13.0.88, cuda-tile 1.3.0, TileLang 0.1.11). 8 are reduced kernel-replay collections and 5 are targeted metric lists: they have no per-opcode or PC-sampling data, and kernel-replay cache metrics are not methodologically identical to application replay.
* Triton/cuTile profiles capture only the operator kernels; PyTorch helper launches inside `run()` (fill/copy) are not in the report (see `profile_index.notes`, 9 profiles), although the formal timing covers the whole `run()`.
* TileLang source correlation points to the generated `tvm_kernels.cu`, which is not embedded in the reports; source-line columns therefore identify generated-code lines only.
* Triton `matmul_fp32_fp16_fp8`, `matmul_int8`, `batched_matmul`, `streamk_matmul` pre-transpose B on the host and cache it (`_bt_cache`): the transpose is outside the timed region and the NCU range.

## Reproduce

```bash
# 1. inventory + sha256 (no download for B200; GH200 reports downloaded from HF into an external cache)
python scripts/paper_figures/nvidia_inventory.py --device B200 --cache $CACHE --revision 21037737b7e371d38f3d029367dc3967d5b31a23 --report-root <report dir> [...]
# 2. offline import of every report (ncu --import / ncu_report only)
/usr/bin/python3 scripts/paper_figures/ncu_extract.py --ncu /opt/nvidia/nsight-compute/2026.1.1/ncu --out-dir $CACHE/extract/B200 <reports>
# 3. tables, evidence, QA, READMEs
PYTHONPATH=.:scripts/paper_figures CUDA_VISIBLE_DEVICES= python scripts/paper_figures/build_nvidia_tables.py --device B200 --repo . --extract-dir $CACHE/extract/B200 --inventory $CACHE/inventory_B200.json --out artifacts/paper_figures/nvidia/B200
python scripts/paper_figures/build_nvidia_evidence.py --repo . --root artifacts/paper_figures/nvidia --inventory-dir $CACHE
python scripts/paper_figures/qa_nvidia.py --repo . --root artifacts/paper_figures/nvidia --cache $CACHE
python scripts/paper_figures/write_nvidia_readmes.py --root artifacts/paper_figures/nvidia --cache $CACHE
```
