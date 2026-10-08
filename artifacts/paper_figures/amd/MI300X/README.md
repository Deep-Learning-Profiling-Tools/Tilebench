# MI300X paper-figure data package (TileArena, Phase 1A)

Structured MI300X data for RQ1 (cross-device Triton performance) and RQ2 (why Triton behaves differently on
MI300X). On MI300X only **Triton** and the **PyTorch** reference exist. There is no cuTile or TileLang data,
and none is invented.

Every file here was extracted from **existing** artifacts. Nothing was re-profiled or re-benchmarked, and no
kernel was run.

## Regenerate and validate

```bash
# optional, only if the hand-written diagnosis is edited: re-freeze it
python3 scripts/paper_figures/vendor_mi300x_diagnosis.py \
    --analysis-dir /root/Tilebench/outputs/profiling/MI300X/analysis_2026-10-05
python3 scripts/paper_figures/extract_mi300x.py \
    --outputs-root /root/Tilebench/outputs --logs-root /root/Tilebench/results/MI300X/logs
python3 scripts/paper_figures/validate_mi300x.py --outputs-root /root/Tilebench/outputs   # writes qa_summary.json
python3 -m unittest tests.test_paper_figures_mi300x                                      # or pytest
```

- **Requirements:** Python ≥ 3.10 and PyYAML.
- **Determinism:** two consecutive runs produce byte-identical files. The `.gz` files use `mtime=0` and no
  embedded file name.
- **Without the raw reports:** the git-ignored profiling reports live outside Git (local `/root/Tilebench/outputs`,
  and the HF dataset `bcui2/NCU_report`, folder `AMD_MI300X`, revision `645591bb`). When they are absent, the
  validator skips only the checks that re-read them.

## Sources

| source | role |
|---|---|
| `results/MI300X/csv/<op>_{autotune,default}.csv` (tracked) | **only** source of formal latency |
| `results/MI300X/logs/{time_measurement_logs,provenance,metadata}` (git-ignored, archived in `8e991537`) | `triton_ok` verification flag, measurement source SHA, device/software metadata; also used to cross-check the CSV rounding |
| `outputs/rocprof_compute/MI300X/<op>/triton_<dt>/` (git-ignored; HF `AMD_MI300X/`) | rocprof-compute 3.7.0 counter reports + PC sampling |
| `outputs/profiling/MI300X/rocprof_compute_sweep/` | report manifest (tree hashes), winner-replay audits, PROVENANCE.md |
| `outputs/profiling/MI300X/analysis_2026-10-05/` | Triton IR/AMDGCN dumps, replay audit, PyTorch kernel table, 9 ATen ATT decodes, diagnostic experiments |
| `scripts/paper_figures/data/mi300x_diagnosis_2026-10-05.json` (tracked) | the 2026-10-05 per-operator diagnosis, copied verbatim with source SHA256 |

## Formal timing protocol

The CSVs were measured from sources `005ab63b` (default) and `4d08985a` (autotune). The settings were:

- **warmup 20, repeat 100.** These come from `config.yaml` at those commits. Provenance shows no
  `--warmup`/`--repeat` override.
- **512 MiB write flush** before each iteration, outside the timed scope.
- **Eager execution.** The HIP-graph request falls back to eager on ROCm.
- **Latency** is the Proton kernel-time sum of the operator.

The `config.yaml` in this branch now says warmup 1 / repeat 3. That is **not** what the MI300X CSVs used. See
`environment.json` → `timing`.

## Files

**Extra columns.** The required columns always come first, in the order given in the specification. Extra
columns after them carry provenance only and are never part of the case identity.

### `benchmark_cases.csv`

- **Rows:** one row per operator × dtype × input case × mode (`autotune`, `default`), with `dsl=triton`; 4,360 rows
  (2,180 per mode).
- **Latency:** `torch_ms` and `dsl_ms` are the CSV strings verbatim, at 4 decimals as written by the benchmark.
  Nothing is rounded or re-parsed.
- **Case identity:** `case_id = sha256("{operator}|{dtype}|{params_json}")`. `params_json` is the CSV `params`
  cell, parsed with a top-level-comma splitter and serialised with sorted keys and `(",", ":")` separators. The
  same rule should be used for every device; see `scripts/paper_figures/case_identity.py`.
- **`validity`:** `verified_ok` when the timing log records `triton_ok=true` for the case. All 4,360 rows are
  `verified_ok`.
- **Provenance columns:** `source_git_sha` is the last commit that touched the CSV in this branch. The extra
  columns are `source_csv_sha256`, `measurement_source_git_sha`, `csv_row_number` and `csv_speedup_triton`.
- **Not done here:** no cross-device matching or aggregation.

### `profile_index.csv`

445 reports in five kinds:

| kind | count | what it is |
|---|---|---|
| `…rocprof_compute` | 109 | full counter collection, 13 passes |
| `…pc_sampling` | 109 | stochastic PC sampling, interval 65536 |
| `…static_isa` | 109 | recompiled IR/AMDGCN of the formal winner kernels |
| `…kernel_trace` (PyTorch) | 109 | rocprofv3 kernel trace of one reference call |
| `…att` (PyTorch) | 9 | ATT decodes of ATen kernels |

- **Profiled input:** every report records it in `profiled_params_full_json`, together with its case. The
  input comes from `capture.json`; it is not assumed to be the largest case.
- **Case matching:** all 109 pairs are `matched`. The swept params equal one autotune CSV row, and the remaining
  params equal `case_defaults` in `config.yaml` at the profiling commit. In every pair that row happens to be the
  largest CSV row.
- **`code_match_status`** combines five checks:
  - the manifest tree hash was reproduced;
  - the captured kernel names equal the expected names;
  - the replay-audit hsaco is identical between the formal path and the profiling replay (810/810 launches);
  - the post-fix audit passed;
  - every PC-sampled instruction text appears in the recompiled AMDGCN: 17,120/17,120. Branch targets are
    compared by opcode only.
- **`report_sha256`:**
  - directories: the manifest's tree-hash algorithm;
  - single files: plain SHA256;
  - static-ISA reports: a hash over the stage directories.

### `kernel_metrics_long.csv.gz`

252,200 rows in long format, of six kinds:

| `counter_kind` | content |
|---|---|
| `rocprof_compute_derived_metric` | Each row of the existing `analysis/workload_csv/kernel_metric.csv`, verbatim. `raw_metric_name` = `"<metric_id> \| <metric_name> \| <value_name>"`; units are from the report. |
| `raw_pmc_counter` | Raw rocprofiler counters per dispatch. `launch_id` = `kernel#ordinal`. |
| `dispatch_record` | Grid, workgroup, LDS, scratch and VGPR/AGPR/SGPR per dispatch. `Grid_Size` is in work-items. |
| `static_compiler_metadata` / `launch_metadata` | AMDGCN header values and replay-audit launch records. |
| `pc_sampling_count` | Samples, issued, stalled and per-reason counts per kernel. |
| `profiler_duration_diagnostic_only` | Profiler and trace kernel times. **Never formal latency.** |

- **Rows left out (counted in `extraction_log.json`):**
  - the per-channel L2 table (705,024 rows), because aggregate L2 metrics are kept;
  - per-instance raw counters `NAME[n]` (1,451,520 rows), because their `_sum` totals are kept;
  - Min/Max/Quartiles of single-dispatch kernels when they equal `Avg` (54,752 rows).
- **Status values:** `collected`, `derived`, `unavailable_empty_in_report` (2,058 empty cells in the source,
  left empty) and `unavailable_no_samples`. No absent value is written as 0.
- **`normalized_group`:** an organisational label only. It does not mean an AMD metric equals any NVIDIA
  metric.

### `instruction_mix.csv`

12,233 rows. **Static counts are never presented as dynamic counts.** The `count_kind` values are:

| `count_kind` | meaning |
|---|---|
| `static_isa_opcode_count` | Opcode occurrences in the AMDGCN text, for Triton stages and for the ATen ATT code objects. Cache/scope modifiers appear in the name, e.g. `buffer_load_dwordx4 [sc0 nt]`. |
| `dynamic_hw_counter_total_summed_over_dispatches` | `SQ_INSTS_*` raw counters summed over a kernel's dispatches. Per-dispatch values are in `kernel_metrics_long`. `SQ_WAIT_INST_ANY` counts cycles, not instructions. |
| `dynamic_att_hitcount_traced_waves` | ATT hit counts. Only the waves traced on one CU are covered, not the whole kernel. |

### `pc_hotspots.csv.gz`

- **PC-sampling rows:** every operator-kernel row of `pc_sampling_instructions.csv`. `sample_count` is filled;
  `executed_instruction_count` is empty because PC sampling has no execution counts. `stall_reason` is the JSON
  of the reported reasons. Kernels with fewer than 100 samples are flagged in `kernel_metrics_long`.
- **ATT rows:** ATen instructions with hit count > 0. `executed_instruction_count` holds the hit count of the
  traced waves.

### `execution_paths.csv`

One row per Triton code object (`stage`) and per PyTorch trace kernel. 523 rows.

- **`access_path` combines four facts:**
  - **Source form:** TensorDescriptor or pointer. It is found by reading the `@triton.jit` body at the
    profiling commit.
  - **Dumped TTIR:** it contains no `tt.descriptor_*` ops. The descriptors are lowered to pointer/buffer loads
    on gfx942.
  - **Static loads and stores:** counted by opcode and width, as scalar (≤ 32 b) or vector (≥ 64 b).
  - **Non-temporal loads:** the number carrying the `nt` modifier.
- **Other columns:**
  - **`matrix_path`:** MFMA opcodes plus the measured MFMA utilisation (2.1.10).
  - **`buffer_location`:** LDS and scratch use.
  - **`layout_operations`:** TTGIR `convert_layout` / `local_alloc` counts, `ds_bpermute`, DPP, `threadsPerWarp`.
  - **`atomic_path`:** static atomics plus 16.3.3, 17.3.4, 17.6.11, 17.2.11 and 3.1.46.
  - **`resource_summary`:** grid in workgroups vs 304 CUs, registers, LDS, scratch, occupancy.
- **PyTorch rows:** only the library family and the dispatch record; no ISA. Unknowns are written as
  `unknown` / `not_observed`.

### `diagnosis_evidence.csv`

56 rows: each operator's primary mechanism, plus a secondary mechanism wherever the original text names one,
e.g. "(M4)".

- **Taxonomy and text:** M1–M7 and C as in the 2026-10-05 analysis. Text is verbatim, `review_status` = draft.
- **Extra columns:** the B200 Table 5 reference, the relationship class, the paper priority, and an automatic
  `trace_check_status` / `trace_check_detail` that re-derives the label from the extracted raw evidence.
- **Trace-check results:**
  - all primary M1, M2, M5, M6 and M7 labels trace;
  - M4 is traced by metric (INT32 VALU);
  - three records do not fully trace (see "Mismatches" below).

### `diagnostic_experiments.csv`

120 rows. These are controlled **diagnostic** measurements, kept apart from the formal data. Their protocol is
`report_benchmark(warmup=2, repeat=10)` with the 512 MiB write flush; the flush experiment varies the flush.
`changed_factor` and `other_configuration_changes` list every dimension that differs.

| experiment | rows |
|---|---|
| `gemm_desc_vs_ptr_{fp16,fp32}` | descriptor vs pointer GEMM |
| `stream_vadd_config_sweep_{fp16,fp32}` | vector_add configuration sweep |
| `cache_modifier_ablation_fp32` | load/store cache modifiers |
| `flush_protocol_variants_fp32` | alternative flush protocols |

Caveats:
- **The FP16 GEMM pointer variant also changes `num_stages` from 3 to 2.** The pointer kernel at 3 stages
  needs 128 KiB of LDS, over the 64 KiB limit. That comparison is not one-factor.
- **`load(default)_store.cg` emits the same ISA as the default variant.** The modifier has no effect on gfx942.

### Other files

- **`environment.json`:** device, software, profiler, timing protocol, provenance commits, the known exclusion,
  and the SHA256 of every input file and extraction script.
- **`qa_summary.json`:** the validation result. It holds counts, metric coverage, missing entries, parse issues,
  unresolved mappings and the cross-check against the prior analysis.
- **`extraction_log.json`:** extraction statistics and issues.

## Known gaps and caveats

- **`matmul_fp32_fp16_fp8` / `fp8_e4m3fn` is excluded.** It is UNSUPPORTED_DTYPE on gfx942: there is no CSV
  row, no autotune winner and no profile. This gives 45 operators and 109 operator/dtype pairs.
- **PyTorch side:**
  - the raw rocprofv3 traces were not retained, only the derived per-kernel table (names truncated to 300
    characters);
  - there are no PyTorch counter profiles;
  - only 9 ATen kernels have ATT decodes, so `nt` load policy on other ATen kernels is inferred from the kernel
    template.
- **Profiling coverage:**
  - no Triton ATT traces exist;
  - every pair was profiled at a single input case, the largest CSV row.
- **Triton IR is recompiled, not saved from the formal run.** The replay audit and the PC-sample text check tie
  it to the profiled code.
- **Metrics not to use as evidence:**
  - rocprof-compute "Spill/Stack" metrics (10.3.4–10.3.7, 15.2.5–15.3.2) count buffer instructions, not spills;
  - the 7.2.4 description string is a duplicate;
  - the 6.2.x workgroup-manager counters are zero for all 109 profiles.

  All three are annotated in `notes`.
- **Two captures record a `dtype` key in params:** `fused_activation` and `quantize_global` have it in
  `capture.json`, but not in the catalogue. This does not affect case matching.

## Mismatches between the 2026-10-05 analysis and the source reports

- **Values reproduced exactly.** The prior `counters.json` (12,624 values), `isa.json` (994 values),
  `pcs.json` (109 totals) and the max-case latencies in `perf.json` (218) all match this package. `perf.json`
  used the full-precision timing logs; they round to the CSV values.
- **Three diagnosis records do not fully trace:**
  - **`batch_normalization`, secondary M6.** The text cites "313 WGs" as grid under-subscription, but 313 ≥ 304
    CUs, so it is not under-subscription by count.
  - **`batch_normalization` and `softmax`, primary M3.** No vendor-library kernel appears in the trace; the
    PyTorch path is ATen.
  - **`destindex`, secondary M2.** The ATen kernel is `index_elementwise`, not a vectorized or reduce kernel.

  The text is kept unchanged and is flagged in `trace_check_*`.
- **Timing protocol.** The 2026-10-05 analysis described the formal protocol as warmup 1 / repeat 3. The
  measurement-commit configuration shows warmup 20 / repeat 100.
