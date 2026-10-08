# TileArena paper-figure data package — schema v1

Shared schema for the per-device profiling packages under `artifacts/paper_figures/<vendor>/<device>/`
(NVIDIA: `nvidia/B200`, `nvidia/GH200`; AMD MI300X is expected to use the same files and columns).
All tables are UTF-8 CSV with a header row and `\n` line endings; `*.csv.gz` files are gzip-compressed CSV.
Columns listed as *required* appear first and in the stated order; further *optional* columns may follow.

## 1. Identifiers

### case_id
A cross-device identity of one benchmark input case.

```python
params = {k: v for k, v in case.items() if k not in ("dtype", "block_size")}   # engine params of the case
blob = json.dumps({"dtype": dtype, "operator": operator, "params": params},
                  sort_keys=True, separators=(",", ":"), ensure_ascii=True)
case_id = hashlib.sha256(blob.encode()).hexdigest()
params_json = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
```

* `params` is the full parameter dict of the case as produced by `tilebench.data.tensors.expand_cases`
  (`case_defaults` merged with the grid point), with the engine default dtype `fp32` when the grid has no dtype.
  Values keep their YAML/JSON types (int, float, bool, str).
* A formal-CSV row is mapped to its case with the rule of `scripts/run_bench.py::_label_matches_result`
  (the CSV `params` column lists only the varying keys). A row that maps to anything other than exactly one case
  gets `case_id = UNMATCHED` (QA check 5 fails).
* `dtype` uses the canonical TileBench labels (`fp32`, `fp16`, `bf16`, `fp8_e4m3fn`, `int8`, `int32`, ...);
  the alias `float32` is normalized to `fp32`.
* Reference implementation: `scripts/paper_figures/nvidia_common.py::case_id`.
* Never use CSV row numbers or profiler launch indices as case identifiers.

### profile_id
`<device>/<dsl>/<operator>/<dtype>` (one NCU/rocprof report per id).

### launch_id
Zero-based index of a profiled launch in report order (range, then action). In `instruction_mix.csv` a group of
launches of one kernel is written as `sum:<first>-<last>` (see `scope`); static rows use `static` or the launch id
whose per-PC table provided the binary.

## 2. Files and columns

### benchmark_cases.csv — formal latency (one row per device × DSL × operator × dtype × case × mode)
Required: `device, architecture, dsl, operator, category, dtype, case_id, params_json, mode, torch_ms, dsl_ms,
validity, source_csv, source_git_sha, notes`
* `dsl_ms`, `torch_ms`: the raw strings of `results/<device>/csv/<op>_<mode>.csv` columns `<dsl>_ms` / `torch_ms`
  (original precision; never recomputed, never NCU durations).
* `mode`: `autotune` (authoritative for the paper) or `default`.
* `validity`: `valid` (finite > 0), `missing` (empty cell), or `invalid:<raw>`.
* `source_git_sha`: last commit that changed `source_csv` in the checkout used for the build.
* `category`: paper Table "Operator benchmark suite" (Point-wise, Reduction/Normalization, Matrix Multiplication/Attention,
  Stencil/Convolution, Data Layout).

### profile_index.csv — one row per profiler report
Required: `profile_id, device, dsl, operator, dtype, case_id, params_json, collection_level, replay_mode, report_path,
report_sha256, report_origin, report_revision, benchmark_case_match_status, code_match_status, launch_count, notes`
Optional (NVIDIA): `extract_status, kernel_names, expected_launch_count, expected_operator_launch_count,
winner_config_json, ncu_version, report_created, profiler_cuda_version, profiler_command_line, benchmark_dsl_ms,
benchmark_torch_ms`
* `collection_level`: `full` (`--set full`), `targeted` (`--metrics` list, application replay), `reduced`
  (`--set none --metrics`, kernel replay), `unknown`. Derived from the profiler command line stored in the report.
* `replay_mode`: profiler replay mode from the report (`application`, `kernel`, ...).
* `report_path`: path inside the Hugging Face dataset; `report_revision`: `<repo_id>@<commit sha>`.
* `report_origin`: `hf_download` or `local_copy_sha256_identical_to_hf`.
* `benchmark_case_match_status`: `matched` | `matched_<validity>` | `case_not_in_benchmark_csv` | `no_profile_params`.
* `code_match_status`: `documented:<kind>` (profiling vs benchmark source relation recorded in a provenance file) or
  `not_recorded`.

### kernel_metrics_long.csv.gz — profiler counters in long format
Required: `profile_id, launch_id, kernel_name, stage, raw_metric_name, normalized_group, value, unit, scope,
counter_kind, extraction_method, status, raw_source, notes`
* `raw_metric_name`, `unit`: exactly as stored in the report.
* `value`: the reported value (Python `repr` for floats); empty when `status = not_collected`.
* `normalized_group`: one of `execution, occupancy, resources, instruction_mix, global_memory, cache, local_memory,
  shared_memory, matrix_pipeline, synchronization, stalls, atomics` (first matching rule of
  `ncu_extract.py::GROUP_RULES`; the raw name is authoritative).
* `scope`: `launch` (value of one kernel launch).
* `counter_kind`: `<metric type>|<subtype>|<rollup>` from the report (e.g. `throughput|pct_of_peak_sustained_elapsed|avg`).
  **Denominator rule:** a percentage keeps its denominator in the raw name (`..._active` vs `..._elapsed`); never compare
  the two as the same measurement and never rescale across architectures.
* `extraction_method`: `ncu_report:core` (CORE set, every launch) or `ncu_report:extended(shape_representative_launch)`
  (all matching scalars of the first launch of each distinct (kernel, grid, block, registers, shared memory) shape).
* `status`: `collected` or `not_collected` (a CORE metric absent from the report; no value is inferred).
* Export filter for the extended set: `.per_second`, `.peak_sustained`, `.per_cycle_elapsed` variants and `.max/.min`
  rollups are dropped; a `.sum.pct_of_peak_*` twin of an `.avg.pct_of_peak_*` is dropped; zero values are kept only for
  diagnostic names (TMA, tensor, TMEM, LDGSTS, local, atomics, bank conflicts, sectors/requests, shared, stalls).
  The external extract keeps every collected scalar.

### instruction_mix.csv
Required: `profile_id, launch_id, instruction_family, instruction_name, count, count_kind, scope, evidence_path, notes`
* `count_kind`:
  * `dynamic_warp_inst_executed` — `sass__inst_executed_per_opcode`
  * `dynamic_warp_inst_executed_with_modifier` — `sass__inst_executed_per_opcode_with_modifier_all`
  * `dynamic_thread_inst_executed_true` — `sass__thread_inst_executed_true_per_opcode`
  * `dynamic_warp_inst_executed_family_total` — family sums of `dynamic_warp_inst_executed` (`instruction_name = __family_total__`)
  * `static_sass_instruction_count` — instruction sites in the kernel binary (**not executed instructions**)
* `scope`: `launch`, `sum_over_launches(n=<k>)` (profiles with more than 16 launches, per kernel name), or
  `kernel_binary:<stage>` (static rows). Dynamic and static rows are never mixed in one row.
* Reports without per-opcode data (reduced/targeted) have only static rows.
* `instruction_family`: `integer_address, predicate_select, floating_point, special_function, conversion_packing,
  data_movement, warp_collective, constant_load, global_load_store, async_copy_ldgsts, atomic, shared_memory,
  local_memory, tma, legacy_mma, wgmma, tcgen05, synchronization, control_flow, uniform_other, other_unknown`
  (opcode lists in `nvidia_common.py::FAMILY_OPCODES`; vendor-specific families are allowed, unknown opcodes map to
  `other_unknown`).

### pc_hotspots.csv.gz
Required: `profile_id, launch_id, pc, instruction_family, instruction_text, sample_count, executed_instruction_count,
stall_reason, source_line, evidence_path, notes`
* Selection: for the first launch of each launch shape, the 40 PCs with most PC-sampling samples plus the 10 PCs with
  most executed warp instructions (union). Rebuildable from the external extracts (all PCs).
* One row `stall_reason = __all__` (total samples of the PC) plus one row per non-zero stall reason.
* `sample_count` is PC-sampling samples; `executed_instruction_count` is executed warp instructions at the PC.
  They are different measurements.
* `source_line`: `<file>:<line>` from the report's source correlation, empty when absent.

### execution_paths.csv — one row per profile × stage (kernel)
Required: `profile_id, stage, access_path, matrix_path, buffer_location, layout_operations, atomic_path,
resource_summary, evidence_path, evidence_confidence, notes`
* Built from dynamic opcode-with-modifier counts (`evidence_confidence = high`) or, when absent, static SASS
  (`medium`; such path strings start with `static_sites:` and their numbers are instruction sites).
* `access_path`: TMA, cp.async (LDGSTS), LDG/STG width distribution (bits: % of executed instructions).
* `matrix_path`: `tcgen05[...]`, `wgmma[...]`, `legacy_mma[...]` with opcode+modifier and count, or `none`.
* `buffer_location`: `TMEM`, `shared`, `local(LDL/STL)`, `registers`.
* `resource_summary`: grid, block, registers, shared memory per block, theoretical and achieved occupancy of the
  longest launch of the stage.

### diagnosis_evidence.csv
Required: `device, dsl, operator, dtype, comparison_scope, mechanism_id, observation, mechanism_hypothesis,
supporting_metric_names, supporting_profile_ids, alternative_explanation, confounders, evidence_quality,
review_status, source_paths`
* `observation` is generated from the tables (values per DSL or per device); `mechanism_hypothesis` is the
  interpretation and is kept separate.
* `comparison_scope`: `cross_dsl:<device>:<dslA>_vs_<dslB>[...]` or `cross_device:<dsl>:<devA>_vs_<devB>`.
* `evidence_quality = high` requires at least three supporting metrics including non-stall evidence (QA-checked).

### cross_device_pairs.csv (nvidia/)
One row per operator × dtype × DSL profiled on both NVIDIA devices: matching case ids, winners, NCU versions,
collection levels, kernel names, matrix/access paths, formal latencies and their ratio, dynamic family totals,
DRAM bytes, registers, shared memory, achieved occupancy, missing metrics, and all known confounders.

### environment.json, qa_summary.json, README.md
Per device: provenance (dataset revision, benchmark campaign, profiler sessions, software), automated QA results,
and human-readable limitations.

## 3. Validation
`python scripts/paper_figures/qa_nvidia.py --repo . --root artifacts/paper_figures/nvidia --cache <cache>` checks
the required columns and the 13 data-integrity rules (operators, coverage, sha256, device/DSL identity, case matching,
raw names and units, static/dynamic separation, no fabricated opcode counts, launch identity, source correlation,
latency only from CSV, no new GPU profiling, no unrelated file changes).
