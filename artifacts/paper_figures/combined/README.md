# Combined cross-device normalization layer (B200, GH200, MI300X)

Plotting-ready tables built **only** from the committed device packages
(`../nvidia/B200`, `../nvidia/GH200`, `../amd/MI300X`), the formal CSVs under `results/<device>/csv/` and git history.
The device packages are not modified (QA check 15 compares them with their extraction commits
`3992d1ce` (NVIDIA) and `25bf1581` (AMD)).

```bash
PYTHONPATH=.:scripts/paper_figures CUDA_VISIBLE_DEVICES= python scripts/paper_figures/build_combined.py --repo .
PYTHONPATH=.:scripts/paper_figures CUDA_VISIBLE_DEVICES= python scripts/paper_figures/validate_combined.py --repo . \
    --nvidia-cache /projects/kzhou6/bcui2/research/tilebench/ncu_report_cache      # writes qa_combined.json (gate)
```

## Canonical case identity (`case_id_v2`)

```python
canonical = {"operator": op, "dtype": dtype, "params": full_input_parameter_dict}
case_id_v2 = sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
```

`full_input_parameter_dict` is the engine parameter dict of the case (`tilebench.data.tensors.expand_cases`, `dtype` and
`block_size` removed, engine default dtype `fp32`). A formal-CSV row (whose `params` column lists only the varying keys) is
mapped with the engine's own matching rule (`scripts/run_bench.py::_label_matches_result`) against the operator config of
the **measurement source**:

| device | config used | evidence |
|---|---|---|
| GH200 | `c882fe50` (89 CSVs), `3c5eccbf` (`batched_matmul_autotune.csv`) | developer_guide, PROVENANCE.md |
| MI300X | `4d08985a` (autotune), `005ab63b` (default) | MI300X `environment.json` |
| B200 | not recorded; the config in the tree of the commit that introduced each CSV column's current values, plus every commit carrying the TileLang column | git history of each CSV |

Every row resolves to exactly one case under every config listed, and all resolutions agree with each other
(and with the current `config.yaml`). The original exporter ids are kept as `source_case_id`: for NVIDIA they equal
`case_id_v2`; the MI300X ids (`sha256('op|dtype|swept_params')`) differ and are mapped in `case_id_crosswalk.csv`.

## Files

| file | content |
|---|---|
| `benchmark_cases_normalized.csv.gz` | one row per device × DSL × operator × dtype × case × mode; raw CSV latency strings, `case_id_v2`, full params, canonical category, analytical `validity` (`valid` / `invalid` / `missing`) next to the original `validity_source` (`valid`, `verified_ok`), CSV path + sha256 + commit, config commits used |
| `case_id_crosswalk.csv` | source id ↔ `case_id_v2` per device × mode × source case |
| `profile_index_normalized.csv` | all 1,105 profiles (NCU 660; MI300X rocprof-compute, PC sampling, static ISA, PyTorch kernel trace, ATT) with `case_id_v2`, report kind, profiler, collection level, replay mode, winner config, code-match status/confidence, and the meaning of their instruction counts |
| `category_mapping.csv` | 45 operators → five canonical categories (paper Table "Operator benchmark suite"), with the MI300X short labels |
| `coverage_summary.csv` | device × DSL × operator × dtype status (`measured`, `unsupported_dtype`, `dsl_not_available_on_device`) |
| `metric_semantic_groups.csv` | organisational semantic labels for the device-native metric groups (labels only, never a shared numeric scale) |
| `comparison_manifest.json` | identity rule, DSL support, explicit exclusions, case intersections, aggregation protocol, sanity aggregates, source-package hashes, device limitations |
| `qa_combined.json` | the validation gate |

## Coverage and intersections (autotune, valid)

* B200 and GH200: Triton, cuTile and TileLang each 2,200 cases (110 operator/dtype pairs, 45 operators).
* MI300X: Triton only, 2,180 cases (109 pairs). `matmul_fp32_fp16_fp8` / `fp8_e4m3fn` is **unsupported** on gfx942
  and stays missing (`coverage_summary.csv`, status `unsupported_dtype`).
* Common cases: Triton B200 ∩ GH200 ∩ MI300X = 2,180; Triton B200 ∩ GH200 = 2,200; cuTile B200 ∩ GH200 = 2,200;
  TileLang B200 ∩ GH200 = 2,200.

## Aggregation protocol (used by every figure)

* `S[o,b,d] = GM_cases(torch_ms / dsl_ms)`; `S[b,d] = GM_operators(S[o,b,d])` (autotune, valid rows).
* `R[X/Triton][o,d] = GM_cases(X_ms / triton_ms)` over cases valid for both (> 1: X slower).
* `delta = log2(S_dev2 / S_dev1)` over matched `case_id_v2`; relative to each device's own PyTorch baseline, **not** an
  absolute hardware speedup.
* Winner: lowest GM latency over cases valid for all DSLs of the device; near parity = within 5 % of the runner-up.

Recomputed sanity values: S = 2.02 / 1.58 / 1.71 (B200 Triton / cuTile / TileLang), 1.85 / 1.53 / 1.57 (GH200),
1.27 (MI300X Triton). Winners: B200 Triton 22, TileLang 18, cuTile 5; GH200 Triton 26, TileLang 17, cuTile 2.

## Measurement semantics that are NOT harmonized

NVIDIA dynamic SASS opcode counts, MI300X static AMDGCN opcode counts, MI300X PC samples and MI300X ATT hit counts are
different measurements. `metric_semantic_groups.csv` assigns labels (indexing, memory_access, matrix_execution, …) for
organisation only; cross-vendor numeric comparisons of counters are not supported by this layer.

## Device limitations carried into every figure

See `comparison_manifest.json` → `device_limitations` (B200 historical cuda-tile and separate TileLang campaign;
GH200 CUDA/cuda-tile versions, 1/3 protocol, Hopper TileLang bodies, cached Triton B transposes; MI300X 20/100 eager
timing, FP8 E4M3FN unsupported, partially traced diagnosis records).
