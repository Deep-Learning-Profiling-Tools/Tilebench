# TileBench Developer Guide

This document contains implementation and maintenance details for extending TileBench. The root [README](../README.md) focuses on using the benchmark.

## Contents

- [CLI Reference](#cli-reference)
- [Operator Layout](#operator-layout)
- [Comparison Policy](#comparison-policy)
- [Configuration](#configuration)
- [Correctness](#correctness)
- [Autotuning](#autotuning)
- [Dtype Handling](#dtype-handling)
- [Metrics and Device Peaks](#metrics-and-device-peaks)
- [Profiling](#profiling)
- [LLM Code Generation](#llm-code-generation)
- [Generated Outputs](#generated-outputs)
- [Pre-Merge Checklist](#pre-merge-checklist)

## CLI Reference

### `scripts/run_bench.py`

| Argument | Default | Description |
|---|---|---|
| `--operator` | `vector_add` | Operator to benchmark |
| `--output` | `results/logs/time_measurement_logs/<op>_results.json` | Local timing JSON |
| `--autotune-log` | `results/logs/autotune_logs/<op>_autotune.json` | Local autotune metadata |
| `--warmup` | from config | Warmup iterations |
| `--repeat` | from config | Timed iterations |
| `--use-cuda-graph` | from config | Enable CUDA graph replay |
| `--flush-l2` | from config | Flush L2 before each iteration |
| `--autotune` | off | Enable autotuned execution |
| `--tile-language` | all | Comma-separated backends: `triton,cutile,tilelang,nki`; PyTorch always runs as the reference |
| `--case-indices` | all | Run a subset of cases, for example `0,1,3` |
| `--keep-proton-files` | false | Keep Proton `.hatchet` files |
| `--proton-output-dir` | system temp | Proton output directory |
| `--no-archive` | false | Disable local raw-log archiving |

A Triton/cuTile run writes the tracked summary CSV to:

```text
results/csv/<operator>_default.csv
results/csv/<operator>_autotune.csv
```

Local logs and generated figures are ignored by Git.

### `scripts/run_bench_all.py`

Runs the operator suite sequentially. Use `--help` for the current command-line options.

### `scripts/visualize.py`

| Argument | Default | Description |
|---|---|---|
| `--operator` | required | Operator name |
| `--input` | default timing JSON path | Input timing JSON |
| `--output-dir` | local figure directory | Figure output |
| `--metrics` | from `config.yaml` | Metrics to plot |
| `--gpu` | none | Device label such as `B200`; loads `tilebench/data/peak_performance/<GPU>.json` |

Supported derived views include latency, bandwidth, speedup, TFLOPS, percentage of peak, arithmetic intensity, and roofline plots.

## Operator Layout

Each operator lives under:

```text
tilebench/benchmarks/operators/<name>/
├── config.yaml
├── impl_torch.py
├── impl_triton.py
├── impl_cutile.py
├── impl_tilelang.py    # optional
└── impl_nki.py         # optional
```

Input generators are registered in:

```text
tilebench/data/tensors.py
```

The benchmark discovers operator resources through the `tilebench` package rather than the current working directory.

## Comparison Policy

TileBench is designed for controlled backend comparison.

- Keep input/output semantics identical across compared implementations.
- Keep custom backend algorithms and implementation structures comparable.
- PyTorch defines the semantic reference and practical software baseline; its internal decomposition does not need to match the custom kernels.
- Preserve the declared precision path and shape assumptions.
- Do not add hidden backend-specific workarounds to make unsupported cases appear successful.
- Record unsupported configurations and correctness failures explicitly.
- Keep default and autotuned execution paths separate.

## Configuration

A typical `config.yaml` contains benchmark controls, a case grid, and metric formulas:

```yaml
benchmark:
  warmup: 20
  repeat: 100
  use_cuda_graph: true
  flush_l2: true
  autotune: false

case_grid:
  n:
    expr: "[1024 * 1024 * i for i in range(1, 21)]"
  dtype: ["fp16", "bf16", "fp32", "int8"]

metrics:
  flops_expr: "n"
  bytes_expr: "n * dtype_size * 2"
  plots:
    - latency_ms
    - bandwidth_GBs
    - speedup
```

Use operator-specific grids. Include representative small, medium, large, and non-power-of-two cases when the operator semantics allow it.

## Correctness

Correctness is checked automatically by `tilebench/core/verifier.py` against the PyTorch reference.

Common defaults are:

| dtype | atol | rtol |
|---|---:|---:|
| float32 | 1e-5 | 1.3e-6 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1.6e-2 |
| integer types | 0 | 0 |

Operator-specific overrides may be declared when the numerical contract requires them. Do not relax tolerances merely to make a broken implementation pass.

## Autotuning

### Triton

Use Triton's native autotuning interface with an explicit candidate space. The selected configuration must be recoverable through `get_last_config()`.

Typical tunables include block sizes, `num_warps`, `num_stages`, and operator-specific scheduling parameters.

### cuTile

Use `tilebench.core.cutile_autotune.CutileAutotuner` around cuTile's tuning interface. Cache the selected configuration in-process so candidate search is not repeated inside the timing loop.

Typical tunables include tile dimensions, occupancy hints, and operator-specific launch choices.

### TileLang

TileLang is optional. Keep the same operator semantics and comparable implementation strategy. The current dependency pins are recorded in `requirements.txt`.

### NKI

NKI is optional and runs on AWS Trainium. NKI autotuning and profiling use the infrastructure under `tilebench/core/nki_*.py`. Winner replay must be deterministic so profiling measures the selected configuration rather than re-running the search.

## Dtype Handling

Support is operator-specific.

- Standard floating-point types should follow the operator's declared precision path.
- Integer inputs should be generated within ranges that avoid unintended overflow when exact comparison is required.
- FP8 support must be validated per operator and backend. Do not infer general arithmetic support from the presence of an FP8 dtype or an FP8 GEMM path.
- Do not silently cast the data path to another precision solely to make a case run.

## Metrics and Device Peaks

Analytical FLOP and byte formulas live in each operator's `config.yaml`.

Canonical device metadata is tracked under:

```text
tilebench/data/peak_performance/
├── B200.json
└── Trainium2.json
```

These files are framework inputs. Detailed measurements produced by `scripts/measure_peak.py` are local generated outputs under `outputs/peak_performance/` and are not tracked.

## Profiling

Profiling support lives under:

```text
tilebench/profiling/
```

Important canonical metadata includes:

- `ncu_catalogue.json`: profiled operator/dtype cases and selected configurations.
- `kernel_counts.json`: expected kernel-count metadata used to validate NCU captures.

Generated NCU reports are written under `outputs/` and are ignored by Git.

If `kernel_counts.json` is missing, the main profiling drivers fail with a clear error instead of silently assuming one kernel per launch. Missing metadata for an individual pair may fall back to one kernel with an explicit warning.

## LLM Code Generation

The iterative generation pipeline is under:

```text
tilebench/llm_codegen/
```

Task descriptions, framework conventions, prompt construction, evaluation, and feedback logic live there. Backend API guides are under `skills/`.

Each run writes its trajectory (prompts, responses, kernels, feedback, token usage) and the selected implementation to:

```text
tilebench/benchmarks/llm_generated/<operator>/<model>/<effort>/
```

This directory is the pipeline's default output location and is Git-ignored: the pipeline creates it on demand, and nothing under it is committed to `main`.

Keep LLM generation separate from the manually implemented benchmark path. Generated code must not delegate the operator computation to PyTorch, vendor libraries, or backend autotuners when the generation protocol forbids them.

## Generated Outputs

The repository tracks only per-case benchmark CSVs under:

```text
results/csv/
```

Other artifacts are generated locally and ignored, including:

```text
results/logs/
results/figures/
results/aggregate/
outputs/
tilebench/benchmarks/llm_generated/
```

Writers should create these directories when needed; a fresh clone must not depend on pre-existing generated directories.

### Archiving artifacts

`scripts/archive_artifacts.sh` backs artifacts up on the `archive/raw-logs-2026-09-18` branch without checking it out or touching the index:

```bash
scripts/archive_artifacts.sh --logs     # results/logs/
scripts/archive_artifacts.sh --llm      # tilebench/benchmarks/llm_generated/
scripts/archive_artifacts.sh --all      # both
scripts/archive_artifacts.sh --llm --push
```

The archive is cumulative: artifacts that the current machine does not hold are carried forward from the archive tip, never dropped. `run_bench.py` calls `scripts/archive_logs.sh` (equivalent to `--logs`) after each run, so raw logs are archived automatically; LLM trajectories are archived only on request, at milestones worth keeping. Nothing is pushed without `--push`.

### Publishing a downloadable artifact

Downloadable artifacts are declared in `artifacts/manifest.json` (URL, SHA256, archive name, and the directories the archive provides). To publish one:

```bash
python scripts/package_artifacts.py --artifact llm-aacl2026
```

writes a reproducible `outputs/artifacts/<archive>.tar.gz` and prints its SHA256. Upload the archive, then put the file's share link and that checksum in the manifest. Readers restore it with `python scripts/fetch_artifacts.py --artifact <name>`.

## Pre-Merge Checklist

- [ ] Operator semantics match the PyTorch reference.
- [ ] All configured dtypes and shapes pass correctness, or unsupported cases are explicitly recorded.
- [ ] Triton and cuTile use comparable algorithms and implementation structures.
- [ ] Default and autotuned paths both work.
- [ ] `get_last_config()` reports the configuration actually used.
- [ ] `config.yaml` includes the intended case grid and metric formulas.
- [ ] Boundary and tail cases are handled correctly.
- [ ] No hidden dtype workaround changes the declared precision path.
- [ ] Optional TileLang/NKI implementations follow the same semantic contract.
- [ ] `pytest` passes.
- [ ] `python -m compileall tilebench scripts tests` passes.
- [ ] No generated artifacts outside `results/csv/` are accidentally tracked.
