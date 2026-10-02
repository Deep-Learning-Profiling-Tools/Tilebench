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
- [Neuron Stack Diagnostics](#neuron-stack-diagnostics)
- [LLM Code Generation](#llm-code-generation)
- [Generated Outputs](#generated-outputs)
- [Pre-Merge Checklist](#pre-merge-checklist)

## CLI Reference

### `scripts/run_bench.py`

| Argument | Default | Description |
|---|---|---|
| `--gpu` | required, no default | Hardware label of the campaign, such as `B200`; names the result namespace `results/<gpu>/` |
| `--operator` | `vector_add` | Operator to benchmark |
| `--output` | `results/<gpu>/logs/time_measurement_logs/<op>_<mode>_<backends>.json` | Local timing JSON |
| `--autotune-log` | `results/<gpu>/logs/autotune_logs/<op>_<mode>_<backends>.json` | Local autotune metadata |
| `--warmup` | from config | Warmup iterations |
| `--repeat` | from config | Timed iterations |
| `--use-cuda-graph` | from config | Enable CUDA graph replay |
| `--flush-l2` | from config | Flush L2 before each iteration |
| `--autotune` | off | Enable autotuned execution |
| `--tile-language` | all GPU backends | Comma-separated backends. GPU backends: `triton,cutile,tilelang` (or `all`); PyTorch always runs as the reference. `nki` only runs when named explicitly, see below |
| `--case-indices` | all | Run a subset of cases, for example `0,1,3` |
| `--keep-proton-files` | false | Keep Proton `.hatchet` files |
| `--proton-output-dir` | system temp | Proton output directory |

```bash
python scripts/run_bench.py --gpu B200 --operator mul2 --tile-language triton,cutile,tilelang
```

The run writes the tracked summary CSV to:

```text
results/<gpu>/csv/<operator>_default.csv
results/<gpu>/csv/<operator>_autotune.csv
```

Local logs and generated figures are ignored by Git.

**Raw JSON names.** The default names carry the mode (`default` or `autotune`) and the backend selection, so a default run, an autotune run, a TileLang-only run and an NKI run of one operator each keep their own files:

```text
results/B200/logs/time_measurement_logs/mul2_default_triton-cutile.json
results/B200/logs/time_measurement_logs/mul2_autotune_triton-cutile.json
results/B200/logs/time_measurement_logs/mul2_default_tilelang.json
results/B200/logs/time_measurement_logs/mul2_default_nki.json
```

The backend tag always lists the backends in the canonical order `triton`, `cutile`, `tilelang`, `nki`, so `--tile-language cutile,triton` and `triton,cutile` name the same file; a torch-only run is tagged `torch`. `run_bench.py` owns these names: callers do not rename or copy results. Build them with `timing_log_path` and `autotune_log_path` from `tilebench/paths.py`, and the selection with `tilebench/backends.py`. A reader must name the run it wants; never pick a file by glob, by modification time, or as "the latest". Explicit `--output` and `--autotune-log` paths still win. The summary CSV stays one per mode.

**Hardware namespaces.** `--gpu` is a label, not a device selector: it names the directory that the results of this machine go to, and `run_bench.py` prints it next to the detected device, with a warning when the label does not appear in the device name. It has no default, so a run on a GH200 or an AMD GPU cannot land in `results/B200/` by omission. There is no list of supported labels; any single path component made of letters, digits, `.`, `_`, `+` or `-` is accepted, so a new GPU needs a new label and no code change. Explicit `--output` and `--autotune-log` paths are respected, while the summary CSV always goes to `results/<gpu>/csv/`.

**What a namespace holds.** The PyTorch, Triton, cuTile and TileLang columns of a CSV under `results/<gpu>/csv/` were measured on that GPU. A TileLang-only run is merged into the existing CSV of the same `--gpu`, leaving the frozen PyTorch, Triton and cuTile columns untouched. A backend that the platform does not support is reported as skipped or failed by the engine, with `nan` latency and `<backend>_ok = false`; never record it as a successful result.

**NKI.** NKI runs on AWS Trainium, not on the GPU. It only runs when named explicitly, and `--gpu` then names the campaign whose record the measurements join, without claiming that NKI ran on that GPU:

```bash
python scripts/run_bench.py --gpu B200 --operator mul2 --tile-language nki
```

On the Neuron host this re-times PyTorch on the Neuron device and merges three columns into `results/B200/csv/<operator>_{default,autotune}.csv`:

| Column | Meaning |
|---|---|
| `torch_nki_ms` | PyTorch reference timed on the Trainium device |
| `nki_ms` | NKI latency on Trainium |
| `speedup_nki` | `torch_nki_ms / nki_ms` |

These are cross-hardware measurements kept beside the B200 columns for a unified per-operator record. `torch_ms`, `triton_ms`, `cutile_ms` and `tilelang_ms` remain B200 measurements and are never modified by an NKI merge. NKI latencies are written as measured, with no B200 drift scaling, and `speedup_nki` is never `torch_ms / nki_ms`. NKI timing and autotune JSON go to `results/<gpu>/logs/` like every other log, and the Neuron profiling artifacts go to `results/<gpu>/logs/nki_profiles/` with their audit index `results/<gpu>/logs/nki_neff_manifest.jsonl`.

### `scripts/run_bench_all.py`

Runs the operator suite sequentially with the GPU backends. `--gpu` is required; each run is stored under `results/<gpu>/runs/<timestamp>/` and its `summary.json` records the label. `--results-root` overrides the location.

```bash
python scripts/run_bench_all.py --gpu B200
```

### `scripts/visualize.py`

| Argument | Default | Description |
|---|---|---|
| `--gpu` | required | Hardware label such as `B200`: selects the result namespace and loads `tilebench/data/peak_performance/<gpu>.json` |
| `--operator` | required | Operator name |
| `--mode` | `default` | Which run to read: `default` or `autotune` |
| `--tile-language` | GPU backends | Backend selection of the run to read, as passed to `run_bench.py`; order does not matter |
| `--input` | `results/<gpu>/logs/time_measurement_logs/<op>_<mode>_<backends>.json` | Input timing JSON; an explicit path wins over `--mode` and `--tile-language` |
| `--output-dir` | `results/<gpu>/figures/<op>/` | Figure output |
| `--metrics` | from `config.yaml` | Metrics to plot |

```bash
python scripts/visualize.py --gpu B200 --operator mul2 --tile-language triton,cutile
python scripts/visualize.py --gpu B200 --operator mul2 --tile-language triton,cutile --mode autotune
```

`--gpu` serves two purposes here: it is the result namespace, and it selects the GPU-specific peak metadata behind the roofline and percentage-of-peak metrics. When no `<gpu>.json` exists those metrics are skipped with a warning. Explicit `--input` and `--output-dir` win over the defaults.

Supported derived views include latency, bandwidth, speedup, TFLOPS, percentage of peak, arithmetic intensity, and roofline plots.

### Other result consumers

Every script that reads or writes results takes the same `--gpu` label:

```bash
python scripts/aggregate_results.py --gpu B200               # results/B200/csv/ -> results/B200/aggregate/
python scripts/plot_sweep_max.py --gpu B200                   # -> results/B200/figures/sweep_max_latency.png
python scripts/profiling/ncu_catalogue.py --gpu B200          # -> outputs/profiling/B200/ncu_catalogue.json
```

`ncu_catalogue` records the Triton and cuTile autotune winners, and reads them from exactly one file per operator: `results/<gpu>/logs/autotune_logs/<op>_autotune_triton-cutile.json`. Pass `--tile-language` to read the winners of another autotune run instead; the selection must include `triton` and `cutile`.

Build result paths with the helpers in `tilebench/paths.py` (`results_root`, `results_csv_dir`, `results_logs_dir`, `results_figures_dir`, `results_aggregate_dir`, `results_runs_dir`), never by concatenating strings.

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

Profiling support is split by role, not by file type:

```text
tilebench/profiling/     # importable library, installed with the package
├── ncu_kernel_select.py # kernel selection, capture validation, metadata loading
└── ncu_catalogue.py     # sweep-max cases and catalogue entries
scripts/profiling/       # command-line tools, run from a checkout, not installed
├── ncu_catalogue.py     # build a GPU's catalogue
├── probe_kernel_count.py
├── ncu_one.py, ncu_driver.py, ncu_writeup.py, hf_upload.py
└── ncu_generic_harness.py   # the process NCU profiles; the drivers start it by path
```

A module belongs in `tilebench/profiling/` only if other code imports it: no `argparse`, no `__main__`, nothing that runs at import. A program goes under `scripts/`, and imports the library. Campaign-specific scripts, cluster job files and measured data do not belong in either: everything the tools generate goes under the Git-ignored `outputs/`. The tools put the repository root on `sys.path` themselves, so they run from any directory without `PYTHONPATH`.

NCU metadata is generated per hardware, because autotune winners, kernel launch counts and kernel names are measured on one GPU:

```text
outputs/profiling/<gpu>/
├── ncu_catalogue.json    # profiled operator/dtype cases and selected configurations
└── kernel_counts.json    # expected kernel counts and names, used to validate NCU captures
```

Nothing is committed for any GPU, and nothing here is needed to run benchmarks: only the NCU tools read these files. `scripts/plot_sweep_max.py` does not need them either; it derives each operator's sweep-max case from `config.yaml` with the same rule as the catalogue (`ncu_catalogue.sweep_max_cases`). Use `ncu_catalogue_path`, `kernel_counts_path` and `ncu_output_dir` from `tilebench/paths.py`; there is no global copy. Every tool that touches this metadata takes `--gpu`, required and without a default:

```bash
python scripts/profiling/ncu_catalogue.py --gpu GH200           # writes outputs/profiling/GH200/ncu_catalogue.json
python scripts/profiling/probe_kernel_count.py --gpu GH200      # writes outputs/profiling/GH200/kernel_counts.json
python scripts/profiling/ncu_one.py --gpu GH200 mul2 fp16       # one operator
python scripts/profiling/ncu_driver.py --gpu GH200              # the whole catalogue
python scripts/profiling/ncu_writeup.py --gpu GH200
```

Generated NCU reports are written under `outputs/ncu/<gpu>/` and are ignored by Git, so the reports of two GPUs never collide. The released Hugging Face dataset holds the paper's 220 B200 reports; `hf_upload.py --gpu <gpu>` uploads under a per-GPU prefix and never writes over them.

If the catalogue or `kernel_counts.json` of the requested GPU is missing, the tools fail with a clear error that names the command to produce it. They never fall back to another GPU's metadata, and never silently assume one kernel per launch. Missing metadata for an individual pair may fall back to one kernel with an explicit warning.

## Neuron Stack Diagnostics

**Formal Trn2 measurement.** `run_bench.py --gpu TRN2 --tile-language nki` on a Trainium host runs on the native PyTorch Neuron stack (`tilebench/core/neuron_native.py`): PyTorch eager `impl_torch.run` on `torch.device("neuron")` against the NKI `run()` with the same neuron tensors, warmup 1 / repeat 3 unless given on the command line. torch.compile is disabled for the whole run, so a `torch.compile` call inside an implementation runs eagerly; there is no torch-xla path and no best-baseline. Both outputs are verified against the CPU reference. The latency is the mean over the repeats of the device busy sum of one `run()` (the durations of its device executions added up, per-core copies of one execution merged; one `torch.profiler` `NeuronConfig(RUNTIME)` session per synchronized call), the same definition as the GPU path, where Proton adds up the CUDA kernel durations inside the scope. Host wall time is recorded next to it and never used in a speedup. `speedup_nki = torch_ms / nki_ms` (PyTorch eager device time over NKI device time).

NKI results live only in their Neuron hardware namespace (`results/TRN2/`, the label must match the detected device): `csv/<op>_<mode>.csv` has the columns `params, dtype, torch_ms, nki_ms, speedup_nki, torch_status, nki_status`, and `logs/hardware_provenance/` records the instance type, LNC, driver and package versions. `run_bench.py` refuses to write NKI into a GPU namespace or to mix NKI with GPU backends. `torch_status` is `ok`, `failed` (did not verify, or no device time) or `unresolved` (`neuron_native.BASELINE_UNRESOLVED`, not run); `nki_status` is `ok`, `failed` or `unsupported` (the implementation raised `NotImplementedError`). Only `ok`/`ok` rows carry a speedup; the others are `nan`. Peak/roofline metadata is looked up by the namespace label, so TRN2 never reads a GPU's peak file. Compile caches and traces stay under `results/TRN2/logs/neuron_native_{cache,profiles}/` (not version controlled). Run it with the interpreter of the native environment, whose `neuronx-cc` is put first on `PATH`.

`scripts/neuron_diag.py` is the diagnostics harness on the same stack (isolated worker per case, input bundles, row validity). It is a separate entry point: it does not change `run_bench.py`, the engine, or anything under `results/`.

| Mode | Role | Meaning |
|---|---|---|
| `native_torch_eager` | benchmark | PyTorch baseline on the native `neuron` device, eager |
| `native_nki` | benchmark | the NKI implementation called with native `neuron` tensors |
| `native_torch_compiled` | diagnostic | the baseline under `torch.compile(backend="neuron")`, static shapes; `run --compile-diagnostics` only |
| `xla_torch`, `xla_nki` | legacy diagnostics | PyTorch/XLA paths, kept only to read archived runs; `run --legacy-xla-diagnostics` only |

```bash
python scripts/neuron_diag.py env     --run-id R        # creates .local/neuron_native_diagnostics/R/
python scripts/neuron_diag.py sources --run-id R        # pins each NKI PR head, builds overlays
python scripts/neuron_diag.py run     --run-id R --ops rmsnorm --cases pilot --native-python <venv>/bin/python
python scripts/neuron_diag.py report  --run-id R
```

- **Benchmark vs diagnostics.** Reports, speedups, coverage counts and tables use `native_torch_eager` and `native_nki` only (`report.benchmark_records`). `run` refuses the `xla_*` modes unless `--legacy-xla-diagnostics` is passed; `report --legacy-xla` writes their separate view to `legacy_xla/report.md`. A run directory containing `ARCHIVED_LEGACY_XLA.json` is frozen. Some `impl_torch.py` files keep a legacy `device.type == "xla"` branch; native runs the normal `neuron`/generic path and the branch does not affect comparability.
- **Algorithm matching.** PyTorch is the semantic reference and practical baseline and need not use the same algorithm. Algorithm matching is judged among Triton, cuTile, TileLang and NKI on the high-level algorithm and math semantics only; core mapping, program_id split, tile shapes, SBUF residency and data reuse, kernel fusion and launch counts, DMA/PSUM/partition layout, hardware arithmetic primitives and backend tuning knobs are allowed to differ.

- **Sources.** NKI implementations live on unmerged PR branches. `sources` pins each branch head, reads the operator files with `git show`, applies only an import-path patch for the pre-package layout (hashes before and after are recorded), and builds an overlay tree per operator: this checkout's `tilebench` package with that operator directory replaced. Both stacks import the same files. The static scan in the manifest reports what the source contains, not what the device executes.
- **Inputs.** Each case gets two CPU bundles (seed and seed+1) with the CPU reference of the pinned `impl_torch.run`, saved with checksums and loaded by every mode. The second bundle checks that a changed input gives a correct new output. The correctness contract is `main`'s `config.yaml` `verify` section; a looser branch override is recorded and not applied.
- **Timing.** `wall_ms` is host time of `run()` plus device synchronization after warmup (`wall_timing_method = wall_sync`), split into dispatch and sync wait. `device_ms` is filled only by a device trace with an executed-artifact identity (`timing_method = native_device_trace`: one `torch.profiler` `NeuronConfig(RUNTIME)` session per synchronized `run()`, busy sum of its executions; `neuron_rt_inspect` in legacy XLA rows); otherwise it is null with a reason. First-call (compile) time is recorded separately. Speedups are only formed within one stack and one timing method.
- **CPU fallback.** On the native stack the state stays `unable_to_determine` until a fallback report API of the installed build has been verified.
- **Runtime adapters.** `tilebench/neuron_diag/runtime.py` probes native APIs (device, synchronize, dynamo metrics) on the installed build and records which one it used. If a required API is missing, the mode is reported as `blocked_env`; nothing is stubbed.
- **Storage.** A run directory is created exclusively, files are never overwritten, and `run` resumes by skipping only (operator, case, mode) rows finished with the same source and environment hash. Everything stays under the Git-ignored `.local/`.

## LLM Code Generation

The iterative generation pipeline is under:

```text
tilebench/llm/
```

Framework conventions, prompt construction, evaluation, and feedback logic live there; run it with `python -m tilebench.llm.generate`. The task descriptions it reads, one `<operator>_current.md` per operator, are under `tilebench/problems/` (`PROBLEMS_ROOT` in `tilebench/paths.py`), a data directory beside `benchmarks/`, `core/` and `data/`. Backend API guides are under `skills/`.

Each run writes its trajectory (prompts, responses, kernels, feedback, token usage) and the selected implementation to:

```text
tilebench/benchmarks/llm_generated/<operator>/<model>/<effort>/
```

This directory is the pipeline's default output location and is Git-ignored: the pipeline creates it on demand, and nothing under it is committed to `main`.

Keep LLM generation separate from the manually implemented benchmark path. Generated code must not delegate the operator computation to PyTorch, vendor libraries, or backend autotuners when the generation protocol forbids them.

## Generated Outputs

Results are scoped by hardware. The repository tracks only per-case benchmark CSVs under:

```text
results/<gpu>/csv/
```

The committed paper results are `results/B200/csv/`. Other artifacts are generated locally and ignored, including:

```text
results/<gpu>/logs/
results/<gpu>/figures/
results/<gpu>/aggregate/
results/<gpu>/runs/
outputs/
tilebench/benchmarks/llm_generated/
```

Writers should create these directories when needed; a fresh clone must not depend on pre-existing generated directories.

Running a benchmark only writes files under these paths. It performs no Git operation and backs nothing up. The paper's frozen raw logs, LLM trajectories and B200 NCU metadata are preserved on the `archive/raw-logs-2026-09-18` branch.

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
- [ ] No generated artifacts outside `results/<gpu>/csv/` are accidentally tracked.
