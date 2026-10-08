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
- [Multi-Architecture Status (TileBench++)](#multi-architecture-status-tilebench)
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
| `--use-cuda-graph` | from config | Request CUDA graph replay (eager on ROCm, see below) |
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

**Architecture-specific code paths.** Code that depends on the hardware asks `tilebench/hardware.py`, never the `--gpu` label: `detect_arch()` returns `"blackwell"` (sm_100), `"hopper"` (sm_90), `"cdna3"` (gfx942) or `None` (no CUDA/HIP GPU, or a device outside this scope), probed from the actual device on first use and cached; `supports_tmem()` and `supports_tma()` are built on it. Importing the module does not initialise the GPU, and on a host without one every probe returns `None`/`False`. An operator that needs two paths keeps one source file and the same kernel names, and branches in Python before the kernel is built. The declared autotune candidates are identical on every architecture: a candidate that does not compile or run on a device fails and is skipped by the existing error handling, it is never filtered out in advance. The timer evicts `2 x last_level_cache_bytes()` before every warmup and timed launch; that is the runtime L2 size unless a measured device-level cache size is registered for the architecture in `hardware._LLC_BYTES`. CDNA3 (MI300X) registers its 256 MiB Infinity Cache, so the timer evicts 512 MiB there instead of 2x the 4 MiB runtime L2. The value comes from an eviction sweep on MI300X (two HBM-bound probes, eager and CUDA-graph, randomized order): a 256 MiB buffer evicts only sometimes, from 384 MiB the probes are cold, and 512 and 768 MiB agree within 1.3%. An architecture in `hardware._LLC_CALIBRATION_REQUIRED` without such an entry fails a flushed measurement with `UncalibratedCacheError` instead of falling back to the L2. `use_cuda_graph` is the requested timing mode; `tilebench.core.timer.effective_use_cuda_graph()` decides the executed one from the torch build, never from `--gpu`: NVIDIA replays the CUDA graph, ROCm always times eagerly, because with Proton's roctracer backend a HIP Graph replay is recorded as its first child kernel only (bitonic_sort on MI300X: 1 of 191 kernels per replay). The run prints this once, and the provenance sidecar records `requested_use_cuda_graph`, `effective_use_cuda_graph`, `timing_execution_mode` and `timing_note` under `timing` (per operator in `run_bench_all.py`'s `summary.json`); the timing logs keep their format.

**Provenance.** Every run records which source, software stack and device produced it, without changing the result JSON formats. `run_bench.py` writes a sidecar with the same file name as the run's logs, `results/<gpu>/logs/provenance/<op>_<mode>_<backends>.json` (`provenance_log_path`; with an explicit `--output`, next to it as `<output stem>.provenance.json`), and `run_bench_all.py` adds the same record under the `provenance` key of `results/<gpu>/runs/<timestamp>/summary.json`. The record (`tilebench/provenance.py`, schema `tilebench-provenance/1`) is captured before any measurement:

| Block | Fields |
|---|---|
| `source` | `git_sha` (full SHA of `HEAD`), `dirty` (a tracked file differs from `HEAD` or an untracked, non-ignored file exists; paths under `results/` never count), `tracked_dirty` (tracked modifications only), `dirty_files`, `untracked_files` |
| `software` | `python`, `torch`, `torch_cuda`, `torch_hip`, `triton`, `tilelang`, `cuda_tile` (`None` when not installed) |
| `device` | `requested_label` (`--gpu`), `name`, `vendor`, `arch` (`detect_arch()`), `compute_capability`, `gcn_arch_name` |
| `host` | `hostname`, `argv` |
| `run` | `run_bench.py`: operator, mode, backends, overrides, the timing log, autotune log and summary CSV it describes, and `tilelang_autotuner_log` (with `tilelang_autotuner_log_note` when nothing was collected). `run_bench_all.py`: `tile_language` and `backends` |

A field that cannot be determined is `None` (with `source.error` when Git is unavailable); collecting provenance never fails a run. The sidecar lives under `logs/`, so it is Git-ignored and archived with the raw logs.

**TileLang autotuner log.** TileLang 0.1.11 writes `autotuner.log` to the working directory: the first `AutoTuner.run()` of a process (default mode included) truncates it, later tunings in that process append to it, cache hits write nothing, and a process that never tunes leaves an older file in place. The file is therefore Git-ignored and never archived as is. When TileLang runs, the runners copy the part of it that the current process wrote for the current operator (`tilebench/core/tilelang_log.py`: the byte range of this process's own log handler, refused if the file was replaced or written by another process) to `results/<gpu>/logs/tilelang_autotuner/`: `<op>_<mode>_<backends>.log` for `run_bench.py` (the timing log's stem; next to an explicit `--output` as `<output stem>.tilelang_autotuner.log`), and `<op>_<run-id>.log` for `run_bench_all.py`, whose `summary.json` lists them under `artifacts.tilelang_autotuner`. A run that tuned nothing (cache hit) leaves no file. `--logs` archives them with the other logs.

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

Runs the operator suite sequentially. `--gpu` is required; each run is stored under `results/<gpu>/runs/<timestamp>/` and its `summary.json` records the label. `--results-root` overrides the location. `--tile-language` selects the backends exactly as in `run_bench.py` (`tilebench.backends.parse_backends`); without it the engine's default selection (`triton`, `cutile`, `tilelang`, and `nki` when importable) runs, as before. The selection passed to the engine is recorded in `summary.json` under `provenance.run.backends`, with the flag as given in `provenance.run.tile_language`.

```bash
python scripts/run_bench_all.py --gpu B200
python scripts/run_bench_all.py --gpu MI300X --tile-language triton
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

`ncu_catalogue` records the Triton and cuTile autotune winners, and reads them from exactly one file per operator: `results/<gpu>/logs/autotune_logs/<op>_autotune_triton-cutile.json`. Pass `--tile-language` to read the winners of another autotune run instead; the selection must include `triton` or `cutile`, the backends the profilers replay (a GPU without cuTile, such as MI300X, passes `--tile-language triton`; its catalogue records no cuTile winners).

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
  warmup: 1
  repeat: 3
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

They go in the operator's `verify:` section (`atol`, `rtol`). An `arch_overrides:` entry keyed by `tilebench.hardware.detect_arch()` (`blackwell`, `hopper`, `cdna3`) replaces the values it names on that architecture only, e.g. `2d_conv` keeps `atol: 1e-1` and sets `arch_overrides: {cdna3: {atol: 2e-1}}`.

## Autotuning

Every `triton.autotune`, `tilelang.autotune` and in-operator `do_bench` candidate loop times each candidate with `warmup=1, rep=3`; cuTile uses its native exhaustive search with crash isolation. The candidate lists are the same on every architecture. See [Common timing and autotune protocol](#common-timing-and-autotune-protocol) for details and for which campaigns used this budget.

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
├── ncu_catalogue.py     # sweep-max cases and catalogue entries
├── replay.py            # one pair's inputs and exact autotune winner, shared by all harnesses
└── rocprof_compute.py   # ROCm Compute Profiler environment, selection, capture validation
scripts/profiling/       # command-line tools, run from a checkout, not installed
├── ncu_catalogue.py     # build a GPU's catalogue
├── probe_kernel_count.py
├── ncu_one.py, ncu_driver.py, ncu_writeup.py, hf_upload.py
├── ncu_generic_harness.py   # the process NCU profiles; the drivers start it by path
├── rocprof_compute_driver.py, hf_upload_rocm_compute.py
└── rocprof_compute_harness.py   # the process rocprof-compute profiles
```

A module belongs in `tilebench/profiling/` only if other code imports it: no `argparse`, no `__main__`, nothing that runs at import. A program goes under `scripts/`, and imports the library. Campaign-specific scripts, cluster job files and measured data do not belong in either: everything the tools generate goes under the Git-ignored `outputs/`. The tools put the repository root on `sys.path` themselves, so they run from any directory without `PYTHONPATH`.

NCU metadata is generated per hardware, because autotune winners, kernel launch counts and kernel names are measured on one GPU:

```text
outputs/profiling/<gpu>/
├── ncu_catalogue.json    # profiled operator/dtype cases and selected configurations
└── kernel_counts.json    # expected kernel counts and names, used to validate profiler captures
```

`kernel_counts.json` holds one row per probed pair: `count` is the number of operator compute-kernel launches of ONE `impl.run()`, and `names` is that launch sequence in order with repeats kept (a pipeline `A, B, B, C` is four launches, never deduplicated). Memcpy/memset, ATen/library helpers and runtime blit kernels (on ROCm a device-to-device copy is a dispatch, `__amd_rocclr_copyBuffer`) are not operator kernels (`ncu_kernel_select.is_aux_kernel`); the probe lists them under `excluded`. `first_call_identical` records whether the first call of the pair launches the same compute sequence as a later call. The probe installs the exact autotune winner the way the harnesses do (`tilebench/profiling/replay.py`), so the counted configuration is the profiled one, and records a pair the GPU's autotune run has no result for (an unsupported dtype) as `skipped` instead of running it. `--tile-language` selects the probed backends (default `triton,cutile`; MI300X probes `triton`). torch.profiler reports HIP kernels on ROCm as well; on MI300X its names and counts match a rocprofv3 kernel trace.

Nothing is committed for any GPU, and nothing here is needed to run benchmarks: only the NCU tools read these files. `scripts/plot_sweep_max.py` does not need them either; it derives each operator's sweep-max case from `config.yaml` with the same rule as the catalogue (`ncu_catalogue.sweep_max_cases`). Use `ncu_catalogue_path`, `kernel_counts_path` and `ncu_output_dir` from `tilebench/paths.py`; there is no global copy. Every tool that touches this metadata takes `--gpu`, required and without a default:

```bash
python scripts/profiling/ncu_catalogue.py --gpu GH200           # writes outputs/profiling/GH200/ncu_catalogue.json
python scripts/profiling/probe_kernel_count.py --gpu GH200      # writes outputs/profiling/GH200/kernel_counts.json
python scripts/profiling/probe_kernel_count.py --gpu MI300X --tile-language triton
python scripts/profiling/ncu_one.py --gpu GH200 mul2 fp16       # one operator
python scripts/profiling/ncu_driver.py --gpu GH200              # the whole catalogue
python scripts/profiling/ncu_writeup.py --gpu GH200
```

Generated NCU reports are written under `outputs/ncu/<gpu>/` and are ignored by Git, so the reports of two GPUs never collide. The Hugging Face dataset `bcui2/NCU_report` has one top-level folder per hardware; `NVIDIA_B200/` holds the paper's 220 B200 reports. `hf_upload.py` uploads the `*.ncu-rep` files of one GPU and nothing else. It takes two independent names: `--gpu`, the local TileBench label whose `outputs/ncu/<gpu>/` is read, and `--hf-folder`, the dataset folder to write, a single path component that is never derived from `--gpu` (there is no mapping between the two). Each operator keeps its directory, `<hf-folder>/<op>/`, and an operator argument uploads only that one:

```bash
python scripts/profiling/hf_upload.py --gpu GH200 --hf-folder NVIDIA_GH200            # outputs/ncu/GH200/ -> NVIDIA_GH200/
python scripts/profiling/hf_upload.py --gpu GH200 --hf-folder NVIDIA_GH200 1d_conv    # outputs/ncu/GH200/1d_conv/ -> NVIDIA_GH200/1d_conv/
```

Profiler output of other hardware is not handled by this uploader: ROCm Compute Profiler artifacts have their own (below), Neuron output is not handled.

If the catalogue or `kernel_counts.json` of the requested GPU is missing, the tools fail with a clear error that names the command to produce it. They never fall back to another GPU's metadata, and never silently assume one kernel per launch. Missing metadata for an individual pair may fall back to one kernel with an explicit warning.

### ROCm Compute Profiler (AMD)

The AMD counterpart of the NCU pipeline keeps its invariants: the sweep-max case of the GPU's catalogue, the exact autotune winner, one profile per (operator, dtype, backend), input generation, warmup and cache eviction outside the profiled set, and every capture validated against `kernel_counts.json` of the same GPU, launch by launch. Every valid catalogue pair is profiled (a dtype without an autotune result on the GPU is excluded and listed, never replaced); the backend is Triton.

`rocprof-compute` (ROCm Compute Profiler, 3.7.0 with ROCm 7.14 on MI300X) has no profiler start/stop range, and its dispatch filter `-d` counts iterations per kernel, not across a run: with three warmups of an `A, B, B, C` pipeline, `-d 4` captures the fourth dispatch of each kernel, one of them inside a warmup. No `-d` range selects one whole call after warmups when a call's kernels launch different numbers of times. `rocprof_compute_harness.py` therefore keeps the warmup out of the profiled process: `PROF_MODE=prime` runs the pair three times without a profiler (JIT warmup into Triton's cache), and the profiled process generates the inputs, evicts the last-level cache like the timer (2x the LLC, 512 MiB on MI300X) and runs ONE `impl.run()`, its first call. The kernel-name filter `-k` alone then selects exactly that call. The probe's `first_call_identical` guards the premise, and the capture is validated anyway. On the MI300X pilot the deterministic counters of this protocol and of an NCU-style warmed process (three warmups, eviction, target selected with `-d 4`, exact for a single-kernel operator) are identical, and duration, CU utilization and active CUs fall within the same run-to-run spread.

Facts the driver relies on, measured with 3.7.0:

- `-k` matches from the start of the kernel name. A regex containing `$` silently disables the filter and every dispatch is profiled, so the selection is `^(?:name|...)` and the validation checks exact names.
- The default profile (no `--block`/`--set`, roofline on) runs one workload execution per counter pass (13 on MI300X) and collects every block gfx942 has: 0-2, 4-7, 10-18 and the roofline microbenchmark; the memory chart (3) is derived from the same counters. `--pc-sampling` turns a profile into a PC-sampling-only run (block 21), so PC sampling is a separate pass.
- A workload whose HIP/HSA runtime comes from the PyTorch wheel aborts counter collection with `aqlprofile API table load failed` unless `<rocm>/lib` is on `LD_LIBRARY_PATH`, which rocprofv3 appends itself; the driver appends it too (torch and Triton keep the runtime from `torch/lib`). On a multi-version ROCm layout (`/opt/rocm/core-<ver>/`) it also sets `ROCM_VER`, which rocprof-compute reads when `/opt/rocm/.info/` is absent. rocprof-compute's Python requirements are installed apart from the benchmark environment; `--rocprof-compute-python` names that interpreter.

One pair is `outputs/rocprof_compute/<gpu>/<op>/<backend>_<dtype>/` (`rocprof_compute_pair_dir`): `workload/` is the complete directory `rocprof-compute profile` wrote, the canonical artifact, never edited; `pc_sampling/` is the PC-sampling pass (stochastic, every 65536 cycles), a workload of its own whose `ps_file_results.json` holds each sample's PC, decoded instruction, source line and stall reason; `analysis/` holds output derived from both (`rocprof-compute analyze` text reports and CSV exports, made on copies because `analyze` writes its joined tables into the directory it reads, and `pc_sampling_instructions.csv`, the per-instruction PC-sampling table, which the CSV export lacks); `logs/` has the console output and `capture.json` the validation record. Validation requires, in every counter pass, exactly the manifest's launch sequence: non-empty, operator kernels only, same count and same order; the PC-sampling pass, which samples the whole process, must show the same operator dispatch sequence.

```bash
python scripts/profiling/rocprof_compute_driver.py --gpu MI300X \
    --rocprof-compute /opt/rocm/bin/rocprof-compute --rocprof-compute-python <python> [--ops vector_add --dtypes fp32]
python scripts/profiling/hf_upload_rocm_compute.py --gpu MI300X --hf-folder AMD_MI300X [vector_add] [--verify]
```

The driver is resumable: it writes `sweep_log.json` after every pair, skips a pair recorded ok only after its artifact validates again, keeps the error of a failed pair, and ends with `coverage.json` (valid pairs, successes, failures, exclusions). `hf_upload_rocm_compute.py` uploads complete pair directories, every file whatever its extension, of the pairs the sweep log records ok and that validate again, to `<hf-folder>/<op>/<backend>_<dtype>/` of `bcui2/NCU_report`, with the two records; `--hf-folder` is independent of `--gpu`. A download is re-analyzed with `rocprof-compute analyze -p <...>/workload` without running the kernel.

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

The committed results are `results/B200/csv/` (the paper's measurements plus TileLang), `results/GH200/csv/` and `results/MI300X/csv/`; see [Multi-Architecture Status](#multi-architecture-status-tilebench). Other artifacts are generated locally and ignored, including:

```text
results/<gpu>/logs/
results/<gpu>/figures/
results/<gpu>/aggregate/
results/<gpu>/runs/
outputs/
tilebench/benchmarks/llm_generated/
```

Writers should create these directories when needed; a fresh clone must not depend on pre-existing generated directories.

Running a benchmark only writes files under these paths. It performs no Git operation and backs nothing up. The paper's frozen raw logs, LLM trajectories and B200 NCU metadata are preserved on the `archive/raw-logs-2026-09-18` branch, which is frozen.

TileBench++ artifacts are archived on `archive/tilebenchpp-2026-10`, by the maintenance tool that lives on that branch only (`scripts/archive_artifacts.sh`, with its tests). It snapshots `results/<gpu>/logs/` (`--logs`, provenance sidecars included), `outputs/profiling/<gpu>/` (`--profiling`) and `tilebench/benchmarks/llm_generated/` (`--llm`); `--all --gpu <gpu>` is all three. `--base` names the source commit the artifacts were produced from; it is recorded as an exact SHA and kept reachable from the archive. A run only adds or updates files, so the artifacts other machines archived are never dropped. Summary CSVs stay on the source branches; NCU reports (`*.ncu-rep`) and ROCm Compute Profiler workloads (`outputs/rocprof_compute/`) go to external storage only:

```bash
git show origin/archive/tilebenchpp-2026-10:scripts/archive_artifacts.sh > /tmp/archive_artifacts.sh
bash /tmp/archive_artifacts.sh --branch archive/tilebenchpp-2026-10 \
    --base <source.git_sha of the runs' provenance> --logs --profiling --gpu GH200 --push
```

### Publishing a downloadable artifact

Downloadable artifacts are declared in `artifacts/manifest.json` (URL, SHA256, archive name, and the directories the archive provides). To publish one:

```bash
python scripts/package_artifacts.py --artifact llm-aacl2026
```

writes a reproducible `outputs/artifacts/<archive>.tar.gz` and prints its SHA256. Upload the archive, then put the file's share link and that checksum in the manifest. Readers restore it with `python scripts/fetch_artifacts.py --artifact <name>`.

## Multi-Architecture Status (TileBench++)

TileBench++ runs the same 45 operators, case grids and autotune candidate lists on three GPUs. `feature/tilebenchpp-multiarch` (f50f9225) integrates the two experiment branches `exp/gh200` (5610f18f) and `exp/mi300x` (124fdc94). Both are frozen. There is no B200 experiment branch: the B200 results are the ones committed on `main`.

### Hardware coverage

The table records how the committed results were produced. The protocol of the current source, which future runs use, is described in [Common timing and autotune protocol](#common-timing-and-autotune-protocol).

| | B200 | GH200 | MI300X |
|---|---|---|---|
| Device, `detect_arch()` | NVIDIA B200, sm_100, `blackwell` | NVIDIA GH200 480GB, sm_90, `hopper` | AMD Instinct MI300X, gfx942, `cdna3` |
| Backends | PyTorch, Triton, cuTile, TileLang | PyTorch, Triton, cuTile, TileLang | PyTorch, Triton. cuTile and TileLang are not run |
| Summary CSVs | 45 default + 45 autotune, 2200 cases per mode | 45 + 45, 2200 cases per mode | 45 + 45, 2180 cases per mode: the 20 `matmul_fp32_fp16_fp8` / `fp8_e4m3fn` cases are unsupported, because the PyTorch ROCm reference rejects e4m3fn |
| Timing | Proton, CUDA graph | Proton, CUDA graph | Proton/roctracer, eager (graph requested, `effective_use_cuda_graph = false`) |
| Eviction before each launch | PyTorch/Triton/cuTile columns: a fixed 64 MB buffer (the timer before #257) for 41 operators; 2x L2, about 253 MB, for `cross_entropy`, `flash_decode`, `moe_topk_gating` (re-measured in #257) and `linear_self_attention` (#258). TileLang columns depend on the run date: 64 MB for 51 of the 90 files and 2x L2 for 6, judged from the timer in the commit that first holds the values; not recoverable for the other 33, which first appear in a squash merge | 120 MiB (2x 60 MiB L2) | 512 MiB (2x 256 MiB Infinity Cache) |
| Final measurement (warmup / repeat) | not recorded; `config.yaml` at every result commit has 20 / 100 | 1 / 3 (`--warmup 1 --repeat 3`) | 20 / 100 (`config.yaml` at 005ab63b and 4d08985a) |
| Software | PyTorch/Triton/cuTile columns from the paper campaign (`cuda-tile` 1.3.0); TileLang 0.1.11 | torch 2.10.0+cu130, Triton 3.6.0, `cuda-tile` 1.5.0 (pip `tileiras` 13.4.92), TileLang 0.1.11 | torch 2.10.0+rocm7.1, Triton 3.6.0, ROCm 7.14.0 |
| Benchmark source | not recorded; the CSVs predate `tilebench/provenance.py` | c882fe50 for 89 CSVs, 3c5eccbf for `batched_matmul_autotune.csv` | 005ab63b default, 4d08985a autotune |
| Raw logs | `archive/raw-logs-2026-09-18` (frozen) | `archive/tilebenchpp-2026-10` 66046918 | `archive/tilebenchpp-2026-10` 06a8ed00 (default), 8e991537 (autotune) |

GH200 TileLang dispatch on Hopper:
- Nine operators have a Blackwell TileLang path that accumulates in tensor memory (TMEM, tcgen05), which sm_90 cannot compile: `1d_conv`, `2d_conv`, `3d_conv`, `batched_matmul`, `block_sparse_attention`, `flash_attention`, `matmul_fp32_fp16_fp8`, `matmul_int8` and `streamk_matmul`.
- Each of them branches at module level on `supports_tmem()`. Blackwell keeps the TMEM path. Hopper builds a fragment-accumulator body with the same kernel names and the same candidate list.
- `dequantize_rowwise` sign-extends int8 explicitly on aarch64 hosts (`_EXPLICIT_INT8_SIGN_EXTENSION`), because plain `char` is unsigned there. This depends on the host CPU, not on the GPU.

### Common timing and autotune protocol

- **Final measurement.**
  - In the current source, every operator's `config.yaml` sets warmup 1 / repeat 3 (iterations). The engine falls back to the same values (`tilebench.core.timer.DEFAULT_WARMUP`, `DEFAULT_REPEAT`) when a config omits them, and `--warmup` / `--repeat` override them.
  - The committed results predate this default. GH200 ran 1 / 3 through `--warmup 1 --repeat 3`. MI300X ran with the config values of that time, 20 / 100, and the B200 result commits carry the same config values; see the table above.
  - Each run's provenance sidecar records its overrides (`run.overrides`, `host.argv`).
- **Triton and TileLang autotune.** Every candidate is timed with `warmup=1, rep=3` (time budgets in ms). This covers all 103 `warmup`/`rep` settings in the operator sources. GH200 (since b7f69a24, included in c882fe50) and the MI300X autotune run (4d08985a) used this budget. The B200 autotune CSVs predate it. In the source on `main`, Triton uses its framework default (25 / 100 ms) in 27 operators, 3 / 10 in `kl_divergence` and `matmul_int8`, 5 / 20 in `softmax` and 1 / 3 in the remaining 15, and most TileLang operators use 20 / 100. This is inferred from the source, because B200 has no provenance.
- **cuTile autotune.** `CutileAutotuner` wraps the native `ct.tune.exhaustive_search`, which exposes no warmup/rep setting.
- **Crash isolation.**
  - Every `exhaustive_search` call passes `single_run_timeout_sec=CRASH_ISOLATION_TIMEOUT_SEC` (60 s, `tilebench/core/cutile_autotune.py`). This covers `CutileAutotuner` and the direct call in `top_k_selection`.
  - Each candidate's first warmup launch runs in a worker subprocess. A candidate that faults the device or runs longer than 60 s is recorded in `TuningResult.failures` and skipped. The remaining warmups and all timing run in the parent process, so the timing of successful candidates is unchanged.
  - It was added in d6a6510e after a GH200 fp32 candidate of `batched_matmul` (tile 128x64x64, occupancy 4) raised a sticky CUDA illegal instruction. GH200 `batched_matmul_autotune.csv` was re-run with it (3c5eccbf). The other 44 GH200 autotune CSVs ran before it, on c882fe50, and hit no device fault.
- **Candidate lists.** They are identical on every architecture and are never filtered by shared memory, LDS or wavefront size. A candidate that fails on a device is skipped by the error handling. `tests/test_measurement_protocol.py` checks the warmup/rep settings above and compares every candidate list, contents and order, with the snapshot `tests/data/autotune_candidates.json`.

### Architecture-specific legality fallbacks

A fallback only replaces the builtin default config of the non-autotuned path, and only on an architecture where that default cannot launch. The replacement is always a member of the unchanged search space.

| Operator (Triton) | Architecture | Builtin default | Fallback | Reason |
|---|---|---|---|---|
| `streamk_matmul` | Hopper, fp32 | 128x128x64, 3 stages | `BLOCK_K` 32 | 262168 B shared memory > 232448 B per block on sm_90 |
| `streamk_matmul` | CDNA3, all dtypes | 128x128x64, 3 stages | `BLOCK_K` 32 | 131072 B LDS for fp32 > 65536 B on gfx942 |
| `batched_matmul` | CDNA3, all dtypes | 128x128x32, 4 stages | 3 stages | 98304 B LDS for fp32 > 65536 B on gfx942 |

`_default_config()` resolves the config in this order:
1. A replayed winner, i.e. a config the profiling replay (`tilebench/profiling/replay.py`) wrote into `_DEFAULT_CONFIG`. It is detected by identity against `_BUILTIN_DEFAULT_CONFIG`.
2. The architecture fallback.
3. The builtin default.

Before 124fdc94 the CDNA3 path returned the fallback unconditionally. The six MI300X reports of these two operators therefore profiled the fallback and not the winner, and were re-profiled.

### bitonic_sort pad block

- `pad_kernel` is not tuned. Both the default and the autotune path launch it with a fixed `_PAD_BLOCK = 1024`.
- Only `bitonic_step_kernel` reads `_DEFAULT_CONFIG`, so a replayed autotune winner changes only the step kernel.
- Before this fix (5610f18f on GH200, 124fdc94 on MI300X), the replay also changed the pad launch:
  - GH200 fp32: BLOCK 512.
  - MI300X fp16/fp32: BLOCK 2048/4096.
- The four affected Triton reports were re-profiled and replaced. The formal CSVs were unaffected: the default and autotune launch sequences did not change.

### 2d_conv tolerance

`2d_conv` verifies with `atol: 1e-1` and `rtol: 1e-2` on B200 and GH200. On CDNA3, `verify.arch_overrides.cdna3` raises `atol` to `2e-1` for the ROCm/MIOpen fp16 reference. `verifier.config_tolerance` resolves the override by `detect_arch()`. `2d_conv` is the only operator with a GPU override.

On the native Trainium stack the engine resolves the key `trn2` (`neuron_native.NEURON_ARCH`) instead of `detect_arch()`, for the PyTorch eager baseline and NKI alike; no GPU key applies there. `cross_entropy` uses it (`arch_overrides.trn2`: `atol` = `rtol` = `1e-3`, added with its NKI implementation): the tolerance its formal Trn2 results were verified with. The GPU tolerances of every operator are unchanged.

### Profiling inventory

| | B200 | GH200 | MI300X |
|---|---|---|---|
| Profiler | Nsight Compute | Nsight Compute 2025.4.0, `--set full` | ROCm Compute Profiler 3.7.0 (13 counter passes + roofline) and a separate PC-sampling pass |
| Coverage | 220 reports: Triton 110, cuTile 110 (no TileLang) | 330 reports: Triton 110, cuTile 110, TileLang 110 | 109 Triton pairs, every valid pair; `matmul_fp32_fp16_fp8` / `fp8_e4m3fn` excluded |
| `bcui2/NCU_report` folder | `NVIDIA_B200/` (layout commit 7316fd6c) | `NVIDIA_GH200/`, revision 7cb81050 | `AMD_MI300X/`, revision 645591bb |
| Profiling source | not recorded | Triton/cuTile 638ea849; TileLang d6ddb622; Triton `bitonic_sort` fp16/fp32 5610f18f | 101 pairs 5b82f8af; 8 replaced pairs 124fdc94 |
| Metadata archive | `archive/raw-logs-2026-09-18` 186c6caf (`outputs/profiling/B200/`) | `archive/tilebenchpp-2026-10` f71f7651, 3df90a5f | `archive/tilebenchpp-2026-10` 94e05ffe, 63748286 |

Dataset revision 7cb81050 holds the current reports of all three folders. For GH200 and MI300X, the authoritative per-report record is `outputs/profiling/<gpu>/{ncu_sweep,rocprof_compute_sweep}/PROVENANCE.md` and `report_manifest.json` on the archive branch. It lists sources, HF commits, content hashes and the superseded versions of replaced reports.

### Reading the results across GPUs

- **B200 `tilelang_ms` is direct.** The committed TileLang columns contain unscaled runtimes. A TileLang-only merge preserves the frozen PyTorch/Triton/cuTile columns and writes the measured TileLang time without torch drift scaling. Its CSV `speedup_tilelang` uses the saved `torch_ms` baseline divided by that direct time; the raw JSON instead records the new run's PyTorch timing and speedup. Separate runs may have different timing conditions. On GH200 all four backends were measured in the same run.
- **Read CSV columns by name, not by position.** The orders differ:
  - B200: the frozen 8 columns followed by `tilelang_ms`, `speedup_tilelang`.
  - GH200: the runner's 10-column order.
  - MI300X: 5 columns.
- **Line endings.** All GH200 and MI300X CSVs, and 56 of the 90 B200 CSVs, use CRLF.
- **One dtype label differs.** `results/B200/csv/kl_divergence_default.csv` labels its dtype `float32`; every other CSV uses `fp32`. Normalize the label on read and do not edit the CSV.
- **Peak metadata exists only for B200** (and Trainium2). Roofline and percentage-of-peak metrics are unavailable for GH200 and MI300X until a measured `<gpu>.json` is added.
- **Timing modes differ.** The final-measurement repeat count differs between campaigns, and MI300X is timed eagerly while the NVIDIA GPUs replay CUDA graphs. State both whenever latencies are compared across GPUs.

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
