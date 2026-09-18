# TileBench

**Controlled performance evaluation and bottleneck diagnosis for tile-based programming models.**

TileBench is a modular accelerator benchmarking framework for comparing kernel implementations under matched operator semantics and standardized evaluation. It combines verified implementations, default and autotuned configurations, hardware-aware performance metrics, profiling-guided diagnosis, and an iterative LLM kernel-generation track.

The accompanying **paper** evaluates **Triton and NVIDIA cuTile on a single NVIDIA B200**, with **PyTorch as the semantic reference, correctness oracle, and practical software baseline**. The repository also includes optional **TileLang** and **AWS Neuron NKI** extensions. Their results are not part of the paper summary below.

[Paper Results](#paper-results) · [Benchmark Overview](#benchmark-overview) · [Quick Start](#quick-start) · [Evaluation Methodology](#evaluation-methodology) · [LLM Track](#llm-kernel-generation-track) · [Developer Reference](#developer-reference) · [Citation](#citation)

## Paper Results

### Pairwise wins and losses

**45 operators · NVIDIA B200 · autotuned Triton and cuTile · complete dtype and input-size sweeps**

Each cell reports **wins / losses for the row backend against the column backend**. One operator contributes one comparison, determined by the geometric mean of the per-case latency ratios over its valid dtype and input-size cases. These are the operator-level results reported in Section 5.1 of the paper.

| Row vs. column | Triton | cuTile | PyTorch |
|:---|:---:|:---:|:---:|
| **Triton** | N/A | **37 / 8** | **36 / 9** |
| **cuTile** | 8 / 37 | N/A | **33 / 12** |
| **PyTorch** | 9 / 36 | 12 / 33 | N/A |

For example, Triton is faster than cuTile on 37 operators and slower on 8. A win uses a strict latency-ratio comparison, not a 5% margin or a statistical-significance test. The paper reports that Triton and cuTile are within 5% on 14 of the 45 operators.

Triton achieves a **2.02×** suite-level geometric-mean speedup over PyTorch, compared with **1.58×** for cuTile. Neither backend wins uniformly: Triton is stronger across many irregular, streaming, and bandwidth-bound operators, while cuTile is competitive on regular tiled workloads, including selected GEMM, attention, and stencil operators.

Relative to the manually selected defaults, autotuning provides geometric-mean gains of **1.18× for Triton** and **1.22× for cuTile** within the declared search spaces. These are configuration-search results, not global performance limits for either programming model.

### Latency at representative large inputs

![PyTorch, Triton, and cuTile latency at each operator's sweep-max case on NVIDIA B200](assets/sweep_max_latency.png)

**Lower is better.** The figure shows one sweep-max case per operator, selected from `tilebench_run/ncu_catalogue.json`. It uses FP16 when available, otherwise the operator's first configured dtype, shown in the label. Triton and cuTile use their autotuned configurations; PyTorch supplies the corresponding reference latency.

**The figure and matrix summarize different case sets.** The figure shows a representative large input and one dtype per operator. The matrix summarizes each operator's complete valid dtype and input-size sweep. The matrix must not be reconstructed by counting the shortest bars in this figure.

Regenerate the figure from the recorded CSV files:

```bash
PYTHONPATH=. python scripts/plot_sweep_max.py
```

## Benchmark Overview

![TileBench overview: benchmark construction, benchmark execution, and comparative analysis](assets/overview.png)

The paper's suite contains **45 operators**, with 26 task definitions derived from TritonBench and 19 from LeetGPU. TileBench supplies manually written and verified implementations, a PyTorch reference, and a per-operator configuration. Each supported dtype is evaluated over 20 input configurations.

| Category | Operators | Representative tasks |
|:---|---:|:---|
| Point-wise | 12 | Vector addition, activations, weight dequantization |
| Reduction / Normalization | 11 | Softmax, layer normalization, argmax, MoE gating |
| Matrix Multiplication / Attention | 8 | GEMM variants, dense and sparse attention |
| Stencil / Convolution | 6 | Convolution, pooling, Jacobi stencil |
| Data Layout | 8 | Copy, transpose, indexing, sorting |

The framework provides a shared correctness and timing harness, explicit default and autotuned execution paths, per-case FLOP and byte formulas, derived bandwidth and throughput metrics, roofline analysis, and profiling tools for selected performance gaps. The LLM track reuses the tasks and verification interface but evaluates generated code separately from the manually implemented benchmark.

### Backend scope

| Implementation | Role in the paper | Repository extension |
|:---|:---|:---|
| **PyTorch** | Semantic reference and software baseline on B200 | Separate device-local reference for NKI evaluation |
| **Triton** | Primary programming model evaluated on B200 | CUDA benchmark backend |
| **NVIDIA cuTile** | Primary programming model evaluated on B200 | CUDA benchmark backend |
| **TileLang** | Not evaluated in this paper | Optional CUDA implementation, where available |
| **AWS Neuron NKI** | Not evaluated in this paper | Optional AWS Trainium implementation, where available |

Optional-backend coverage is operator-specific. NKI uses a separate Trainium timing path and is compared against its own PyTorch reference on that device. Do not interpret a B200-versus-Trainium latency difference as an isolated programming-model effect.

## Software Environments

### Environment reported in the paper

| Component | Reported configuration |
|:---|:---|
| Accelerator | One NVIDIA B200, 180 GB HBM3e |
| PyTorch | 2.10 |
| CUDA | 13.0 |
| Triton | 3.6.0 |
| cuTile | `cuda-tile` 1.3.0 |
| Measured peak bandwidth used for roofline analysis | 6539.4 GB/s |

The headline results above are taken from the paper. The development environment below is different, particularly in its CUDA toolkit and cuTile versions. A run on the current development stack is a new measurement, not an exact reproduction of the reported paper environment.

<details>
<summary><strong>Recorded development environments</strong></summary>

The following environments were recorded during development. The NVIDIA and Trainium rows describe different hosts, not one installation.

| Component | Recorded version or configuration |
|:---|:---|
| NVIDIA GPU | B200, 180 GB; driver 595.58.03 |
| NVIDIA host OS | Red Hat Enterprise Linux 10.0; kernel 6.12 |
| CUDA toolkit | 13.2; `nvcc` V13.2.78 |
| Python | 3.10.19 |
| PyTorch | 2.10.0+cu130; CUDA 13.0 runtime |
| Triton | 3.6.0, including Proton |
| cuTile packages | `cuda-tile` 1.5.0; `cuda-bindings` 13.0.3; `cuda-tile-experimental` 0.0.1, installed from the `experimental/` directory of the NVIDIA `cutile-python` Git repository (it is not on PyPI, and no code in this repository imports it) |
| TileLang extension | `tilelang` 0.1.11; `apache-tvm-ffi` 0.1.11 |
| NKI extension, separate host | `nki` 0.6.0; `neuronx-cc` 2.27; AWS trn2.3xlarge, Trainium2 |
| Nsight Compute | 2026.1.1 |
| Supporting Python packages | NumPy 2.2.6; PyYAML 6.0.3; Matplotlib 3.10.8 |

The TileLang pins reflect an import failure reported for `apache-tvm-ffi==0.1.12` in this development environment. Install Neuron dependencies on the Trainium host separately; the CUDA setup below does not install the Neuron SDK.

</details>

## Quick Start

The following steps target the **NVIDIA CUDA development path**. They assume a configured NVIDIA driver and CUDA toolkit corresponding to the recorded development environment. Python package installation does not replace host driver or toolkit setup.

### 1. Clone the repository and create an environment

```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/Tilebench.git
cd Tilebench

conda create -n tilebench_env python=3.10 -y
conda activate tilebench_env

python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install \
  triton==3.6.0 cuda-tile==1.5.0 cuda-bindings==13.0.3 \
  numpy==2.2.6 PyYAML==6.0.3 matplotlib==3.10.8 tabulate
```

This is a package-level starting point based on the recorded development versions, not a complete environment lock or a verified installer for every host. The existing `requirements.txt` includes optional-backend dependencies and leaves several core packages unpinned; it should not be treated as the paper's reproducibility specification.

### 2. Run the paper's backend pair

**Result-file warning:** Triton/cuTile runs write to `results/csv/<operator>_{default,autotune}.csv`. A partial run with `--case-indices` can replace the corresponding full-sweep CSV. Use a separate checkout for experiments that must preserve the recorded results. Custom `--output` paths affect the timing JSON, not the CSV destination.

```bash
# Fixed default configuration; mul2/config.yaml sets autotune: false.
PYTHONPATH=. python scripts/run_bench.py \
  --operator mul2 --tile-language triton,cutile --no-archive \
  --output results/logs/time_measurement_logs/mul2_default.json \
  --autotune-log results/logs/autotune_logs/mul2_default.json

# Autotuned configurations; keep the timing JSON separate from the default run.
PYTHONPATH=. python scripts/run_bench.py \
  --operator mul2 --tile-language triton,cutile --autotune --no-archive \
  --output results/logs/time_measurement_logs/mul2_autotune.json \
  --autotune-log results/logs/autotune_logs/mul2_autotune.json
```

PyTorch runs automatically as the baseline. Explicitly selecting `triton,cutile` avoids invoking optional extension backends. Timing options come from the operator's `config.yaml`; the `mul2` configuration uses 20 warmup launches, 100 timed launches, CUDA graph replay, and L2 flushing.

The examples use `--no-archive` to avoid creating local raw-log archive commits. Timing JSON and CSV results are still written normally.

### 3. Visualize the results

```bash
PYTHONPATH=. python scripts/visualize.py \
  --operator mul2 \
  --input results/logs/time_measurement_logs/mul2_autotune.json \
  --gpu B200 \
  --metrics latency_ms bandwidth_GBs speedup pct_peak_bw roofline
```

Figures are written to `results/figures/mul2/`. The `--gpu B200` option loads the peak values from `data/peak_performance/B200.json` for roofline and percentage-of-peak metrics.

For other cases and options, see the [CLI reference](#cli-reference). To implement an operator, see the [operator authoring guide](#operator-authoring-guide).

## Evaluation Methodology

### Correctness and timing

PyTorch defines the reference output. The verifier uses dtype-aware tolerances with documented operator-specific overrides; integer outputs use exact comparison under the paper protocol. Unsupported cases and failed implementations must be distinguished from successful verification and reported with their coverage.

For the NVIDIA paper evaluation, each per-case latency is the mean over **100 timed launches after 20 warmup launches**, measured using **Triton Proton and CUDA graph replay**. L2 eviction occurs before the measured operator invocation, outside its timed scope. A multi-kernel implementation is measured as the complete operator `run()` sequence, preserving producer-to-consumer reuse within that boundary.

Autotuning selects a configuration before final timing. Nsight Compute and generated-code inspection are used for diagnosis, not as a replacement for the Proton latency measurements. The NKI extension has a separate Neuron timing and profiling path, described in the authoring guide below.

### Aggregation and interpretation

The matrix follows **Section 5.1 and Appendix C.1** of the paper:

1. For each valid case, compute the latency ratio between the compared implementations.
2. Take the geometric mean over the operator's valid dtype and input-size cases, using the same matched cases for both implementations.
3. Count one win or loss per operator. For suite-level speedups, take the geometric mean of the 45 operator-level results, giving each operator equal weight.

Use full-precision measurements for regeneration, especially near parity. The display-rounded CSV columns should not be treated as a substitute for the full-precision timing records without checking their effect on the result.

PyTorch is a practical software baseline, not a third matched tile-based implementation. Its path may use eager operations, generic kernels, compiled kernels, or vendor libraries. A speedup over PyTorch can therefore reflect fusion or specialization as well as backend code generation. The direct Triton/cuTile comparison uses the controlled custom implementations.

Bandwidth and roofline metrics are derived from the analytical `flops_expr` and `bytes_expr` in each operator's configuration. They are not measurements of physical DRAM traffic. The B200 peak table uses the TF32 Tensor Core ceiling for the FP32 dot/MMA path described in Appendix A.4; that entry is not a general FP32 scalar-compute peak.

## LLM Kernel-Generation Track

The separate LLM track evaluates Triton and cuTile generation under a **10-iteration budget**. Prompts combine task descriptions, PyTorch references, operator configurations, framework conventions, and backend API references. Iterations receive correctness and performance feedback; generated implementations may not invoke backend autotuners or delegate the operator's computation to the reference or vendor libraries.

The paper reports the following coverage and geometric-mean token efficiency:

| Model | Backend | At least one verify-clean result | Faster than PyTorch | Token efficiency, speedup per million tokens |
|:---|:---|:---:|:---:|---:|
| GPT-5.5 | Triton | 45/45 | 38/45 | 15.83 |
| GPT-5.5 | cuTile | 43/45 | 36/43 | 10.79 |
| Claude Opus 4.7 | Triton | 45/45 | 38/45 | 12.90 |
| Claude Opus 4.7 | cuTile | 41/45 | 33/41 | 7.61 |

**Denominators matter:** the cuTile faster-than-PyTorch counts are conditional on obtaining at least one verify-clean implementation. They are not full-suite correctness rates. The manual-track matrix and LLM-track results use different implementations and case-selection protocols and must not be combined.

The pipeline is under `tools/llm_codegen/`, backend guides are under `skills/`, and generated trajectories and final implementations are under `benchmarks/llm_generated/`. See Section 3.3, Section 5.4, Appendix B, and Table 6 of the paper for the experimental description.

## Results and Raw Logs

| Location | Contents |
|:---|:---|
| `results/csv/` | Recorded per-case summary CSVs, separated by default and autotuned mode |
| `results/aggregate/` | Per-operator summaries by dtype and mode; check the aggregation convention before using them for paper-level results |
| `results/figures/` | Per-operator plots |
| `results/logs/` | Raw timing and tuning records generated locally |
| `tilebench_run/` | Batch launchers, profiling tools, and the NCU case catalogue |

The documented raw-log archive branch is `archive/raw-logs-2026-09-18`; `results/logs/` is ignored on `main`. Without `--no-archive`, `run_bench.py` invokes the archive script to snapshot local raw logs into a local archive-branch commit. This is separate from publishing the archive:

```bash
# Explicitly publish local archive snapshots.
bash scripts/archive_logs.sh --push
```

A dated archive branch can still receive new commits. Record the exact code and result commit used for any reproduced result rather than relying on the branch name alone.

## Project Structure

```text
Tilebench/
├── benchmarks/
│   ├── operators/<name>/       # config.yaml, PyTorch reference, custom implementations
│   └── llm_generated/          # Generated trajectories and selected implementations
├── core/                      # Orchestration, timing, verification, metrics, tuning
├── data/
│   ├── tensors.py             # Input-generator registry
│   └── peak_performance/      # Device peak values used in derived metrics
├── scripts/                   # Benchmark, visualization, peak, and archive entry points
├── tools/llm_codegen/          # Iterative generation pipeline and task descriptions
├── skills/                    # Backend programming and API guides
├── tilebench_run/             # Profiling workflow and NCU catalogue
├── tests/                     # Tests, including the NKI profiling flow
├── results/                   # Recorded results and locally generated artifacts
├── assets/                    # README figures
└── requirements.txt           # Development dependencies; not a paper environment lock
```

## Developer Reference

### CLI Reference

<details>
<summary><strong>Expand benchmark and visualization options</strong></summary>

#### `run_bench.py`

| Argument | Default | Description |
|---|---|---|
| `--operator` | `vector_add` | Operator to benchmark |
| `--output` | `results/logs/time_measurement_logs/<op>_results.json` | Timing output |
| `--autotune-log` | `results/logs/autotune_logs/<op>_autotune.json` | Autotune config log |
| `--warmup` | from config | Warmup iterations |
| `--repeat` | from config | Measurement iterations |
| `--use-cuda-graph` | from config | Enable CUDA graph |
| `--flush-l2` | from config | Flush L2 before each iteration |
| `--autotune` | omitted | Enable the autotuned path and select the `_autotune.csv` filename. Without this flag, the script uses `_default.csv`; keep `benchmark.autotune: false` for default-mode runs. |
| `--tile-language` | all | Comma-separated backends to run: `triton`, `cutile`, `tilelang`, `nki` (PyTorch always runs as the baseline) |
| `--no-archive` | false | Disable the automatic local raw-log archive commit. This does not disable timing JSON or CSV output. |
| `--case-indices` | all | e.g. `0,1,3` to run subset |
| `--keep-proton-files` | false | Keep `.hatchet` files for inspection |
| `--proton-output-dir` | system temp | Directory for Proton files |

#### `visualize.py`

| Argument | Default | Description |
|---|---|---|
| `--operator` | *(required)* | Operator name; locates config.yaml and default paths |
| `--input` | `results/logs/time_measurement_logs/<op>_results.json` | Timing JSON |
| `--output-dir` | `results/figures/<op>/` | Output directory for PNGs |
| `--metrics` | from `config.yaml metrics.plots` | Metrics to plot |
| `--gpu` | none | GPU short name (e.g. `B200`); loads `data/peak_performance/<GPU>.json` for the roofline and % of peak metrics |

Available metrics: `latency_ms` · `bandwidth_GBs` · `pct_peak_bw` · `tflops` · `pct_peak_tflops` · `speedup` · `arithmetic_intensity` · `roofline`

**Output safety:** `--output` changes the timing JSON path, not the CSV destination. Triton/cuTile runs, including runs with `--case-indices`, write the selected rows to the operator CSV. Use a separate checkout for experiments that must not modify the recorded results.


</details>

### Operator Authoring Guide

<details>
<summary><strong>Expand implementation rules, tuning templates, dtype handling, and NKI profiling</strong></summary>

This guide describes the manually implemented benchmark track. Match operator semantics across every compared implementation, and keep the algorithms and implementation structures of the tile-based backends comparable. PyTorch defines the reference output and practical software baseline; its internal kernel decomposition and library dispatch need not match the custom kernels.

The goal is a controlled comparison, not an unrestricted search for the fastest possible implementation. Backend autotuning explores declared configuration spaces. It does not replace algorithm design or change the source-level algorithm. The separate LLM track does not permit backend autotuning.

The templates below are schematic. Start from a verified operator in `benchmarks/operators/` for complete, executable examples.

#### 1. Comparison Policy

- Use the same input/output semantics for PyTorch and all custom implementations. Keep the Triton and cuTile algorithms and code structures comparable, for example grouped tiling for GEMM or one-dimensional blocking for element-wise operations. Document differences in the PyTorch reference path rather than claiming identical internals.
- Provide both a fixed default configuration and an autotuned path. Triton uses `@triton.autotune`; cuTile uses `ct.tune.exhaustive_search` through `core.cutile_autotune.CutileAutotuner`. Define the search space explicitly, record its winner, and keep the tuning budget and granularity comparable.
- Keep custom-kernel structures conceptually aligned, including task decomposition, tiling strategy, loop organization, and boundary semantics. Use each backend's native primitives; document material structural differences.
- Avoid backend-specific tricks unless discussed and documented.
- TileLang (`impl_tilelang.py`) and NKI (`impl_nki.py`) are optional extensions. Declare their implemented operator and dtype coverage explicitly. Preserve the comparison contract and document hardware-specific differences. NKI is evaluated on a separate Trainium host against its own PyTorch baseline.

#### 2. Required Files per Operator

```
benchmarks/operators/<name>/
├── config.yaml        # case_grid, benchmark params, metrics expressions
├── impl_torch.py      # def run(*inputs) -> Tensor
├── impl_triton.py     # def run(*inputs, block_size=...) -> Tensor  + get_last_config()
├── impl_cutile.py     # def run(*inputs, block_size=...) -> Tensor  + get_last_config()
├── impl_tilelang.py   # optional TileLang implementation
└── impl_nki.py        # optional Trainium implementation
```

Register the input generator in `data/tensors.py`:
```python
def generate_<name>_inputs(n, dtype=torch.float32, device='cuda'):
    ...
GENERATORS["<name>"] = generate_<name>_inputs
```

#### 3. Minimal Coding Rules

- Same input/output semantics across every backend included in the comparison.
- Preserve the declared precision and dtype behavior. Do not add silent conversions or change the task to make an unsupported case pass. Record unsupported cases and failures explicitly; a skipped case is not a successful correctness check.
- Same shape assumptions; correct boundary/tail handling (not only power-of-two).
- Keep kernels clean; no dtype dispatch inside the kernel unless inherent to the algorithm.

#### 4. `config.yaml` Structure

```yaml
benchmark:
  warmup: 20            # warm-up iterations (outside Proton scope, for JIT + cache)
  repeat: 100           # timed iterations (inside proton.scope)
  use_cuda_graph: true  # recommended for all operators
  flush_l2: true        # recommended for memory-bound operators
  autotune: false       # use --autotune to select the tuned mode and CSV suffix

case_grid:
  # Use expr for a range of problem sizes; enables performance-vs-size curves
  n:
    expr: "[1024 * 1024 * i for i in range(1, 21)]"   # 20 sizes, as in mul2/config.yaml
  dtype: ["fp16", "bf16", "fp32", "int8"]
  # Declare the common dtype coverage for the intended comparison.
  # See §6 for FP8 / integer handling.

metrics:
  flops_expr: "<Python expression>"   # e.g. "n" for element-wise, "2*M*N*K" for matmul
  bytes_expr: "<Python expression>"   # bytes read+written, e.g. "n * dtype_size * 2"
  plots:
    - latency_ms
    - bandwidth_GBs
    - speedup
```

Device peaks are not set per operator. `pct_peak_bw`, `pct_peak_tflops` and `roofline` read them from `data/peak_performance/<GPU>.json`, selected with `visualize.py --gpu B200`. For B200 that file holds the measured peak bandwidth (6539.4 GB/s) and the per-dtype peak TFLOPS (fp16/bf16 2250, fp32 1100 via TF32, int8/fp8 4500).

##### Input size guidelines

The published suite uses 20 input configurations per supported dtype. The ranges below are authoring examples, not a replacement for the recorded per-operator grids.

| Operator type | Recommended `n` range |
|---|---|
| Element-wise (vector) | See the 20-size sweep in `mul2/config.yaml` |
| Reduction (softmax) | rows × cols; vary cols from 512 to 8192 |
| GEMM (matmul) | M=N=K from 256 to 4096, step 256 |
| Attention | `seq_len` from 512 to 8192 |

Always include at least one **non-power-of-two** size to test tail masking.

#### 5. Autotune

##### Triton

```python
_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}

@triton.jit
def my_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...

_my_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["n_elements"],   # re-tune when n_elements changes
)(my_kernel)

def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _my_kernel_autotuned[grid](x, out, n_elements)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        my_kernel[grid](x, out, n_elements,
                        BLOCK_SIZE=cfg["BLOCK_SIZE"], num_warps=cfg["num_warps"])
    return out

def get_last_config() -> dict | None:
    cfg = getattr(_my_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
```

- `get_last_config()` reads `best_config` straight off the autotuned kernel. Do not keep a module-level `_last_config`, and never use `global` in `run()`.
- The autotune key should be whatever drives the optimal config change. For element-wise ops: `["n_elements"]`. For matmul: `["M", "N", "K"]`.
- Distinguish compiled-kernel caching from autotuning-result caching. Do not assume that the winning configuration persists across processes merely because compiled kernels are cached. Record the cache policy used for the experiment.

##### cuTile

```python
from types import SimpleNamespace
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

_last_autotune_config: dict = {}          # mutable-dict pattern; never `global`

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]

@ct.kernel
def my_kernel(x_ptr, out_ptr, TILE: ct.Constant[int]):
    ...

_tuner = CutileAutotuner(my_kernel)

def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    stream = torch.cuda.current_stream()
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, str(x.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((n + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (x, out, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"tile": cfg.tile, "occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, ((n + cfg.tile - 1) // cfg.tile, 1, 1), kernel, (x, out, cfg.tile))
    return out

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```

- `ct.tune.exhaustive_search` only tunes; it does not launch and has no kernel-level cache. `CutileAutotuner.tune_or_cached` runs it once per `shape_key` and caches the winner, so repeated `run(autotune=True)` calls inside the timing loop do not repeat the sweep.
- `kernel_with_hints` memoises `replace_hints`, so the tuned hints are actually applied at launch and CUDA-graph capture sees a stable kernel object.
- Both caches are in-memory only; tuning re-runs in every new process.
- The keys returned by `get_last_config()` must match the names in `_DEFAULT_CONFIG`. The NCU harness replays winners by those names, and a mismatch makes it silently profile the defaults.
- The selected configs of every backend are printed during the run and saved to `results/logs/autotune_logs/<op>_autotune.json`.

##### TileLang (optional)

The recorded development environment uses `tilelang==0.1.11` with `apache-tvm-ffi==0.1.11`. The repository pins both versions because it reports a C++-level import failure with `apache-tvm-ffi==0.1.12` in that environment.

```python
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_last_autotune_config: dict = {}          # mutable-dict pattern; never `global`

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}

@tilelang.autotune(
    configs=[dict(BLOCK_SIZE=bs, threads=nt)
             for bs in [512, 1024, 2048] for nt in [64, 128, 256]],
)
@tilelang.jit
def my_kernel(x, out, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    ...

def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs):
    dtype = str(x.dtype).removeprefix("torch.")
    out = torch.empty_like(x)
    if autotune:
        with set_autotune_inputs(x, out):
            kernel = my_kernel.compile(x, out, dtype=dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config))
        kernel(x, out)
    else:
        cfg = _DEFAULT_CONFIG
        # Passing every tunable param explicitly bypasses the sweep
        # (tilelang logs "Skipping compilation and using direct JIT").
        my_kernel(x, out, dtype=dtype, BLOCK_SIZE=cfg["BLOCK_SIZE"], threads=cfg["threads"])
    return out

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```

- Cache and replay the selected configuration within the process. Repeated timed calls must not repeat the candidate search.
- The existing TileLang authoring pattern uses `T.symbolic` for swept problem-size dimensions to avoid unnecessary per-shape specialization. Use compile-time constants where required by the kernel, and verify the specialization behavior of the installed version.
- CUDA-graph capture of the compiled kernel works; keep `use_cuda_graph: true`.

##### NKI (optional; AWS Trainium)

NKI autotuning used by TileBench MUST go through `core.nki_autotune.NkiAutotuner`
(or an explicit replay adapter with the same trace semantics). Manual autotuners
without replay semantics are unsupported; the profiling flow cannot replay them
and will profile the operator as untuned (`autotune=False`) when no NKI winner
records exist; never fabricate a winner.

```python
from types import SimpleNamespace
from core.nki_autotune import NkiAutotuner

if nki is not None:
    @nki.jit
    def my_kernel(a_input, TILE_FREE): ...
    _tuner = NkiAutotuner(my_kernel)          # stable name: <module>.<func>
    # NkiAutotuner(my_kernel, name="...")     # override on collisions

_DEFAULT_CONFIG = SimpleNamespace(tile_free=2048)
_SEARCH_SPACE = [SimpleNamespace(tile_free=t) for t in (512, 1024, 2048, 4096)]
_last_autotune_config: dict = {}
```

- **Configs and shape keys must be deterministically serializable**: dicts,
  dataclasses, `SimpleNamespace`, namedtuples, or plain objects with stable
  public fields, holding only `None`/bool/int/float/str/tuple/list/dict values.
  Anything else raises at tune time; no `repr()` fallback.
- **Profiling flow** (`core/nki_orchestrator.py`): a SELECTOR process runs the
  sweep and exports the canonical winner trace; a fresh PROFILE process installs
  that trace and **replays the winner exactly**; zero candidate timings; a
  replay mismatch (changed search space, stale winner) fails loudly before
  anything is profiled.
- **Artifact identity** is a canonical launch spec (`spec_id`): the profile
  worker runs in a private per-spec CWD with a private Neuron compile cache and
  records EVERY NEFF/HLO pair each phase (torch baseline, NKI) compiled; an
  operator may compile several graphs per `run()` (one per radix-sort pass).
  The `AwsNeuronCustomNativeKernel` HLO marker validates *that a graph is an
  NKI graph*; it does NOT distinguish two autotune candidates of the same
  kernel; that distinction comes only from exact winner replay.
- **Timing** comes from the Neuron runtime's inspect trace
  (`NEURON_RT_INSPECT_*`, private output dir per spec): the worker runs
  `run()` `warmup` + `repeat` times on the case's real inputs and records, per
  timed iteration, the range of XLA execution indices it covered; the parent
  orders the trace's executions, checks their count against the worker's,
  sums the device time inside each range (multi-graph `run()`s are summed;
  no host clock is ever compared with the trace's timebase) and matches
  every executed NEFF (written back by the runtime, byte-identical to the
  compiler dump) to a recorded pair by SHA256. `NKI(ms)` is the mean over the
  timed iterations. The per-iteration graph pattern must be identical, NKI
  iterations must execute a marker-bearing graph and the torch baseline none.
- **Artifact selection must never depend on mtime**, sequence numbers, glob
  order, or "most recent compile event". An execution that cannot be attributed
  to this run's artifacts, a varying graph pattern, or a missing kernel graph is
  a benchmark failure (`nki_ok=False`), never a guess. Each profiled case
  writes `results/logs/nki_profiles/<op>/<case>/<spec>/manifest.json`
  (per-target `artifacts` with paths + SHA256s, `executed` graphs with
  per-iteration counts, `stats.per_iteration_ms`) and a line in
  `results/logs/nki_neff_manifest.jsonl`; the runtime session (`inspect/`:
  executed NEFFs, one `.ntff` per NEFF, `ntrace.pb`) is kept for a
  `neuron-explorer` deep-dive.

#### 6. Dtype Handling

##### Standard floating-point types (fp16, bf16, fp32)

Support is operator-specific. Declare the dtypes in `config.yaml`, and verify each configured backend on every supported shape and dtype rather than inferring support from the dtype name alone.

##### Integer types (int8, int16, int32)

- Generate inputs with bounded values to avoid overflow after the operation.
- Example for `int8` with `x * 2`: use `torch.randint(-32, 33, ...)` so result stays in `[-128, 127]`.
- `torch.testing.assert_close` uses exact comparison (`atol=0, rtol=0`) for integer types; this is intentional.

```python
# data/tensors.py
def generate_<name>_inputs(n, dtype=torch.float32, device='cuda'):
    if dtype == torch.int8:
        x = torch.randint(-32, 33, (n,), device=device).to(torch.int8)
    else:
        x = torch.randn(n, dtype=dtype, device=device)
    return (x,)
```

##### FP8 (`fp8_e4m3fn`, `fp8_e5m2`)

FP8 support must be established for the specific operator, implementation, and software stack. Do not infer general arithmetic support from the presence of an FP8 dtype or an FP8 GEMM path.

Native FP8 `mul2` tests failed with PyTorch 2.10, Triton 3.6.0, and cuda-tile 1.5.0 on B200. Treat this as a result for those tested paths, not a blanket claim that all FP8 operations are unsupported. The paper includes FP8 GEMM cases.

Do not introduce a silent precision workaround to force a case to run. For a new operator, declare FP8 only after confirming compatible reference and backend paths, input generation, accumulation behavior, and output verification. Record unsupported configurations explicitly.

```yaml
dtype: ["fp16", "bf16", "fp32", "int8"]
# Add fp8_e4m3fn or fp8_e5m2 only after validating the operator's FP8 path.
```

#### 7. Correctness Check

- The framework runs `core/verifier.py` automatically; no custom correctness script needed.
- Default tolerances are **dtype-aware**, as defined in `core/verifier.py`. Operators may provide documented `verify.atol` and `verify.rtol` overrides in `config.yaml`; these are part of the task definition, not settings to relax after a failure. The common defaults are:

| dtype | atol | rtol |
|---|---|---|
| float32 | 1e-5 | 1.3e-6 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1.6e-2 |
| int8 / int16 / int32 / int64 | 0 | 0 (exact) |

- If a backend fails correctness, it is excluded from timing but the run continues.
- Do **not** weaken tolerances to make a broken kernel pass. Fix the implementation; any justified change to the task's numerical contract requires explicit review.

#### 8. Benchmark Settings

| Setting | Recommendation | Reason |
|---|---|---|
| `use_cuda_graph: true` | All operators | Eliminates CPU launch overhead for steady-state measurement |
| `flush_l2: true` | Memory-bound operators (element-wise, attention) | Prevents artificially fast L2-cache-hot results |
| `warmup: 20` | Paper protocol | Warmup launches before timed measurement |
| `repeat: 100` | Paper protocol | Mean over 100 timed launches; disclose changes in exploratory runs |

#### 9. Metrics Configuration

Add a `metrics` section to `config.yaml` so `visualize.py` can compute derived metrics:

```yaml
metrics:
  # Variables available in expressions: n, dtype_size (bytes/element)
  flops_expr: "n"                   # mul2, relu, vector_add: 1 FLOP/element
  bytes_expr: "n * dtype_size * 2"  # 1 read + 1 write

  # For matmul:
  # flops_expr: "2 * M * N * K"
  # bytes_expr: "(M*K + K*N + M*N) * dtype_size"
```

If `flops_expr` is `null` or omitted, TFLOPS and `pct_peak_tflops` are not computed (safe to omit for attention where FLOPs are complex to define).

#### 10. Pre-Merge Checklist

- [ ] Coverage, unsupported cases, and failures are explicitly recorded for all configured backends
- [ ] Correctness check passes for all configured dtypes
- [ ] Autotune is set up for both Triton and cuTile; `get_last_config()` is implemented
- [ ] `config.yaml` has a `metrics` section with `flops_expr`, `bytes_expr`, and `plots`
- [ ] Input sizes cover at least one small, one medium, one large, and one non-power-of-two case
- [ ] Custom backends use comparable algorithms and implementation structures; the PyTorch reference path is documented
- [ ] No dtype-specific workarounds inside kernel code
- [ ] FP8 is included only where the operator's reference and compared backends support the declared precision path
- [ ] If `impl_tilelang.py` is provided: same algorithmic strategy, mutable-dict `get_last_config()`, default path passes explicit config kwargs (skips the tuning sweep), swept size dims use `T.symbolic` rather than `T.const`
- [ ] NKI autotuning, when provided, follows the winner-trace and exact-replay contract above
- [ ] Main-track autotuning and LLM-track iterative generation remain separate


</details>

## Attribution

TileBench uses TritonBench and LeetGPU as sources of operator coverage and task semantics. The paper describes the independently written TileBench implementations and third-party attribution in Appendix G. Dependencies and third-party artifacts retain their own notices and terms.

<!-- Release action: add the authors' chosen project LICENSE and complete attribution notices. Do not infer a project license from a dependency's license. -->

## Citation

**TileBench: A Controlled Benchmark for Performance Evaluation and Bottleneck Diagnosis of Tile-Based Programming Models**<br>
Bowen Cui, Zhongchun Zhou, Hao Wu, Tejas Ramesh, Junyu Yin, Jialiang Gu, and Keren Zhou.

The official proceedings BibTeX and paper link will replace the entry below once they are available.

```bibtex
@misc{cui2026tilebench,
  title = {{TileBench}: A Controlled Benchmark for Performance Evaluation and Bottleneck Diagnosis of Tile-Based Programming Models},
  author = {Cui, Bowen and Zhou, Zhongchun and Wu, Hao and Ramesh, Tejas and Yin, Junyu and Gu, Jialiang and Zhou, Keren},
  year = {2026}
}
```
