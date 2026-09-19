# TileBench

<div align="center">
  <img src="assets/tilebench_icon.png" alt="TileBench icon" width="120" />
</div>

**Controlled performance evaluation and bottleneck diagnosis for tile-based programming models.**

TileBench is a modular accelerator benchmarking framework for comparing kernel implementations under standardized operator semantics, correctness checks, timing protocols, autotuning, and hardware-aware performance metrics.

It currently supports **PyTorch**, **Triton**, **NVIDIA cuTile**, **TileLang**, and **AWS Neuron NKI** backends.

![TileBench overview](assets/overview.png)

## Contents

- [Features](#features)
- [Results](#results)
- [Benchmark Suite](#benchmark-suite)
- [Backend Support](#backend-support)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Methodology](#evaluation-methodology)
- [LLM Kernel Generation](#llm-kernel-generation)
- [Project Structure](#project-structure)
- [Recorded Results](#recorded-results)
- [Developer Guide](#developer-guide)
- [Attribution](#attribution)

## Features

- **45 operator tasks** covering point-wise, reduction/normalization, matrix multiplication/attention, stencil/convolution, and data-layout workloads.
- **Shared correctness harness** using PyTorch references and dtype-aware verification.
- **Multiple backends**: PyTorch, Triton, cuTile, TileLang, and NKI.
- **Default and autotuned execution paths** for controlled configuration studies.
- **Proton-based timing** with warmup, CUDA graph replay, and optional L2 flushing.
- **Hardware-aware metrics** including latency, speedup, effective bandwidth, TFLOPS, arithmetic intensity, percentage of peak, and roofline utilization.
- **Profiling support** with Nsight Compute metadata and kernel-count validation.
- **Iterative LLM kernel generation** with correctness and performance feedback.

## Results

### Pairwise wins and losses

**45 operators · NVIDIA B200 · autotuned Triton and cuTile · complete valid dtype and input-size sweeps**

Each cell reports **wins / losses for the row backend against the column backend**. Each operator contributes one comparison based on the geometric mean of its valid per-case latency ratios.

| Row vs. column | Triton | cuTile | PyTorch |
|:---|:---:|:---:|:---:|
| **Triton** | N/A | **37 / 8** | **36 / 9** |
| **cuTile** | 8 / 37 | N/A | **33 / 12** |
| **PyTorch** | 9 / 36 | 12 / 33 | N/A |

Across the suite, Triton reaches a **2.02×** geometric-mean speedup over PyTorch, while cuTile reaches **1.58×**.

Triton and cuTile remain workload-dependent rather than uniformly ordered. Triton is stronger across many irregular, streaming, and bandwidth-bound operators, while cuTile is competitive on regular tiled workloads with strong reuse, including selected GEMM, attention, and stencil cases.

Autotuning improves both backends relative to their manually selected defaults, with geometric-mean gains of **1.18× for Triton** and **1.22× for cuTile** within the declared search spaces.

### Representative large-input latency

![PyTorch, Triton, and cuTile latency at representative large inputs](assets/sweep_max_latency.png)

The figure shows one sweep-max case per operator. FP16 is used when available; otherwise the first configured dtype is shown in the label. Triton and cuTile use their autotuned configurations.

The figure and the pairwise matrix summarize different case sets: the figure shows one representative large case per operator, while the matrix uses the complete valid dtype and input-size sweep.

Regenerate the figure with:

```bash
python scripts/plot_sweep_max.py
```

## Benchmark Suite

TileBench contains **45 operators**, with 26 task definitions derived from TritonBench and 19 from LeetGPU. Each operator provides a PyTorch reference, backend implementations, a case grid, benchmark controls, and metric formulas.

| Category | # Operators | Representative tasks |
|:---|---:|:---|
| Point-wise | 12 | vector addition, activations, weight dequantization |
| Reduction / Normalization | 11 | softmax, layer normalization, argmax, MoE gating |
| Matrix Multiplication / Attention | 8 | GEMM variants, dense attention, sparse attention |
| Stencil / Convolution | 6 | 1D/2D/3D convolution, pooling, Jacobi stencil |
| Data Layout | 8 | copy, transpose, indexing, sorting |
| **Total** | **45** | |

Operator definitions live under:

```text
tilebench/benchmarks/operators/
```

Each configured dtype is evaluated over the operator's declared input-size sweep.

## Backend Support

| Backend | Accelerator | Role | Results |
|:---|:---|:---|:---:|
| **PyTorch** | NVIDIA GPU / device-local baseline | Semantic reference and correctness baseline | Baseline |
| **Triton** | NVIDIA GPU | Tile-based kernel backend | Available |
| **NVIDIA cuTile** | NVIDIA GPU | Tile-based kernel backend | Available |
| **TileLang** | NVIDIA GPU | Optional tile-language backend | N/A |
| **AWS Neuron NKI** | AWS Trainium | Optional accelerator backend | N/A |

Backend coverage is operator-specific. Unsupported dtype/backend combinations are reported rather than silently treated as successful cases.

NKI runs on Trainium and is compared against a device-local PyTorch reference on the same platform.

## Installation

TileBench requires **Python 3.10+**.

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/Tilebench.git
cd Tilebench

python -m pip install -e .
```

The Python dependencies are declared in `requirements.txt`. NVIDIA CUDA and AWS Neuron runtime/toolchain setup remain platform-specific and must be installed for the backend being used.

The NKI backend requires a configured AWS Trainium/Neuron environment.

## Quick Start

### Run one operator

Run the fixed default configuration for Triton and cuTile:

```bash
python scripts/run_bench.py \
  --operator mul2 \
  --tile-language triton,cutile \
  --no-archive
```

Run the autotuned configuration:

```bash
python scripts/run_bench.py \
  --operator mul2 \
  --tile-language triton,cutile \
  --autotune \
  --no-archive
```

PyTorch runs automatically as the reference baseline.

Tracked summary CSVs are written to:

```text
results/csv/mul2_default.csv
results/csv/mul2_autotune.csv
```

Local timing logs, profiling outputs, figures, and other generated artifacts are ignored by Git.

### Run selected cases

```bash
python scripts/run_bench.py \
  --operator mul2 \
  --tile-language triton,cutile \
  --case-indices 0,1,2 \
  --no-archive
```

### Visualize results

```bash
python scripts/visualize.py \
  --operator mul2 \
  --gpu B200 \
  --metrics latency_ms bandwidth_GBs speedup pct_peak_bw roofline
```

Generated figures are local outputs and are not tracked.

### Run the full suite

```bash
python scripts/run_bench_all.py --help
```

Use the command options to select the desired backends and execution mode.

## Evaluation Methodology

### Correctness

PyTorch defines the reference output. Each backend is verified before timing using dtype-aware tolerances from `tilebench/core/verifier.py`, with operator-specific overrides where required by the numerical contract.

Integer outputs use exact comparison by default.

A backend that fails verification is excluded from timing for that case.

### Timing

The standard NVIDIA timing path uses:

- 20 warmup launches
- 100 timed launches
- Triton Proton
- CUDA graph replay
- optional L2 flushing outside the timed operator invocation

A multi-kernel implementation is timed as one complete operator `run()`, preserving intended producer-to-consumer reuse inside the operator boundary.

### Autotuning

The default path uses a fixed manually selected configuration.

The autotuned path searches a declared backend-specific configuration space and records the selected configuration. Triton and cuTile expose different tuning controls, so TileBench aligns search scope and granularity rather than forcing a one-to-one parameter mapping.

### Aggregation

For each valid case, TileBench computes latency ratios between implementations.

Operator-level values use the geometric mean over valid dtype and input-size cases. Suite-level speedups use the geometric mean over operator-level values, giving every operator equal weight.

### Hardware-aware metrics

Each operator declares analytical FLOP and byte formulas in `config.yaml`. These formulas are used consistently to derive:

- TFLOPS
- effective bandwidth
- arithmetic intensity
- percentage of peak
- roofline utilization

Canonical device metadata is stored under:

```text
tilebench/data/peak_performance/
```

Profiling metadata used by the NCU harness is stored under:

```text
tilebench/profiling/
├── ncu_catalogue.json
└── kernel_counts.json
```

Generated NCU reports are local artifacts and are not tracked.

## LLM Kernel Generation

TileBench also includes an iterative LLM kernel-generation workflow under:

```text
tilebench/llm_codegen/
```

The workflow combines operator descriptions, backend API references, framework constraints, PyTorch references, correctness feedback, and performance feedback across refinement iterations.

Generated trajectories and selected implementations are stored under:

```text
tilebench/benchmarks/llm_generated/
```

The generation workflow is separate from the manually implemented benchmark path. Generated implementations are evaluated under their own protocol and do not modify the manually maintained kernels.

## Project Structure

```text
Tilebench/
├── tilebench/
│   ├── core/                    # Benchmark engine, timing, verification, metrics, autotuning
│   ├── data/                    # Input generators and canonical device metadata
│   ├── benchmarks/
│   │   ├── operators/           # 45 operator definitions and backend implementations
│   │   └── llm_generated/       # Generated trajectories and selected kernels
│   ├── profiling/               # NCU harness, catalogue, kernel-count metadata
│   ├── llm_codegen/             # Iterative LLM generation and evaluation pipeline
│   └── paths.py                 # Repository/package resource resolution
├── scripts/                     # Benchmark, visualization, peak-measurement entry points
├── skills/                      # Backend API/programming guides used by generation
├── tests/                       # Test suite
├── results/
│   └── csv/                     # Tracked per-case benchmark summaries
├── assets/                      # README images
├── docs/                        # Extended documentation
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Recorded Results

Only benchmark CSVs under `results/csv/` are version-controlled.

```text
results/csv/
├── <operator>_default.csv
└── <operator>_autotune.csv
```

The repository contains 45 operators, with one default-mode and one autotuned CSV per operator.

Other outputs are generated locally and ignored, including timing logs, aggregate summaries, plots, detailed peak measurements, and NCU reports.

## Developer Guide

Implementation details, CLI options, operator-authoring rules, tuning conventions, dtype handling, and profiling internals are documented separately:

**[TileBench Developer Guide](docs/developer_guide.md)**

## Attribution

TileBench uses TritonBench and LeetGPU as sources of operator coverage and task semantics. TileBench implementations and configurations are maintained independently in this repository.

Third-party libraries, tools, and dependencies remain governed by their respective licenses and terms.
