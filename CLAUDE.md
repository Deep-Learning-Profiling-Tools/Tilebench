# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TileBench** is a GPU performance benchmarking framework comparing three backend implementations for the same operators:
- **PyTorch** — reference implementation
- **Triton** — high-performance kernel compiler
- **cuTile** — NVIDIA CUDA Tile experimental API (targeting Blackwell/B200 GPUs)

Target hardware: NVIDIA B200 with 8000 GB/s HBM3e bandwidth.

## Common Commands

All commands require `PYTHONPATH=.` from the repo root:

```bash
# Run benchmarks for a single operator
PYTHONPATH=. python scripts/run_bench.py --operator mul2

# Run with specific cases and options
PYTHONPATH=. python scripts/run_bench.py --operator mul2 --case-indices 0,1,2 \
  --warmup 30 --repeat 200 --use-cuda-graph --flush-l2

# Run all operators sequentially
PYTHONPATH=. python scripts/run_bench_all.py

# Visualize results (reads results/logs/..., writes results/figures/...)
PYTHONPATH=. python scripts/visualize.py --operator mul2

# Run tests
PYTHONPATH=. pytest tests/test_metrics.py -v

# Run tests for specific operators
PYTHONPATH=. pytest tests/test_metrics.py -v --operator mul2 --operator destindex
```

## Architecture

### Core Pipeline

```
scripts/run_bench.py (CLI)
  → core/engine.py::run_benchmark_suite()
      ├─ Loads benchmarks/operators/<op>/config.yaml
      ├─ Imports impl_torch, impl_triton, impl_cutile dynamically
      ├─ Fetches input generators from data/tensors.py::GENERATORS
      ├─ Expands test cases (via case_grid / case_preset / test_cases)
      └─ For each case: generate inputs → verify → time → collect autotune config
  → Writes JSON to results/logs/time_measurement_logs/<op>_results.json
  → Writes JSON to results/logs/autotune_logs/<op>_autotune.json
```

### Key Modules

- **`core/engine.py`** — Orchestrates the full benchmark loop; handles case expansion, backend dispatch, and result aggregation
- **`core/timer.py`** — GPU timing via `triton.profiler` (Proton); wraps each repeat in `proton.scope("launch")`, parses hatchet tree for mean latency
- **`core/metrics.py`** — Expression-based metric computation (bandwidth, TFLOPS, Roofline); expressions can reference `n`, `dtype_size`, `math`, and **all numeric params from the case** (e.g. `m`, `k`, `batch`, `H`, `in_channels`, etc.)
- **`core/verifier.py`** — Correctness checking with per-dtype tolerances (fp32: 1e-5/1.3e-6, fp16: 1e-3/1e-3, bf16: 1e-2/1.6e-2, int8: exact)
- **`core/dtypes.py`** — Dtype name resolution (e.g. `"fp16"` → `torch.float16`) and byte sizes
- **`data/tensors.py`** — Input generator registry (`GENERATORS` dict); maps operator names to functions that produce input tensors

### Operator Structure

Each operator lives in `benchmarks/operators/<name>/` and contains:

```
config.yaml      # benchmark params, case generation, metric expressions
impl_torch.py    # def run(*inputs) -> Tensor
impl_triton.py   # def run(*inputs, ...) -> Tensor  +  def get_last_config()
impl_cutile.py   # def run(*inputs, ...) -> Tensor  +  def get_last_config()
```

Use `benchmarks/operators/_template/` as a scaffold when creating new operators. Also see `OPERATOR_AUTHORING_GUIDE.md`.

### Case Generation (config.yaml)

Priority: `test_cases` > `case_grid` > `case_preset`

`case_grid` supports Python expressions and an optional `case_defaults` for fixed parameters:
```yaml
case_defaults:        # merged into every case; use for fixed architectural params
  batch: 1
  M: 2048

case_grid:
  K:
    expr: "[512*i for i in range(1, 21)]"
  dtype: ["fp16", "bf16", "fp32"]
```

`case_defaults` values are consumed by the input generator in `data/tensors.py`; `impl_*.py` files only receive the generated tensors, not the raw params. See `benchmarks/operators/BENCHMARK_PARAMS.md` for a summary of fixed vs. swept parameters per operator.

### Autotune conventions

- **Triton**: `triton.autotune` caches `best_config` on the kernel object. `get_last_config()` reads it directly — no module-level state needed.
- **cuTile**: `ct_experimental.autotune_launch()` returns a one-shot result with no kernel-level cache. A module-level `_last_autotune_config` variable is required to persist the result for `get_last_config()`.
- Both backends expose `get_last_config() -> dict | None` for the engine to log selected configs.
- If a cuTile operator cannot be implemented (e.g. dynamic scatter index not supported), `run()` raises `NotImplementedError` and `get_last_config()` returns `None`. The engine's `try/except` handles this gracefully.

### Timing Methodology

1. **Warmup** runs outside Proton (JIT compilation + cache warming)
2. **Measurement** runs inside `proton.scope("launch")` for `repeat` iterations
3. Optional **CUDA graph capture** for stable steady-state latency
4. Optional **L2 cache flush** (64 MB write) outside measured scope
5. Mean latency = total_gpu_time_ns / repeat, parsed from hatchet tree

### Adding a New Operator

1. Copy `benchmarks/operators/_template/` to `benchmarks/operators/<new_op>/`
2. Implement `impl_torch.py`, `impl_triton.py`, `impl_cutile.py` (each exporting `run()`)
3. Add input generator to `data/tensors.py::GENERATORS`
4. Configure `config.yaml` with case grid and metric expressions (`bytes_expr`, `flops_expr`, `peak_bw_GBs`, `peak_tflops`)
5. Add entry to `benchmarks/operators/BENCHMARK_PARAMS.md` documenting fixed vs. swept parameters

#### impl_triton.py conventions
- `get_last_config()` reads directly from `_xxx_kernel_autotuned.best_config` — no module-level `_last_config` global needed
- `run()` must not contain `global _last_config`

#### impl_cutile.py conventions
- `ct.store()` only accepts static indices; use `ct.scatter()` for runtime-computed destination indices
- `ct.load/store` tile shape dimensions must be powers of two; handle non-power-of-2 problem dimensions with a tiled loop + `padding_mode=ct.PaddingMode.ZERO` (see rmsnorm for reference)
- Use `padding_mode=ct.PaddingMode.NEG_INF` when OOB elements must not affect a max/argmax reduction (see argmax)
- `autotune_launch()` has no kernel-level cache; store its result in a module-level `_last_autotune_config` variable, written only in the autotune path
- If an operator genuinely cannot be implemented in the current cuTile DSL, raise `NotImplementedError` with a clear explanation — the engine's `try/except` handles it gracefully

#### config.yaml guidelines
- `bytes_expr` and `flops_expr` must reflect the operator's actual memory traffic and compute
- Sweep enough data to exceed B200 L2 cache (~50 MB) so benchmarks measure HBM bandwidth, not cache performance
- `peak_tflops` entries must match the dtypes listed in `case_grid`
- Add a `verify:` section with loose `atol`/`rtol` for operators where accumulation error makes tight tolerances impractical (e.g. GEMM with TF32):
  ```yaml
  verify:
    atol: 1.0
    rtol: 1e-2
  ```
  When omitted, the verifier falls back to per-dtype defaults (fp32: 1e-5/1.3e-6, fp16: 1e-2/1e-2, bf16: 1e-2/1.6e-2).
