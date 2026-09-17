# TileBench

A modular GPU performance benchmarking framework for comparing **NVIDIA cuTile (CUDA 13.1)**, **Triton**, **TileLang**, and **PyTorch** kernel implementations.

## Features

- **Multi-backend**: PyTorch · Triton · cuTile (CUDA 13.2 / Blackwell) · TileLang (optional — skipped gracefully when `impl_tilelang.py` or the `tilelang` package is absent)
- **Proton timing**: Mean latency via Triton Proton `data="tree"`, with optional CUDA graph replay
- **Autotune**: `@triton.autotune` for Triton; `ct_experimental.autotune_launch` for cuTile; `@tilelang.autotune` for TileLang — runs before timing, results logged separately
- **Correctness checks**: dtype-aware tolerance (`torch.testing.assert_close`) against PyTorch reference; unsupported dtypes skipped gracefully
- **Flexible case generation**: `case_grid` with `expr` syntax (Python expressions), `test_cases`, or `case_preset` in `config.yaml`
- **Derived metrics**: bandwidth (GB/s), % peak BW, TFLOPS, % peak TFLOPS, arithmetic intensity, speedup — computed via per-operator expressions in `config.yaml`
- **Visualization**: latency · bandwidth · speedup · % peak BW · Roofline plots per dtype

---

## Quick Start

### 1. Activate environment
```bash
conda activate tilebench_env
cd Tilebench
```

### 2. Run benchmark
```bash
# All cases for mul2 (outputs operator-named JSON automatically)
PYTHONPATH=. python scripts/run_bench.py --operator mul2

# Only specific cases
PYTHONPATH=. python scripts/run_bench.py --operator mul2 --case-indices 0,1,2
```

Output files (operator-bound defaults):
```
results/logs/time_measurement_logs/mul2_results.json
results/logs/autotune_logs/mul2_autotune.json
```

### 3. Visualize results
```bash
# Plots all metrics defined in config.yaml → results/figures/mul2/
PYTHONPATH=. python scripts/visualize.py --operator mul2

# Override metrics or paths
PYTHONPATH=. python scripts/visualize.py --operator mul2 \
    --metrics latency_ms bandwidth_GBs speedup pct_peak_bw roofline
```

Output: `results/figures/<operator>/<operator>_<metric>.png`

### 4. Author a new operator
- Follow `OPERATOR_AUTHORING_GUIDE.md`
- Copy from `benchmarks/operators/_template/`
- Register input generator in `data/tensors.py`

---

## Project Structure

```
Tilebench/
├── core/
│   ├── engine.py        # Benchmark orchestration: iterate cases, verify, time all backends
│   ├── timer.py         # Proton-based GPU timing (warmup outside, repeat inside scope)
│   ├── metrics.py       # Compute derived metrics (bandwidth, TFLOPS, speedup, …)
│   ├── verifier.py      # dtype-aware correctness check via torch.testing.assert_close
│   └── dtypes.py        # resolve_dtype() string→torch.dtype; dtype_size() bytes/element
│
├── data/
│   └── tensors.py       # Input generators per operator; expand_cases() for case_grid
│
├── scripts/
│   ├── run_bench.py     # CLI entry point for benchmarking
│   ├── visualize.py     # CLI entry point for plotting derived metrics + Roofline
│   └── parse_proton_trace.py  # Utility to inspect raw Proton hatchet files
│
├── benchmarks/
│   └── operators/
│       ├── _template/          # Scaffold for new operators
│       ├── mul2/               # x * 2  (element-wise, memory-bound)
│       ├── vector_add/         # a + b
│       ├── sin/                # sin(x)
│       ├── relu/               # relu(x)
│       ├── softmax/            # softmax(x)
│       ├── rope/               # RoPE positional encoding
│       ├── destindex/          # KV-cache scatter write
│       ├── flash_attention/    # Flash Attention
│       └── flash_decode/       # Flash Decode stage 2
│
├── results/
│   ├── logs/
│   │   ├── time_measurement_logs/   # <operator>_results.json
│   │   └── autotune_logs/           # <operator>_autotune.json
│   └── figures/
│       └── <operator>/              # PNG plots per metric
│
├── OPERATOR_AUTHORING_GUIDE.md
└── requirements.txt
```

---

## Operator Config (`config.yaml`)

Each operator defines its own `config.yaml`:

```yaml
benchmark:
  warmup: 20            # iterations outside Proton scope
  repeat: 100           # iterations inside proton.scope("launch")
  use_cuda_graph: true  # capture + replay for stable steady-state latency
  flush_l2: true        # flush L2 cache before each iteration (memory-bound fairness)

case_grid:
  n:
    expr: "[1024 * 1024 * i for i in range(1, 17)]"  # Python expression
  dtype: ["fp16", "bf16", "fp32", "int8"]

metrics:
  flops_expr: "n"                    # FLOPs per invocation (Python expr, vars: n, dtype_size)
  bytes_expr: "n * dtype_size * 2"   # bytes read+written
  peak_bw_GBs: 8000.0                # B200 HBM3e peak
  peak_tflops:
    fp16: 400.0
    bf16: 400.0
    fp32:  80.0
    int8: 800.0
  plots:
    - latency_ms
    - bandwidth_GBs
    - speedup
    - pct_peak_bw
    - roofline
```

---

## Timing Methodology

| Step | Details |
|---|---|
| **Warmup** | `warmup` iterations outside Proton session — JIT compile + cache warm |
| **Measurement** | `repeat` iterations each wrapped in `proton.scope("launch")` |
| **CUDA graph** | Graph captured inside Proton session; replayed each iteration |
| **L2 flush** | 64 MB write outside the scope — overhead excluded from timing |
| **Result** | `mean_ms = total_gpu_time_ns / repeat / 1e6` parsed from hatchet tree |

---

## CLI Reference

### `run_bench.py`

| Argument | Default | Description |
|---|---|---|
| `--operator` | `vector_add` | Operator to benchmark |
| `--output` | `results/logs/time_measurement_logs/<op>_results.json` | Timing output |
| `--autotune-log` | `results/logs/autotune_logs/<op>_autotune.json` | Autotune config log |
| `--warmup` | from config | Warmup iterations |
| `--repeat` | from config | Measurement iterations |
| `--use-cuda-graph` | from config | Enable CUDA graph |
| `--flush-l2` | from config | Flush L2 before each iteration |
| `--case-indices` | all | e.g. `0,1,3` to run subset |
| `--keep-proton-files` | false | Keep `.hatchet` files for inspection |
| `--proton-output-dir` | system temp | Directory for Proton files |

### `visualize.py`

| Argument | Default | Description |
|---|---|---|
| `--operator` | *(required)* | Operator name; locates config.yaml and default paths |
| `--input` | `results/logs/time_measurement_logs/<op>_results.json` | Timing JSON |
| `--output-dir` | `results/figures/<op>/` | Output directory for PNGs |
| `--metrics` | from `config.yaml metrics.plots` | Metrics to plot |

Available metrics: `latency_ms` · `bandwidth_GBs` · `pct_peak_bw` · `tflops` · `pct_peak_tflops` · `speedup` · `arithmetic_intensity` · `roofline`

---

## Autotune

- **Triton**: `@triton.autotune` with `key=["n_elements"]`. Results cached persistently in `~/.triton/cache/`; re-used across runs for the same problem size.
- **cuTile**: `ct_experimental.autotune_launch` with in-memory cache. Re-runs on every new process invocation.
- **TileLang**: `@tilelang.autotune` + `set_autotune_inputs`; tuning result cached in-process (repeat calls are cheap); selected config exposed via `kernel.config`.
- Both selected configs are printed during the run and saved to `<op>_autotune.json`.

---

## Adding a New Operator

```
benchmarks/operators/<name>/
├── config.yaml      # case_grid, benchmark params, metrics expressions
├── impl_torch.py    # def run(*inputs) -> Tensor
├── impl_triton.py   # def run(*inputs) -> Tensor  (+get_last_config for autotune)
├── impl_cutile.py   # def run(*inputs) -> Tensor  (+get_last_config for autotune)
└── impl_tilelang.py # optional — def run(*inputs) -> Tensor  (+get_last_config)
```

Then register in `data/tensors.py`:
```python
def generate_<name>_inputs(n, dtype, device='cuda'):
    ...

GENERATORS["<name>"] = generate_<name>_inputs
```

See `OPERATOR_AUTHORING_GUIDE.md` for full details.
