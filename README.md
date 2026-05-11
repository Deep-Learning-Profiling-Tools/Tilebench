# TileBench

A modular GPU performance benchmarking framework for comparing **NVIDIA cuTile (CUDA 13.1)**, **Triton**, and **PyTorch** kernel implementations.

## Features

- **Multi-backend**: PyTorch · Triton · cuTile (CUDA 13.1 / Blackwell)
- **Proton timing**: Mean latency via Triton Proton `data="tree"`, with optional CUDA graph replay
- **Autotune**: `@triton.autotune` for Triton; `ct_experimental.autotune_launch` for cuTile — runs before timing, results logged separately
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
- Both selected configs are printed during the run and saved to `<op>_autotune.json`.

---

## Adding a New Operator

```
benchmarks/operators/<name>/
├── config.yaml      # case_grid, benchmark params, metrics expressions
├── impl_torch.py    # def run(*inputs) -> Tensor
├── impl_triton.py   # def run(*inputs) -> Tensor  (+get_last_config for autotune)
└── impl_cutile.py   # def run(*inputs) -> Tensor  (+get_last_config for autotune)
```

Then register in `data/tensors.py`:
```python
def generate_<name>_inputs(n, dtype, device='cuda'):
    ...

GENERATORS["<name>"] = generate_<name>_inputs
```

See `OPERATOR_AUTHORING_GUIDE.md` for full details.

---

## LLM Kernel Generation (TileBench-LLMGen)

`llm_kernelgen/` is an extension that evaluates how well Large Language Models
can generate correct, performant Triton and cuTile kernels **without access to
any existing hand-written implementation**.

### Design principles

| Principle | How it is enforced |
|---|---|
| **No leakage** | `impl_triton.py` / `impl_cutile.py` are never included in prompts |
| **Reproducible** | Every prompt, raw response, error log, and repair round is archived |
| **Quantifiable** | compile rate, correctness pass rate, repair success, token cost, LOC |
| **Replaceable models** | OpenAI Responses API + any OpenAI-compatible `/chat/completions` endpoint |
| **Integrated** | Generated kernels run through the identical TileBench correctness + timing suite |

---

### Prerequisites

```bash
pip install openai jinja2 pyyaml
```

Set API key(s):
```bash
export OPENAI_API_KEY="sk-..."           # For OpenAI Responses API
export OPENAI_COMPAT_API_KEY="..."       # For any compatible endpoint
export OPENAI_COMPAT_BASE_URL="https://..." # Base URL for compatible endpoint
```

---

### Quick start

#### Step 1 — Generate one kernel (smoke test)

```bash
# Generate a Triton kernel for softmax, zero-shot, gpt4o model
PYTHONPATH=. python llm_kernelgen/scripts/generate.py \
    --operator softmax \
    --backend triton \
    --model gpt4o \
    --experiment exp_smoke \
    --samples 1
```

Output directory: `llm_kernelgen/generated/exp_smoke/softmax/triton/sample_00/`

```
sample_00/
├── prompt.md           # Full prompt sent to the LLM
├── response.raw.txt    # Raw LLM response
├── impl_triton.py      # Extracted code
└── metadata.json       # Model, tokens, latency, git commit, context audit
```

#### Step 2 — Evaluate the generated kernel

```bash
PYTHONPATH=. python scripts/run_generated.py \
    --operator softmax \
    --backend triton \
    --impl llm_kernelgen/generated/exp_smoke/softmax/triton/sample_00/impl_triton.py
```

#### Step 3 — Run the full pipeline (generate → evaluate → repair)

```bash
PYTHONPATH=. python scripts/eval_llm_kernels.py \
    --experiment smoke_triton
```

This runs the `smoke_triton` experiment defined in
`llm_kernelgen/configs/experiments.yaml`:
- Model: `gpt4o`
- Backend: Triton
- Operators: from `dev_ops` split (dropout, swiglu, l2_norm, softmax)
- 1 sample per operator, no repair

#### Step 4 — Summarise results

```bash
PYTHONPATH=. python llm_kernelgen/scripts/summarize.py \
    --experiment smoke_triton
```

Prints per-operator correctness / performance / cost metrics and writes
`llm_kernelgen/generated/smoke_triton/summary.json`.

#### Step 5 — Repair a failed sample

```bash
PYTHONPATH=. python llm_kernelgen/scripts/repair.py \
    --operator softmax \
    --backend triton \
    --sample-dir llm_kernelgen/generated/exp_smoke/softmax/triton/sample_00 \
    --model o3_medium \
    --max-rounds 2
```

---

### Module layout

```
llm_kernelgen/
├── configs/
│   ├── models.yaml            # LLM provider / model aliases
│   ├── experiments.yaml       # Named experiment configs (model, operators, samples, repair)
│   └── prompt_profiles.yaml   # Zero-shot / few-shot profile definitions
│
├── clients/
│   ├── base.py                # Abstract BaseClient + LLMResponse dataclass
│   ├── openai_responses.py    # OpenAI Responses API (/v1/responses)
│   └── openai_chat_compat.py  # Any OpenAI-compatible /chat/completions endpoint
│
├── prompts/
│   ├── system.md              # System prompt (rules, interface requirements)
│   ├── task_triton.md.j2      # Jinja2 task template for Triton
│   ├── task_cutile.md.j2      # Jinja2 task template for cuTile
│   ├── repair.md.j2           # Repair prompt template
│   ├── dsl_reference/
│   │   ├── triton_minimal.md  # Triton DSL cookbook (injected into prompts)
│   │   └── cutile_minimal.md  # cuTile DSL cookbook (injected into prompts)
│   └── examples/              # Few-shot examples (train split only)
│
├── dataset/
│   ├── manifest.yaml          # Three-way train / dev / test split
│   ├── train_examples.yaml    # Operators used as few-shot demonstrations
│   ├── dev_ops.yaml           # Operators for prompt engineering / smoke tests
│   └── test_ops.yaml          # Held-out test set for paper results
│
├── runtime/
│   ├── module_loader.py       # Dynamic loader for generated kernel files
│   ├── sandbox.py             # Static safety + syntax checks
│   └── adapters.py            # Build clients from config dicts
│
├── scripts/
│   ├── build_prompt.py        # Assemble leakage-free prompts
│   ├── generate.py            # Call LLM and archive all artefacts
│   ├── extract_code.py        # Parse code block from LLM response
│   ├── evaluate.py            # Multi-stage evaluation (syntax → correctness → perf)
│   ├── repair.py              # Repair loop (error log → LLM → re-evaluate)
│   └── summarize.py           # Aggregate pass rates, speedups, token costs
│
└── generated/                 # Auto-created; never commit to git
    └── <experiment_id>/
        └── <operator>/
            └── <backend>/
                └── sample_<N>/
                    ├── prompt.md
                    ├── response.raw.txt
                    ├── impl_<backend>.py
                    ├── metadata.json
                    ├── bench.json
                    ├── eval_status.json
                    └── eval.log

scripts/
├── run_generated.py      # Evaluate one generated kernel via TileBench engine
└── eval_llm_kernels.py   # Batch generate + evaluate + repair for an experiment
```

---

### Configuration

#### `llm_kernelgen/configs/models.yaml`

Defines LLM providers and model aliases.  Add your own entry:

```yaml
providers:
  my_provider:
    api_type: chat_completions   # or "responses"
    base_url: "https://my-endpoint/v1"
    api_key_env: "MY_API_KEY"

models:
  my_model:
    provider: my_provider
    model: "model-name"
    temperature: 0.2
    max_output_tokens: 12000
```

#### `llm_kernelgen/configs/experiments.yaml`

Defines complete experiment runs:

```yaml
experiments:
  my_experiment:
    model_alias: my_model
    backends: [triton]
    prompt_profile: triton_zero_shot
    operators_split: dev_ops      # train_examples | dev_ops | test_ops
    num_samples: 5
    repair_rounds: 2
    description: "My custom experiment"
```

---

### Evaluation stages

Each generated kernel is evaluated in stages.  The highest stage reached is
recorded in `eval_status.json`:

| Stage | Name | Criterion |
|---|---|---|
| 0 | `syntax_import` | Python parses and imports without error |
| 1 | `run_case0` | `run()` executes on case 0 without exception |
| 2 | `correct_case0` | Output matches `impl_torch` on case 0 |
| 3 | `correct_all` | Correctness passes on all configured cases |
| 4 | `bench_complete` | Full benchmark timing completes |
| 5 | `perf_score` | Performance metrics computed |

---

### Paper metrics

The summary script reports:

**Generation quality**
- `syntax_pass_rate`, `runtime_pass_rate`, `correctness_pass_rate`
- `pass@1`, `pass@k` (Chen et al. 2021 unbiased estimator)
- `repair_success_rate`

**Performance** (correctness-passing kernels only)
- `mean_speedup_vs_torch`, `median_speedup_vs_torch`

**Effort / cost**
- `mean_prompt_tokens`, `mean_completion_tokens`, `mean_reasoning_tokens`
- `mean_latency_s`, `mean_loc`, total API calls

---

### Leakage audit checklist

- [ ] `impl_triton.py` / `impl_cutile.py` are **not** present in any `prompt.md`
- [ ] Few-shot examples are from `train_examples` split only
- [ ] `metadata.json` records `"forbidden_context_enforced": true`
- [ ] `allowed_context` in `metadata.json` lists only permitted files
- [ ] Repair prompts contain only the generated code + error log
- [ ] Test operator split is fixed before any generation begins
