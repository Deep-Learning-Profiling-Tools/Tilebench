# TileBench

A modular accelerator performance benchmarking framework for comparing **NVIDIA cuTile (CUDA 13.2)**, **Triton**, **TileLang**, **AWS Neuron NKI**, and **PyTorch** kernel implementations.

![TileBench overview: benchmark construction, benchmark execution and comparative analysis](assets/overview.png)

![PyTorch, Triton and cuTile latency at each operator's sweep-max case on NVIDIA B200](assets/sweep_max_latency.png)

Autotuned latency of PyTorch, Triton and cuTile on one NVIDIA B200, at the largest swept case of each of the 45 operators. The dtype is fp16 where the operator sweeps it, otherwise the dtype shown after the operator name. Regenerate with `PYTHONPATH=. python scripts/plot_sweep_max.py`.

## Features

- **Multi-backend**: PyTorch · Triton · cuTile (CUDA 13.2 / Blackwell) · TileLang (optional — skipped gracefully when `impl_tilelang.py` or the `tilelang` package is absent) · NKI (optional — AWS Trainium; skipped gracefully when `impl_nki.py` is absent)
- **Proton timing**: Mean latency via Triton Proton `data="tree"`, with optional CUDA graph replay
- **Autotune**: `@triton.autotune` for Triton; `ct.tune.exhaustive_search` (through `core/cutile_autotune.py`) for cuTile; `@tilelang.autotune` for TileLang — runs before timing, results logged separately
- **Correctness checks**: dtype-aware tolerance (`torch.testing.assert_close`) against PyTorch reference; unsupported dtypes skipped gracefully
- **Flexible case generation**: `case_grid` with `expr` syntax (Python expressions), `test_cases`, or `case_preset` in `config.yaml`
- **Derived metrics**: bandwidth (GB/s), % peak BW, TFLOPS, % peak TFLOPS, arithmetic intensity, speedup — computed via per-operator expressions in `config.yaml`
- **Visualization**: latency · bandwidth · speedup · % peak BW · Roofline plots per dtype

---

## Software Versions

Software stack of the current environment. The TileBench paper results were measured with `cuda-tile` 1.3.0; the follow-up work in this repository (TileBench++: TileLang and NKI backends) uses `cuda-tile` 1.5.0.

| Component | Version |
|---|---|
| GPU | NVIDIA B200 (180 GB), driver 595.58.03 |
| OS | Red Hat Enterprise Linux 10.0, kernel 6.12 |
| CUDA toolkit | 13.2 (`nvcc` V13.2.78; cuTile's `tileiras` compiler comes from this toolkit) |
| Python | 3.10.19 |
| PyTorch | 2.10.0+cu130 (bundles the CUDA 13.0 runtime, cuDNN 9.15.1, cuBLAS 13.1.0.3) |
| Triton | 3.6.0 (includes the Proton profiler used for timing) |
| cuTile | `cuda-tile` 1.5.0 (1.3.0 for the TileBench paper), `cuda-tile-experimental` 0.0.1, `cuda-bindings` 13.0.3 |
| TileLang | `tilelang` 0.1.11 with `apache-tvm-ffi` 0.1.11 (the pin is required: 0.1.12 crashes `import tilelang`) |
| NKI | `nki` 0.6.0 with `neuronx-cc` 2.27 on an AWS trn2.3xlarge (Trainium2) |
| Nsight Compute | 2026.1.1 (NCU profiling under `tilebench_run/`) |
| Python packages | NumPy 2.2.6, PyYAML 6.0.3, Matplotlib 3.10.8 |

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

`results/logs/` is git-ignored on `main`; only the summary CSVs under `results/csv/` are tracked. Raw logs live on the `archive/raw-logs-2026-09-18` branch, which is `main` plus every raw log archived so far. `run_bench.py` snapshots `results/logs/` onto the local copy of that branch after every run, without checking anything out (skip it with `--no-archive`). Publish the snapshots with `scripts/archive_logs.sh --push`.

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
- Follow the [Operator Authoring Guide](#operator-authoring-guide) below
- Start from an existing operator of the same family, e.g. `benchmarks/operators/mul2/` for element-wise kernels
- Register input generator in `data/tensors.py`

---

## Project Structure

```
Tilebench/
├── core/
│   ├── engine.py            # Benchmark orchestration: iterate cases, verify, time all backends
│   ├── timer.py             # Proton-based GPU timing (warmup outside, repeat inside scope, L2 eviction)
│   ├── metrics.py           # Derived metrics (bandwidth, TFLOPS, speedup, roofline)
│   ├── verifier.py          # dtype-aware correctness check via torch.testing.assert_close
│   ├── dtypes.py            # resolve_dtype() string→torch.dtype; dtype_size() bytes/element
│   ├── cutile_autotune.py   # CutileAutotuner: per-shape tuning cache + cached hint variants
│   └── nki_*.py             # NKI (AWS Trainium) autotune, timing and profiling flow
│
├── data/
│   ├── tensors.py           # Input generators per operator (GENERATORS registry)
│   └── peak_performance/    # Per-device peaks (B200.json, Trainium2.json) + check_peak_specs.py
│
├── scripts/
│   ├── run_bench.py         # Benchmark one operator
│   ├── run_bench_all.py     # Benchmark every operator sequentially
│   ├── visualize.py         # Per-operator plots of derived metrics + Roofline
│   ├── measure_peak.py      # Measure peak bandwidth / FLOPS on the current GPU
│   ├── plot_sweep_max.py    # README figure: latency at each operator's sweep-max case
│   └── archive_logs.sh      # Snapshot results/logs/ onto the raw-log archive branch (run_bench.py calls it)
│
├── benchmarks/
│   ├── operators/<name>/    # 45 operators: config.yaml + impl_{torch,triton,cutile,tilelang}.py
│   └── llm_generated/<name>/<model>/<effort>/   # LLM track: iter_N/ prompts, responses, kernels; final/
│
├── tools/
│   └── llm_codegen/         # Iterative LLM kernel-generation pipeline; problems/ holds the per-operator task descriptions
│
├── skills/                  # Triton and cuTile programming guides embedded in the LLM prompts as API references
│
├── tilebench_run/           # Batch launchers + Nsight Compute (NCU) profiling harness and reports
├── tests/                   # Unit tests for the NKI profiling flow
│
├── results/
│   ├── csv/                 # <operator>_{default,autotune}.csv (tracked)
│   ├── aggregate/           # Per-operator summaries across dtypes and modes
│   ├── figures/<operator>/  # PNG plots per metric
│   └── logs/                # Raw timing + autotune JSON; git-ignored on main, see the archive branch
│
├── assets/                  # README figures
└── requirements.txt
```

---

## Timing Methodology

| Step | Details |
|---|---|
| **Warmup** | `warmup` iterations outside Proton session — JIT compile + cache warm |
| **Measurement** | `repeat` iterations each wrapped in `proton.scope("launch")` |
| **CUDA graph** | Graph captured inside Proton session; replayed each iteration |
| **L2 flush** | Before every warmup and every timed launch, a write of 2× the device's L2 size (queried at runtime; 253 MB on B200, whose L2 is 126.5 MB) evicts L2. It runs outside the scope, so its cost is excluded from timing |
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
| `--autotune` | from config | Run every backend's autotune path; writes `results/csv/<op>_autotune.csv` instead of `<op>_default.csv` |
| `--tile-language` | all | Comma-separated backends to run: `triton`, `cutile`, `tilelang`, `nki` (PyTorch always runs as the baseline) |
| `--no-archive` | false | Do not snapshot `results/logs/` onto the raw-log archive branch after the run |
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
| `--gpu` | none | GPU short name (e.g. `B200`); loads `data/peak_performance/<GPU>.json` for the roofline and % of peak metrics |

Available metrics: `latency_ms` · `bandwidth_GBs` · `pct_peak_bw` · `tflops` · `pct_peak_tflops` · `speedup` · `arithmetic_intensity` · `roofline`

---

## Operator Authoring Guide

Goal: fair performance comparison between PyTorch, Triton, cuTile, and (optionally) TileLang under **similar implementation strategies**.
Not about squeezing peak performance — autotune handles parameter selection; the focus is on structural equivalence.

### 1. Phase Policy

- Compare `torch`, `triton`, and `cutile` using the **same algorithmic strategy** (e.g., all use grouped tiling for matmul, or all use 1D blocking for element-wise).
- **Autotune is enabled**: Triton uses `@triton.autotune`; cuTile uses `ct.tune.exhaustive_search` through `core.cutile_autotune.CutileAutotuner`. Do not manually tune `BLOCK_SIZE` / `tile`.
- Keep kernel structure conceptually aligned across backends (same tiling pattern, same loop order, same masking approach).
- Avoid backend-specific tricks unless discussed and documented.
- TileLang is an **optional 4th backend** (`impl_tilelang.py`): the engine skips it gracefully when the file or the `tilelang` package is absent. When provided, it must follow the same algorithmic strategy as the other backends.

### 2. Required Files per Operator

```
benchmarks/operators/<name>/
├── config.yaml        # case_grid, benchmark params, metrics expressions
├── impl_torch.py      # def run(*inputs) -> Tensor
├── impl_triton.py     # def run(*inputs, block_size=...) -> Tensor  + get_last_config()
├── impl_cutile.py     # def run(*inputs, block_size=...) -> Tensor  + get_last_config()
└── impl_tilelang.py   # optional — same run()/get_last_config() contract
```

Register the input generator in `data/tensors.py`:
```python
def generate_<name>_inputs(n, dtype=torch.float32, device='cuda'):
    ...
GENERATORS["<name>"] = generate_<name>_inputs
```

### 3. Minimal Coding Rules

- Same input/output semantics across all 3 backends.
- Same dtype behavior: if a dtype is not supported by a backend, **do not add workaround code inside the kernel**. Let the framework's try/except skip the case automatically.
- Same shape assumptions; correct boundary/tail handling (not only power-of-two).
- Keep kernels clean — no dtype dispatch inside the kernel unless inherent to the algorithm.

### 4. `config.yaml` Structure

```yaml
benchmark:
  warmup: 20            # warm-up iterations (outside Proton scope, for JIT + cache)
  repeat: 100           # timed iterations (inside proton.scope)
  use_cuda_graph: true  # recommended for all operators
  flush_l2: true        # recommended for memory-bound operators

case_grid:
  # Use expr for a range of problem sizes — enables performance-vs-size curves
  n:
    expr: "[1024 * 1024 * i for i in range(1, 17)]"   # 1M … 16M, step 1M
  dtype: ["fp16", "bf16", "fp32", "int8"]
  # Keep dtype list to what ALL three backends can handle without workarounds.
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

#### Input size guidelines

| Operator type | Recommended `n` range |
|---|---|
| Element-wise (vector) | 1M–16M elements, step 1M |
| Reduction (softmax) | rows × cols; vary cols from 512 to 8192 |
| GEMM (matmul) | M=N=K from 256 to 4096, step 256 |
| Attention | `seq_len` from 512 to 8192 |

Always include at least one **non-power-of-two** size to test tail masking.

### 5. Autotune

#### Triton

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
    cfg = _my_kernel_autotuned.best_config
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
```

- `get_last_config()` reads `best_config` straight off the autotuned kernel. Do not keep a module-level `_last_config`, and never use `global` in `run()`.
- The autotune key should be whatever drives the optimal config change. For element-wise ops: `["n_elements"]`. For matmul: `["M", "N", "K"]`.
- Triton caches results in `~/.triton/cache/` — no re-tuning on subsequent runs for the same config.

#### cuTile

```python
from types import SimpleNamespace
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

_last_autotune_config: dict = {}          # mutable-dict pattern — never `global`

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
- Both caches are in-memory only — tuning re-runs in every new process.
- The keys returned by `get_last_config()` must match the names in `_DEFAULT_CONFIG`. The NCU harness replays winners by those names, and a mismatch makes it silently profile the defaults.
- The selected configs of every backend are printed during the run and saved to `results/logs/autotune_logs/<op>_autotune.json`.

#### TileLang (optional)

Requires `tilelang==0.1.11` with `apache-tvm-ffi==0.1.11` — the pin is load-bearing
(tvm-ffi 0.1.12 makes `import tilelang` abort at C++ level).

```python
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_last_autotune_config: dict = {}          # mutable-dict pattern — never `global`

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

- The tuning sweep result is cached in-process — repeat `autotune=True` calls cost
  ~ms, so the Proton measurement window stays clean.
- Prefer `T.symbolic` over `T.const` for swept problem-size dims: `T.const` makes the
  size a compile-time constant and recompiles per shape (~3.4 s each on B200, ×80
  cases adds minutes of wall-clock per operator).
- CUDA-graph capture of the compiled kernel works — keep `use_cuda_graph: true`.

#### NKI (optional; AWS Trainium)

NKI autotuning used by TileBench MUST go through `core.nki_autotune.NkiAutotuner`
(or an explicit replay adapter with the same trace semantics). Manual autotuners
without replay semantics are unsupported — the profiling flow cannot replay them
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
  Anything else raises at tune time — no `repr()` fallback.
- **Profiling flow** (`core/nki_orchestrator.py`): a SELECTOR process runs the
  sweep and exports the canonical winner trace; a fresh PROFILE process installs
  that trace and **replays the winner exactly** — zero candidate timings; a
  replay mismatch (changed search space, stale winner) fails loudly before
  anything is profiled.
- **Artifact identity** is a canonical launch spec (`spec_id`): the profile
  worker runs in a private per-spec CWD with a private Neuron compile cache and
  records EVERY NEFF/HLO pair each phase (torch baseline, NKI) compiled — an
  operator may compile several graphs per `run()` (one per radix-sort pass).
  The `AwsNeuronCustomNativeKernel` HLO marker validates *that a graph is an
  NKI graph* — it does NOT distinguish two autotune candidates of the same
  kernel; that distinction comes only from exact winner replay.
- **Timing** comes from the Neuron runtime's inspect trace
  (`NEURON_RT_INSPECT_*`, private output dir per spec): the worker runs
  `run()` `warmup` + `repeat` times on the case's real inputs and records, per
  timed iteration, the range of XLA execution indices it covered; the parent
  orders the trace's executions, checks their count against the worker's,
  sums the device time inside each range (multi-graph `run()`s are summed —
  no host clock is ever compared with the trace's timebase) and matches
  every executed NEFF — written back by the runtime, byte-identical to the
  compiler dump — to a recorded pair by SHA256. `NKI(ms)` is the mean over the
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

### 6. Dtype Handling

#### Standard floating-point types (fp16, bf16, fp32)

All three backends handle these natively. No special code needed.

#### Integer types (int8, int16, int32)

- Generate inputs with bounded values to avoid overflow after the operation.
- Example for `int8` with `x * 2`: use `torch.randint(-32, 33, ...)` so result stays in `[-128, 127]`.
- `torch.testing.assert_close` uses exact comparison (`atol=0, rtol=0`) for integer types — this is intentional.

```python
# data/tensors.py
def generate_<name>_inputs(n, dtype=torch.float32, device='cuda'):
    if dtype == torch.int8:
        x = torch.randint(-64, 65, (n,), device=device).to(torch.int8)
    else:
        x = torch.randn(n, dtype=dtype, device=device)
    return (x,)
```

#### FP8 (`fp8_e4m3fn`, `fp8_e5m2`) — ⚠️ Special Handling Required

> **Current status (B200 / CUDA 13.2, PyTorch 2.10, Triton 3.6.0, cuda-tile 1.5.0):** FP8 element-wise arithmetic fails on all three backends. Checked with the `mul2` kernels on `float8_e4m3fn` and `float8_e5m2` inputs.
> - **PyTorch**: the FP8 dtypes exist, but element-wise arithmetic (`x * 2`, `x + y`, etc.) raises `NotImplementedError` (`"mul_cuda" not implemented for 'Float8_e4m3fn'`). Only matrix-multiply paths via `torch._scaled_mm` / Transformer Engine are supported.
> - **cuTile**: Blackwell supports FP8 in hardware, but the DSL rejects FP8 arithmetic at compile time (`TileTypeError: non-arithmetic dtype float8_e4m3fn`).
> - **Triton**: FP8 load/store works, but arithmetic directly on an FP8 value does not compile in Triton 3.6.0 (`'ir.builder' object has no attribute 'get_fp8e4nv'`).

**Rule: do NOT add FP8 workarounds inside `impl_torch.py`, `impl_triton.py`, or `impl_cutile.py`.**

Instead, comment out FP8 from `config.yaml` until both backends support it:

```yaml
dtype: ["fp16", "bf16", "fp32", "int8"]
# dtype: ["fp16", "bf16", "fp32", "fp8_e4m3fn", "fp8_e5m2", "int8"]
# fp8: PyTorch has no native FP8 element-wise arithmetic;
#       cuTile DSL does not yet compile FP8 element-wise kernels.
#       Re-enable when both backends support it.
```

If FP8 causes a runtime error, the framework's `try/except` in `engine.py` will skip the case automatically and print a clear message. The benchmark will continue with other dtypes.

FP8 **is** meaningful for GEMM kernels (matmul, attention) where Tensor Core throughput matters. For those, PyTorch provides `torch._scaled_mm` and Transformer Engine — check if those are available before adding FP8 to the dtype list.

### 7. Correctness Check

- The framework runs `core/verifier.py` automatically — no custom correctness script needed.
- Tolerances are **dtype-aware** (defined in `core/verifier.py`):

| dtype | atol | rtol |
|---|---|---|
| float32 | 1e-5 | 1.3e-6 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1.6e-2 |
| int8 / int16 / int32 / int64 | 0 | 0 (exact) |

- If a backend fails correctness, it is excluded from timing but the run continues.
- Do **not** weaken tolerances to make a broken kernel pass — fix the kernel.

### 8. Benchmark Settings

| Setting | Recommendation | Reason |
|---|---|---|
| `use_cuda_graph: true` | All operators | Eliminates CPU launch overhead for steady-state measurement |
| `flush_l2: true` | Memory-bound operators (element-wise, attention) | Prevents artificially fast L2-cache-hot results |
| `warmup: 20` | Default | Enough for JIT compilation + TLB warm |
| `repeat: 100` | Default | Stable mean; reduce to 20 for slow kernels (attention at large seq_len) |

### 9. Metrics Configuration

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

### 10. Pre-Merge Checklist

- [ ] All configured dtypes run without error on all 3 backends (or are explicitly commented out with reason)
- [ ] Correctness check passes for all configured dtypes
- [ ] Autotune is set up for both Triton and cuTile; `get_last_config()` is implemented
- [ ] `config.yaml` has a `metrics` section with `flops_expr`, `bytes_expr`, and `plots`
- [ ] Input sizes cover at least one small, one medium, one large, and one non-power-of-two case
- [ ] Three backends follow the same algorithmic strategy
- [ ] No dtype-specific workarounds inside kernel code
- [ ] FP8 is commented out (with explanation) if not supported by all backends
- [ ] If `impl_tilelang.py` is provided: same algorithmic strategy, mutable-dict `get_last_config()`, default path passes explicit config kwargs (skips the tuning sweep), swept size dims use `T.symbolic` rather than `T.const`
