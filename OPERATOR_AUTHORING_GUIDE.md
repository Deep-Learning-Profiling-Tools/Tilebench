# TileBench Operator Authoring Guide

Goal: fair performance comparison between PyTorch, Triton, cuTile, and (optionally) TileLang under **similar implementation strategies**.
Not about squeezing peak performance — autotune handles parameter selection; the focus is on structural equivalence.

---

## 1. Phase Policy

- Compare `torch`, `triton`, and `cutile` using the **same algorithmic strategy** (e.g., all use grouped tiling for matmul, or all use 1D blocking for element-wise).
- **Autotune is enabled**: Triton uses `@triton.autotune`; cuTile uses `ct_experimental.autotune_launch`. Do not manually tune `BLOCK_SIZE` / `tile`.
- Keep kernel structure conceptually aligned across backends (same tiling pattern, same loop order, same masking approach).
- Avoid backend-specific tricks unless discussed and documented.
- TileLang is an **optional 4th backend** (`impl_tilelang.py`): the engine skips it gracefully when the file or the `tilelang` package is absent. When provided, it must follow the same algorithmic strategy as the other backends.

---

## 2. Required Files per Operator

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

---

## 3. Minimal Coding Rules

- Same input/output semantics across all 3 backends.
- Same dtype behavior: if a dtype is not supported by a backend, **do not add workaround code inside the kernel**. Let the framework's try/except skip the case automatically.
- Same shape assumptions; correct boundary/tail handling (not only power-of-two).
- Keep kernels clean — no dtype dispatch inside the kernel unless inherent to the algorithm.

---

## 4. `config.yaml` Structure

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
  peak_bw_GBs: 8000.0                 # B200 HBM3e
  peak_tflops:                        # B200 peak compute per dtype
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

### Input size guidelines

| Operator type | Recommended `n` range |
|---|---|
| Element-wise (vector) | 1M–16M elements, step 1M |
| Reduction (softmax) | rows × cols; vary cols from 512 to 8192 |
| GEMM (matmul) | M=N=K from 256 to 4096, step 256 |
| Attention | `seq_len` from 512 to 8192 |

Always include at least one **non-power-of-two** size to test tail masking.

---

## 5. Autotune

### Triton

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048, 4096, 8192]
        for nw in [4, 8, 16]
    ],
    key=["n_elements"],   # re-tune when n_elements changes
)
@triton.jit
def my_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...

_last_config: dict | None = None

def run(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    global _last_config
    ...
    my_kernel[grid](x, out, n_elements)
    cfg = my_kernel.best_config
    if cfg is not None:
        _last_config = {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
    return out

def get_last_config() -> dict | None:
    return _last_config
```

- The autotune key should be whatever drives the optimal config change. For element-wise ops: `["n_elements"]`. For matmul: `["M", "N", "K"]`.
- Triton caches results in `~/.triton/cache/` — no re-tuning on subsequent runs for the same config.

### cuTile

```python
from types import SimpleNamespace
import cuda.tile_experimental as ct_experimental  # may be None if unavailable

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]

_last_config: dict | None = None

def run(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    global _last_config
    ...
    if ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (math.ceil(n / cfg.tile), 1, 1),
            kernel=my_kernel,
            args_fn=lambda cfg: (x, out, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_config = {"tile": result.tuned_config.tile, "occupancy": result.tuned_config.occupancy}
    else:
        ct.launch(stream, (math.ceil(n / block_size), 1, 1), my_kernel, (x, out, block_size))
    return out
```

- cuTile's autotune cache is **in-memory only** — re-runs on every new process.
- Always provide a fallback to `ct.launch` for environments where `ct_experimental` is unavailable.

---

### TileLang (optional)

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

### NKI (optional; AWS Trainium)

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
  worker runs in a private per-spec CWD with a private Neuron compile cache, and
  the exactly-one validated NEFF/HLO pair there is what gets hardware-timed.
  The `AwsNeuronCustomNativeKernel` HLO marker validates *that a graph is an
  NKI graph* — it does NOT distinguish two autotune candidates of the same
  kernel; that distinction comes only from exact winner replay.
- **Artifact selection must never depend on mtime**, sequence numbers, glob
  order, or "most recent compile event". Zero or multiple matching artifacts is
  a benchmark failure (`nki_ok=False`), never a guess. Each profiled case
  writes `results/logs/nki_profiles/<op>/<case>/<spec>/manifest.json` (exact
  NEFF path + SHA256) and a line in `results/logs/nki_neff_manifest.jsonl`.

---

## 6. Dtype Handling

### Standard floating-point types (fp16, bf16, fp32)

All three backends handle these natively. No special code needed.

### Integer types (int8, int16, int32)

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

### FP8 (`fp8_e4m3fn`, `fp8_e5m2`) — ⚠️ Special Handling Required

> **Current status (B200 / CUDA 13.1):**
> - **PyTorch**: FP8 dtype exists, but **element-wise arithmetic (`x * 2`, `x + y`, etc.) is not supported**. Only matrix-multiply paths via `torch._scaled_mm` / Transformer Engine are supported.
> - **cuTile**: Hardware (Blackwell) supports FP8, but the **cuTile DSL compiler does not yet compile FP8 element-wise kernels**. All autotune configs fail with `TileCompilerExecutionError`.
> - **Triton**: FP8 is fully supported — load/store and arithmetic with auto type-promotion.

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

---

## 7. Correctness Check

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

---

## 8. Benchmark Settings

| Setting | Recommendation | Reason |
|---|---|---|
| `use_cuda_graph: true` | All operators | Eliminates CPU launch overhead for steady-state measurement |
| `flush_l2: true` | Memory-bound operators (element-wise, attention) | Prevents artificially fast L2-cache-hot results |
| `warmup: 20` | Default | Enough for JIT compilation + TLB warm |
| `repeat: 100` | Default | Stable mean; reduce to 20 for slow kernels (attention at large seq_len) |

---

## 9. Metrics Configuration

Add a `metrics` section to `config.yaml` so `visualize.py` can compute derived metrics:

```yaml
metrics:
  # Variables available in expressions: n, dtype_size (bytes/element)
  flops_expr: "n"                   # mul2, relu, sin, vector_add: 1 FLOP/element
  bytes_expr: "n * dtype_size * 2"  # 1 read + 1 write

  # For matmul:
  # flops_expr: "2 * M * N * K"
  # bytes_expr: "(M*K + K*N + M*N) * dtype_size"
```

If `flops_expr` is `null` or omitted, TFLOPS and `pct_peak_tflops` are not computed (safe to omit for attention where FLOPs are complex to define).

---

## 10. Pre-Merge Checklist

- [ ] All configured dtypes run without error on all 3 backends (or are explicitly commented out with reason)
- [ ] Correctness check passes for all configured dtypes
- [ ] Autotune is set up for both Triton and cuTile; `get_last_config()` is implemented
- [ ] `config.yaml` has a `metrics` section with `flops_expr`, `bytes_expr`, and `plots`
- [ ] Input sizes cover at least one small, one medium, one large, and one non-power-of-two case
- [ ] Three backends follow the same algorithmic strategy
- [ ] No dtype-specific workarounds inside kernel code
- [ ] FP8 is commented out (with explanation) if not supported by all backends
- [ ] If `impl_tilelang.py` is provided: same algorithmic strategy, mutable-dict `get_last_config()`, default path passes explicit config kwargs (skips the tuning sweep), swept size dims use `T.symbolic` rather than `T.const`
