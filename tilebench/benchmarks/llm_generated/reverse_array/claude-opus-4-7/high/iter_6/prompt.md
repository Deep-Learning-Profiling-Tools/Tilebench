# Task: improve operator `reverse_array` for TileBench (iteration 6)

Your previous iteration (5) did not yet meet the stopping criterion (`stop_score` ≥ 80% on the **single largest case per dtype**) for `triton`, `cutile`. Read the trajectory below carefully — if your last iteration **regressed** vs the best verify-clean iter so far, you should consider going back to that approach as your starting point and trying a different optimization. Then re-emit `impl_triton.py`, `impl_cutile.py`.


## Iteration trajectory so far

| iter | triton cfg | triton score | triton speedup_vs_torch | triton verify | cutile cfg | cutile score | cutile speedup_vs_torch | cutile verify |
|---|---|---|---|---|---|---|---|---|
| 0 | {BLOCK_SIZE:4096, num_warps:8, num_stages:2} | 69.4% | 2.66× | ✓ | — | 0.0% | — | ✗4 |
| 1 | {BLOCK_SIZE:8192, num_warps:8, num_stages:4} | 70.9% | 2.76× | ✓ | {TILE:4096, occupancy:4} | 69.8% | 2.65× | ✓ |
| 2 | {BLOCK_SIZE:8192, num_warps:8, num_stages:4} | 3.2% | 0.10× | ✓ | {TILE:8192, occupancy:4} | 68.1% | 2.59× | ✓ |
| 3 | {BLOCK_SIZE:8192, num_warps:16, num_stages:2} | 69.1% | 2.64× | ✓ | {TILE:8192, occupancy:8} | 63.6% | 2.32× | ✓ |
| 4 | {BLOCK_SIZE:16384, num_warps:8, num_stages:2} | 65.5% | 2.54× | ✓ | {TILE:16384, occupancy:4} | 61.8% | 2.39× | ✓ |
| 5 | {BLOCK_SIZE:8192, num_warps:8, num_stages:2} | 3.2% | 0.10× | ✓ | — | 0.0% | — | ✗4 |

_`cfg` is the configuration your kernel actually used, as returned by `get_last_config()`. `speedup_vs_torch` is `torch_latency / kernel_latency` — values >1 mean your kernel beat the PyTorch reference; <1 means torch is still faster._

**Best verify-clean `triton` so far: iter 1 with stop_score=70.9%.**
⚠ Your `triton` last iter (iter 5) **regressed by 67.7 pp** vs iter 1 — consider reverting to that approach.
**Best verify-clean `cutile` so far: iter 1 with stop_score=69.8%.**
⚠ Your `cutile` last iter (iter 5) **broke verification** (4 cases). Revert to iter 1's code and apply a smaller, safer change.

## Feedback from iteration 5

### ❌ Verification failures (4 cases)
- `cutile` / dtype=`fp16` / params={'n': 20000000, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 227, in _run_one_case
    output = impl.run(*inputs)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/reverse_array/claude-opus-4-7/high/iter_5/impl_cutile.py", line 27, in run
    kernel = _reverse_kernel.with_hints(occupancy=occupancy)
AttributeError: 'kernel' object has no attribute 'with_hints'

- `cutile` / dtype=`bf16` / params={'n': 20000000, 'dtype': 'bf16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 227, in _run_one_case
    output = impl.run(*inputs)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/reverse_array/claude-opus-4-7/high/iter_5/impl_cutile.py", line 27, in run
    kernel = _reverse_kernel.with_hints(occupancy=occupancy)
AttributeError: 'kernel' object has no attribute 'with_hints'

- `cutile` / dtype=`fp32` / params={'n': 20000000, 'dtype': 'fp32'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 227, in _run_one_case
    output = impl.run(*inputs)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/reverse_array/claude-opus-4-7/high/iter_5/impl_cutile.py", line 27, in run
    kernel = _reverse_kernel.with_hints(occupancy=occupancy)
AttributeError: 'kernel' object has no attribute 'with_hints'

- `cutile` / dtype=`int8` / params={'n': 20000000, 'dtype': 'int8'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 227, in _run_one_case
    output = impl.run(*inputs)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/reverse_array/claude-opus-4-7/high/iter_5/impl_cutile.py", line 27, in run
    kernel = _reverse_kernel.with_hints(occupancy=occupancy)
AttributeError: 'kernel' object has no attribute 'with_hints'


### `triton` performance — stop_score=**3.2%** (target ≥ 80%), report_mean=3.2%
### `cutile` performance — stop_score=**0.0%** (target ≥ 80%), report_mean=0.0%

Per-(backend,dtype,case) detail (sorted by worst first):
- `triton` / `int8` / {'n': 20000000, 'dtype': 'int8'}: 2.4% of roofline (bandwidth), 0.17× torch, cfg={BLOCK_SIZE:8192, num_warps:8, num_stages:2}
- `triton` / `fp16` / {'n': 20000000, 'dtype': 'fp16'}: 2.6% of roofline (bandwidth), 0.10× torch, cfg={BLOCK_SIZE:8192, num_warps:8, num_stages:2}
- `triton` / `bf16` / {'n': 20000000, 'dtype': 'bf16'}: 2.6% of roofline (bandwidth), 0.10× torch, cfg={BLOCK_SIZE:8192, num_warps:8, num_stages:2}
- `triton` / `fp32` / {'n': 20000000, 'dtype': 'fp32'}: 5.2% of roofline (bandwidth), 0.11× torch, cfg={BLOCK_SIZE:8192, num_warps:8, num_stages:2}

→ Action: focus on the worst-performing combinations. If the case is bandwidth-bound, optimize memory coalescing / tile layout / async copies. If compute-bound, ensure Tensor Cores (tl.dot/ct.mma) and high arithmetic intensity per load.

---

## Your previous `impl_triton.py` (iteration 5)

```python
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    k = tl.arange(0, BLOCK_SIZE)
    # Output positions (ascending, contiguous)
    dst_offs = pid * BLOCK_SIZE + k
    dst_mask = dst_offs < N
    # Source positions: we want output[dst_offs[i]] = input[N-1-dst_offs[i]]
    # If we load source ASCENDING starting at (N - (pid+1)*BLOCK_SIZE) and then
    # flip the loaded tile, we get the correct values stored ascending.
    src_start = N - (pid + 1) * BLOCK_SIZE
    src_offs = src_start + k
    src_mask = src_offs >= 0  # only the LAST tile may have negative offsets
    x = tl.load(in_ptr + src_offs, mask=src_mask, other=0)
    x = tl.flip(x, 0)
    tl.store(out_ptr + dst_offs, x, mask=dst_mask)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_SIZE = 8192
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _reverse_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

```

## Your previous `impl_cutile.py` (iteration 5)

```python
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _reverse_kernel(input_arr, output_arr, N, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32) + bid * TILE
    src = (N - 1) - offs
    vals = ct.gather(input_arr, src)
    ct.scatter(output_arr, offs, vals)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 2

    grid = (ct.cdiv(N, TILE), 1, 1)
    kernel = _reverse_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (input, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

```

---

## Best verify-clean code so far

### Best-so-far `impl_triton.py` (iter 1, stop_score=70.9%)

Your previous iteration regressed (or broke verify) for this backend. Use this code as the starting point and apply a different optimization.

```python
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    src = N - 1 - offs
    x = tl.load(in_ptr + src, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_SIZE = 8192
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _reverse_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

```

### Best-so-far `impl_cutile.py` (iter 1, stop_score=69.8%)

Your previous iteration regressed (or broke verify) for this backend. Use this code as the starting point and apply a different optimization.

```python
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _reverse_kernel(input_arr, output_arr, N, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32) + bid * TILE
    src = (N - 1) - offs
    vals = ct.gather(input_arr, src)
    ct.scatter(output_arr, offs, vals)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    TILE = 4096

    grid = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid, _reverse_kernel, (input, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 4})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None

```

---

# Reminders: TileBench Framework Conventions

# TileBench Framework Guide

Read this carefully. Your generated kernels MUST follow this contract or the
benchmark harness will fail to load / run / evaluate them.

## Files you must produce

For operator `<op>`, write exactly two files:

1. **`impl_triton.py`** — Triton kernel + Python `run()` wrapper
2. **`impl_cutile.py`** — cuTile kernel + Python `run()` wrapper

You do NOT generate `impl_torch.py` — it is copied from the human-written
reference implementation. Your `run()` function's signature MUST match
`impl_torch.run()` exactly (same positional args, same keyword args).

## Required exports (both impls)

```python
def run(*args, **kwargs):
    """Execute the operator. Must produce a single torch.Tensor (or tuple
    of tensors) that bit-equivalently matches impl_torch.run() within the
    tolerances in config.yaml's `verify:` section."""

def get_last_config() -> dict | None:
    """Return the configuration this run() actually used, as a flat dict
    (e.g. {'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}).
    The harness logs this so the iterative-refinement loop can show you
    the trajectory of configurations you've tried."""
```

## ⛔️ No autotune — you pick one configuration per iteration

The harness does **NOT** call autotune. Your kernel runs with a single
hard-coded configuration that **YOU** choose. The 10-iteration refinement
loop is the search mechanism: at each iteration you see the previous
iteration's `(config → roofline_pct, latency_ms, speedup_vs_torch)`, and
you propose the next configuration based on that signal.

The following are **FORBIDDEN** in your code and will be rejected at load:

- `triton.autotune(...)` decorator
- `from core.cutile_autotune import CutileAutotuner` (or any use of `CutileAutotuner`)
- `ct.tune.exhaustive_search` / `ct_experimental.autotune_launch`
- Any other in-kernel autotune-style search

If you import autotune machinery the harness rejects the file before it
is ever executed.

## Triton template (no autotune)

```python
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}     # populated by run(); read by get_last_config()


@triton.jit
def _op_kernel(x_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr,
               # ... other constexpr params your kernel needs
               ):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    # ... your computation ...
    tl.store(out_ptr + offs, x, mask=mask)


def run(x):
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Pick ONE configuration. Iterate to refine across iterations.
    BLOCK_SIZE = 2048
    num_warps  = 4
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _op_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_SIZE": BLOCK_SIZE,
                      "num_warps":  num_warps,
                      "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

**Rules:**
- `_LAST_CFG` is a module-level `dict` — mutate with `.clear()` + `.update()`,
  never use the `global` keyword.
- Record EVERY parameter that influenced the kernel launch (block sizes,
  `num_warps`, `num_stages`, group size for grouped-launch matmuls, etc.)
  so the iterative-refinement loop can see exactly what you tried.

## cuTile template (no autotune)

```python
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}     # populated by run(); read by get_last_config()


@ct.kernel
def _op_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    # ... your computation ...
    ct.store(output, index=(bid,), tile=x_tile)


def run(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    # Pick ONE configuration. Iterate to refine across iterations.
    TILE       = 2048
    occupancy  = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    # `kernel_with_hints` is how cuTile communicates occupancy / num_warps
    # to the launcher. Build it from the bare `@ct.kernel` directly.
    kernel = _op_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

**Rules:**
- `_LAST_CFG` is a module-level `dict` — mutate with `.clear()` + `.update()`,
  never use the `global` keyword.
- cuTile tile-shape dimensions MUST be powers of 2.
- For non-power-of-2 problem dims: use `padding_mode=ct.PaddingMode.ZERO`
  on `ct.load`, and `ct.store` silently drops OOB writes.
- For max-reduction kernels needing to neutralize OOB: use
  `padding_mode=ct.PaddingMode.NEG_INF`.
- `ct.store()` only accepts static indices. For runtime-computed scatter
  indices, use `ct.scatter()`.

## Choosing tile / BLOCK sizes

Tiles below the thresholds below are provably suboptimal on B200 (each
thread loads / computes too few elements to amortise launch overhead,
miss memory coalescing windows, or fall off the Tensor Core path).

**Lower bounds — do NOT pick below these.** All values must be powers of 2.

| Operator shape | Triton `BLOCK_SIZE` / cuTile `tile` | Triton matmul `BLOCK_M`,`BLOCK_N` | Triton matmul `BLOCK_K` | cuTile matmul `tm`,`tn` | cuTile matmul `tk` |
|---|---|---|---|---|---|
| 1D elementwise / pointwise | ≥ **512** | — | — | — | — |
| Row-reduction / softmax / norm (per-row inner tile) | ≥ **256** along reduction dim | — | — | — | — |
| Stencil / 2D conv (sliding window) | ≥ **256** along the spatial inner dim | — | — | — | — |
| Matmul / attention (Tensor Core) | — | ≥ **64** | ≥ **32** | ≥ **64** | ≥ **32** |

**Upper bounds — do NOT pick above these.** Larger tiles run out of
registers / shared memory on B200 (sm_100, 228 KB shmem, 64 K registers
per CTA).

| Operator shape | Reasonable upper bound |
|---|---|
| 1D elementwise / pointwise | `BLOCK_SIZE` ≤ **8192** for fp16/fp32; ≤ **16384** for int8 |
| Matmul tile area | `BLOCK_M × BLOCK_N` ≤ **256 × 256** for fp16, ≤ **128 × 256** for fp32, ≤ **256 × 256** for fp8/int8 |
| Matmul K-tile | `BLOCK_K` ≤ **128** for fp16/fp8/int8, ≤ **64** for fp32 |

**Sensible starting points for iter 0:**

| Use case | Reasonable iter-0 cfg |
|---|---|
| 1D pointwise (Triton) | `BLOCK_SIZE=2048, num_warps=4, num_stages=2` |
| 1D pointwise (cuTile) | `tile=2048, occupancy=8` |
| Per-row reduction (Triton, 1 CTA per row) | `BLOCK_N=1024, num_warps=4` |
| Matmul (Triton, fp16/bf16) | `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3` |
| Matmul (cuTile, fp16/bf16) | `tm=128, tn=128, tk=64, occupancy=8` |
| Matmul (Triton, fp32) | `BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2` |

These are reasonable defaults — your job across the 10 iterations is to
beat them by trying configurations that fit the operator's arithmetic
intensity and problem size better.

## Multi-dtype support

If `config.yaml`'s `case_grid.dtype` lists multiple dtypes (e.g.
`["fp16", "bf16", "fp32"]`), your `run()` function must work for **all
of them** in a single `run()` call with **ONE configuration shared across
all dtypes**. Use a single kernel that is dtype-polymorphic:

- Triton's `tl.dot` and cuTile's `ct.mma` adapt to input dtype
  automatically.
- Do NOT branch `BLOCK_M`, `num_warps`, etc. on `a.dtype`; pick one
  tile that is correct for **all** dtypes in the grid.

**Picking a tile that works for every dtype**: the binding constraint is
usually fp32 (4 bytes/element × 3 stages × the tile area must fit in
228 KB shared memory on B200). If your fp16 tile would not fit in fp32,
shrink the tile so it fits fp32 — then fp16/bf16 just use spare
capacity. Concretely, a Triton matmul with `BLOCK_M=128, BLOCK_N=128,
BLOCK_K=64, num_stages=3` needs ~192 KB shmem in fp32 (close to the
limit); the safe default is `BLOCK_K=32, num_stages=2` for fp32-capable
matmul configs, which leaves room for the TMA descriptor and async
copy slots.

For mixed-dtype matmul (e.g. fp32 + fp16 + fp8 in one op), look at
`benchmarks/operators/matmul_fp32_fp16_fp8/impl_triton.py` for reference.

## Hardware constraints to remember (B200, sm_100)

- **Tensor Memory Accelerator (TMA)** via `tl.make_tensor_descriptor`:
  the descriptor's `block_shape` last dimension must occupy ≥ 16 bytes
  (i.e. ≥ 8 elements for fp16, ≥ 4 for fp32, ≥ 16 for int8). If your op
  is 1D pointwise, do NOT use TMA descriptors — use plain `tl.load`.
- **Tensor Core**: `tl.dot` / `ct.mma` triggers MMA hardware. Triton's
  default `input_precision="tf32"` promotes fp32 inputs to TF32 TC
  (1100 TFLOPS dense ceiling). Use this unless strict IEEE-754 fp32 is
  required.
- **Powers of two**: cuTile tile dims must be powers of 2; non-pow2
  cases use padding.

## ⛔️ FORBIDDEN: delegating the actual computation to PyTorch / cuDNN / cuBLAS

Your `run()` function in `impl_triton.py` MUST perform the operator's
computation through your `@triton.jit` kernel, and your `run()` in
`impl_cutile.py` MUST perform it through your `@ct.kernel`. You are
NOT allowed to compute the output by calling any of the following from
inside `run()` (this list is non-exhaustive but covers the common cheats
the harness detects):

- `torch.nn.functional.*` — `scaled_dot_product_attention`, `softmax`,
  `layer_norm`, `linear`, `conv1d`/`conv2d`/`conv3d`, `cross_entropy`,
  `max_pool2d`, `batch_norm`, `dropout`, etc.
- `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`,
  `torch.linalg.*`
- `torch.softmax`, `torch.argmax`, `torch.sort`, `torch.topk`,
  `torch.bincount`, `torch.flip`, `torch.transpose` *as a substitute
  for the actual operator's computation*
- Any function whose name on its own is a literal description of the
  operator (e.g. you cannot use `torch.norm` inside the `l2_norm`
  operator's `run`).
- Importing the reference (`from impl_torch import run`).

The harness will reject any `impl_triton.py` or `impl_cutile.py` whose
`run()` body contains a call to a forbidden pattern, even if the kernel
file *also* defines a `@triton.jit`/`@ct.kernel` function. A "decoy"
kernel that is defined but not invoked on the real workload, or invoked
only on a placeholder one-element input, will be detected and rejected.

Valid uses of `torch.*` inside `run()` are limited to: tensor allocation
(`torch.empty`, `torch.empty_like`, `torch.zeros`, `torch.zeros_like`),
shape manipulation that does not compute (`.contiguous()`, `.view()`,
`.reshape()`, `.transpose()`, `.unsqueeze()`, `.permute()` — but
**not** as a substitute for the operator), stream / event
management (`torch.cuda.current_stream`, `torch.cuda.synchronize`), and
dtype-only casts on metadata (**not** on the data path that should
be computed by your kernel).

## ⛔️ FORBIDDEN: caching outputs across `run()` calls

Do NOT add a Python-level cache that returns a previously-computed output when
the inputs look identical. The harness will pass the same input objects (or
freshly-regenerated tensors with the same values) to `run()` many times during
warmup, verify, and timing — and an output cache would make the kernel appear
to run in ~0 ms, inflating the measured throughput far past the hardware
roofline. The harness detects this (it regenerates fresh inputs between
warmup and the timed loop) and will flag any implementation whose measured
throughput exceeds the hardware roofline as a cache cheat.

Every `run()` call must perform the actual GPU work — no Python shortcuts,
no `_OUTPUT_CACHE`-style dicts, no `is`/`data_ptr`/`_version` shortcuts that
skip the kernel launch.

## Common pitfalls (the harness rejects these)

1. **Importing modules you didn't list**: stay within `torch`, `triton`,
   `triton.language`, `cuda.tile`, `cuda.tile_experimental` (optional),
   `math`, `numpy as np`. Do NOT import `core.cutile_autotune`.
2. **Wrong `run()` signature**: if `impl_torch.run(x, y, BATCH, M, N, K)`
   takes 6 positional args, your `run()` must take the same 6.
3. **Returning a list when impl_torch returns a tensor** (or vice versa).
4. **Not handling OOB tail tiles**: input sizes from `config.yaml` may
   not be exact multiples of your `BLOCK_SIZE`. Always mask.
5. **Mutating inputs in-place**: the harness re-uses input tensors
   across the verify + timing repetitions. Clone if you need to.
6. **`get_last_config()` returning wrong type**: must return `dict` or
   `None`. Do not return `Config` objects or `SimpleNamespace`.

## What the engine does with your code

For each `(case, dtype)` from `config.yaml`'s `case_grid`:
1. Generates inputs via `data/tensors.py`'s `GENERATORS[<op>]`.
2. Calls `impl_torch.run(*inputs)` and times it via NVIDIA Proton (the
   PyTorch baseline used for the `speedup_vs_torch` column).
3. Calls `impl_triton.run(*inputs)` once for verify; checks output
   matches the torch reference within `verify:` tolerances. On failure,
   no timing is collected for that backend.
4. If verify passes: times `impl_triton.run(*inputs)` over a rotating
   sequence of fresh inputs (defeats output-caching cheats), 5 warmup
   + 20 measured iterations, inside a `proton.scope("launch")`. Mean
   latency from the Proton hatchet tree is reported.
5. Same flow for `impl_cutile.run()`.
6. `get_last_config()` is called once per backend per case and the
   returned dict is included in the per-iteration feedback so the next
   iteration's prompt shows what you tried.

The metric for stopping is per-backend: each of Triton and cuTile is
frozen independently when its `stop_score` (capped `roofline_pct` on
the single largest case for that backend's dtype — averaged across
dtypes if the op has multiple) reaches ≥ 0.80 AND that iteration is
verify-clean for that backend.

## TL;DR checklist before you return code

- [ ] Two files: `impl_triton.py` and `impl_cutile.py`
- [ ] Each exports `run(...)` and `get_last_config() -> dict | None`
- [ ] `run()` signature matches `impl_torch.run()` exactly (no `autotune` kwarg)
- [ ] **NO** `triton.autotune` decorator, **NO** `CutileAutotuner` import
- [ ] One hard-coded configuration, recorded in `_LAST_CFG`
- [ ] No `_DEFAULT_CONFIG`, no `global` keyword for `_LAST_CFG`
- [ ] Multi-dtype handled in a single `run()`
- [ ] OOB / non-pow-2 handled with masks (Triton) or `padding_mode` (cuTile)
- [ ] No in-place mutation of inputs


---


# Triton (triton / triton.language) API Reference

Treat the API reference below as authoritative for the Triton version installed in this repo (3.6.0); do NOT use APIs from later versions you may have seen in training data. When writing `impl_triton.py`, every `tl.*` / `triton.*` symbol you use must appear in this reference.

**IMPORTANT — autotune sections of this reference do NOT apply.** The reference below was written as a general Triton programming guide and discusses `triton.autotune` / `triton.Config` / `_DEFAULT_CONFIG` extensively. In THIS pipeline you must NOT use any of them — see the 'No autotune' rule in the TileBench framework conventions above. Use the reference for kernel-body syntax (tl.load, tl.store, tl.dot, masking, make_block_ptr, make_tensor_descriptor, etc.); ignore everything about cfg search / autotune wrappers.

# Triton Comprehensive Programming Guide

Triton is an open-source Python DSL that compiles through MLIR/LLVM to high-performance GPU kernels (NVIDIA PTX, AMD GCN). Every Triton kernel is an SPMD program: a **grid** of independent **programs** (CTAs) that operate on **block tensors** held in registers/SRAM.

This guide covers Triton **3.6.0** as shipped in `lib/Triton/triton-3.6.0/`. All snippets are valid inside a `@triton.jit` function unless marked otherwise.

---

## 1. Import Boilerplate

Every `impl_triton.py` MUST start with:

```python
import torch
import triton
import triton.language as tl
```

Optional imports for advanced features:

```python
from triton.tools.tensor_descriptor import TensorDescriptor   # host-side TMA desc
from triton.runtime import driver                             # device-info queries
import triton.profiler as proton                              # GPU profiler (used by TileBench timer)
```

Key namespaces:

| Namespace | Purpose |
|-----------|---------|
| `triton` | Top-level: `jit`, `autotune`, `Config`, `heuristics`, `cdiv`, `next_power_of_2`, `compile`, `testing` |
| `triton.language` (alias `tl`) | All in-kernel primitives: `load`, `store`, `dot`, `arange`, `program_id`, `exp`, `sum`, ... |
| `triton.language.math` (alias `tl.math`) | Approximate / precise math: `exp2`, `log2`, `rsqrt`, `sqrt_rn`, `div_rn`, `erf`, `fma` |
| `triton.runtime` | `driver`, `JITFunction`, `Config`, `Autotuner`, `Heuristics` |
| `triton.testing` | `do_bench`, `do_bench_cudagraph`, `assert_close`, `perf_report`, `Benchmark` |

Top-level re-exports (accessible from `triton.*`):
`autotune`, `Config`, `heuristics`, `JITFunction`, `KernelInterface`, `reinterpret`, `TensorWrapper`, `OutOfResources`, `InterpreterError`, `MockTensor`, `jit`, `constexpr_function`, `compile`, `CompilationError`, `TritonError`, `set_allocator`, `AsyncCompileMode`, `FutureKernel`, `cdiv`, `next_power_of_2`.

---

## 2. Data Types

### 2.1 DType Constants (all on `tl`)

| Integer | Float | FP8 (NVIDIA names) | FP8 (aliases) |
|---------|-------|--------------------|---------------|
| `tl.int1` (bool) | `tl.float16` | `tl.float8e4nv` (E4M3, NVIDIA) | `tl.float8e4b8`, `tl.float8e4b15` |
| `tl.int8` / `tl.uint8` | `tl.bfloat16` | `tl.float8e5` (E5M2) | `tl.float8e5b16` |
| `tl.int16` / `tl.uint16` | `tl.float32` | | |
| `tl.int32` / `tl.uint32` | `tl.float64` | | |
| `tl.int64` / `tl.uint64` | | | |

DType properties (used at compile time via `constexpr`):
```python
tl.float16.primitive_bitwidth   # 16
tl.int32.is_int_signed()        # True
tl.float16.is_floating()        # True
tl.float8e5.is_fp8()            # True
```

### 2.2 Dtype of a Pointer / Tensor

```python
x_ptr.dtype.element_ty          # element dtype behind the pointer
x.dtype                         # dtype of a loaded block tensor
x.type.scalar                   # scalar dtype (for blocks)
x.type.scalar.primitive_bitwidth
```

### 2.3 Common Casts

```python
y = x.to(tl.float32)                                  # upcast for accumulation
y = x.to(tl.float16)                                  # downcast to store
y = x.to(tl.int32, bitcast=True)                      # reinterpret bits
y = x.to(tl.bfloat16, fp_downcast_rounding="rtne")    # explicit rounding
```

`.to()` is the canonical cast; `tl.cast(x, dtype, ...)` is the functional form.

**Rule**: compute in fp32, store in original dtype. Triton auto-casts on `tl.store` if target tensor's dtype differs.

---

## 3. The Kernel Programming Model

### 3.1 Single Program = Single Tile

A kernel launched on a grid `(G0, G1, G2)` runs `G0 * G1 * G2` programs in parallel. Each program:

1. Queries its coordinates via `tl.program_id(axis)`.
2. Computes pointer offsets based on its coordinates.
3. `tl.load()`s a block of data into registers.
4. Computes on the block (arithmetic, matmul, reductions).
5. `tl.store()`s results back.

### 3.2 Block Tensors

Triton's core abstraction. A block tensor is:
- **Shape**: a tuple of **compile-time** ints (except for block pointers, which read their shape from runtime `shape=` arg of `make_block_ptr`).
- **Dtype**: a single scalar dtype.
- **Layout**: chosen by compiler; transparent to the user.

Shape dims need not be powers of 2 *in principle*, but reductions (`tl.sum`, `tl.max`), `tl.dot`, and certain ops strongly prefer pow-2 dims. For safety, **use pow-2 BLOCK sizes and mask out OOB elements**.

### 3.3 Program ID & Num Programs

```python
pid  = tl.program_id(axis)     # int32, axis in {0, 1, 2}
n    = tl.num_programs(axis)   # total programs along axis
```

Typical 1D grid:
```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Typical 2D grid (matmul):
```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

### 3.4 Grid Launch Syntax

```python
kernel[grid](arg1, arg2, ..., META_PARAM=value, num_warps=4, num_stages=3)
```

- `grid` is either a `tuple[int, ...]` or a `Callable[[dict], tuple[int, ...]]` (required for autotune, because BLOCK sizes that define the grid are meta-params).
- Meta-params (anything declared `: tl.constexpr` in the kernel signature) must be passed as kwargs.
- `num_warps`, `num_stages`, `num_ctas`, `maxnreg` can also be passed as kwargs at launch (in the non-autotune path).

```python
# Static grid
kernel[(triton.cdiv(N, 1024), )](x, y, output, N, BLOCK_SIZE=1024)

# Grid as callable (needed when BLOCK is autotuned)
grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
kernel[grid](x, y, output, N)
```

### 3.5 Kernel Parameter Rules

| Annotation | Meaning | Specialization |
|------------|---------|----------------|
| no annotation | runtime value | specialized on alignment (passed as-is) |
| `: tl.constexpr` | compile-time constant | triggers recompile when value changes |
| `: tl.const` | read-only pointer (equivalent to `const T*`) | alignment only |
| `arg=default` | normal Python default | applied if kwarg omitted |

`tl.constexpr` is used for:
- Shape-like values (`BLOCK_SIZE`, `HEAD_DIM`, `GROUP_SIZE_M`)
- Booleans that select code paths (`CAUSAL`, `IS_TRAINING`)
- Dtype selectors (`OUT_DTYPE: tl.constexpr`)

**CRITICAL**: anything that feeds `shape=` in `tl.arange`, `tl.zeros`, `tl.full`, or `make_block_ptr` block_shape must be `constexpr`.

### 3.6 tl.assume — Integer Bound Hints

```python
tl.assume(pid_m >= 0)
tl.assume(stride_am > 0)
```

Hints the compiler about integer relationships. Helps the backend fold address calculations and avoid overflow checks. Commonly used in matmul for `pid_m/n >= 0` and `stride* > 0`.

---

## 4. Pointer & Offset Arithmetic

### 4.1 The Pointer-Arithmetic Model

Kernel arguments that are `torch.Tensor`s become **base pointers** (the kernel receives `.data_ptr()`). You compute per-element offsets by building an **offset tensor** and adding it to the base pointer. The resulting **tensor of pointers** is passed to `tl.load` / `tl.store`.

```python
# 1D: load BLOCK_SIZE contiguous elements
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask    = offsets < n_elements
x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

```python
# 2D: explicit row/col offsets with strides
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # [BLOCK_M]
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # [BLOCK_N]
offs_k = tl.arange(0, BLOCK_K)                             # [BLOCK_K]
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
```

### 4.2 Broadcasting (NumPy-style)

- `[:, None]` → adds a new axis of size 1 on the right of axis 0.
- `[None, :]` → adds a new axis of size 1 on the left of axis 0.
- Arithmetic between shapes broadcasts the same way NumPy does.

### 4.3 Masking OOB Accesses

Always pass a `mask=` on loads/stores when BLOCK may exceed the remaining problem size:

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)     # other= fills OOB
tl.store(out_ptr + offsets, y, mask=mask)              # masked stores silently skip
```

`other` values (for `tl.load`) — default is an undefined value:
- `0.0` for sum/accumulate
- `-float('inf')` for max/softmax
- `float('inf')` for min
- `1.0` for multiplicative identity (e.g. RMSNorm weight)

### 4.4 Block Pointers (`make_block_ptr`) — Structured Pointer Arithmetic

For 2D+ tensors with clean strides, prefer block pointers: they encode shape/strides/offsets once, and the compiler can emit TMA or swizzled loads.

```python
A_block = tl.make_block_ptr(
    base=A_ptr,
    shape=(M, K),                # parent tensor shape (runtime)
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K),    # constexpr shapes
    order=(1, 0),                # fastest-varying axis last; (1,0) for row-major 2D
)
a = tl.load(A_block, boundary_check=(0, 1), padding_option="zero")

# Advance the block pointer along the K axis each iteration
A_block = tl.advance(A_block, (0, BLOCK_K))
```

- `order` lists axes from slowest to fastest varying; `(1, 0)` means axis 1 is fastest (row-major inner dim).
- `boundary_check=(0, 1)` enables bounds checking on axes 0 and 1.
- `padding_option` ∈ `{"", "zero", "nan"}`; `""` = undefined OOB.
- `tl.advance(block_ptr, offsets)` returns a **new** block pointer; you must reassign it.

### 4.5 Tensor Descriptors (TMA, Hopper / Blackwell)

On NVIDIA sm_90+ with TMA, `make_tensor_descriptor` creates hardware TMA descriptors for 2D–5D tensors:

```python
desc = tl.make_tensor_descriptor(
    base=in_out_ptr,
    shape=[M, N],
    strides=[N, 1],                     # last dim must be contiguous (stride 1)
    block_shape=[M_BLOCK, N_BLOCK],     # constexpr
    padding_option="zero",              # "zero" | "nan"
)
value = desc.load([m_offset, n_offset])
desc.store([m_offset, n_offset], tl.abs(value))
```

Host-side `TensorDescriptor` (from `triton.tools.tensor_descriptor`) is used for persistent matmul + warp specialization. TMA requires a pinned global allocator:

```python
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)
```

TMA constraints:
- Base pointer must be 16-byte aligned.
- Leading strides must be multiples of 16 bytes.
- Last dim must be contiguous (stride 1).
- 2–5 dimensions supported.

---

## 5. Memory Operations

### 5.1 tl.load()

```python
tl.load(pointer, mask=None, other=None, boundary_check=(), padding_option="",
        cache_modifier="", eviction_policy="", volatile=False)
```

Three modes:

| Mode | pointer | mask | other | boundary_check / padding_option |
|------|---------|------|-------|--------------------------------|
| Scalar | single ptr | scalar bool | scalar | empty |
| Tensor-of-ptrs | tensor of ptrs | bool tensor | broadcasts | empty |
| Block pointer | `make_block_ptr` | **None** | **None** | used for OOB control |

NVIDIA cache modifiers (via `cache_modifier`):

| Value | Meaning |
|-------|---------|
| `""` | default |
| `".ca"` | cache at all levels (L1+L2) |
| `".cg"` | cache at global level (L2 only, bypass L1) |
| `".cv"` | don't cache; refetch (treat as volatile global) |

Eviction policies (via `eviction_policy`):
- `"evict_first"` — evict soon (one-shot reads)
- `"evict_last"` — keep long (reuse-heavy reads like K,V in attention)

Padding options (block ptr / tensor desc only):
- `""` → undefined OOB
- `"zero"` → OOB = 0
- `"nan"` → OOB = NaN

### 5.2 tl.store()

```python
tl.store(pointer, value, mask=None, boundary_check=(),
         cache_modifier="", eviction_policy="")
```

NVIDIA store cache modifiers:
- `""` default
- `".wb"` write-back all coherent levels
- `".cg"` cache global
- `".cs"` cache streaming
- `".wt"` write-through

Masked stores silently skip OOB elements.

`value` is auto-broadcast to `pointer.shape` and auto-cast to `pointer.dtype.element_ty`.

### 5.3 tl.gather() / tl.histogram()

```python
result = tl.gather(src, index, axis)          # src[..., index[i], ...] along axis
hist   = tl.histogram(input, num_bins, mask=None)
```

`gather` is 1-axis dynamic indexing; for multi-axis, compute flat indices manually.

### 5.4 Atomic Operations

Return the **old** value. All accept `mask=`, `sem=`, `scope=`:

```python
tl.atomic_cas(ptr, cmp, val, sem=None, scope=None)
tl.atomic_xchg(ptr, val, mask=None, sem=None, scope=None)
tl.atomic_add(ptr, val, mask=None, sem=None, scope=None)
tl.atomic_max(ptr, val, mask=None, sem=None, scope=None)
tl.atomic_min(ptr, val, mask=None, sem=None, scope=None)
tl.atomic_and(ptr, val, mask=None, sem=None, scope=None)
tl.atomic_or(ptr, val, mask=None, sem=None, scope=None)
tl.atomic_xor(ptr, val, mask=None, sem=None, scope=None)
```

- `sem` ∈ `"acquire" | "release" | "acq_rel" | "relaxed"` (default `"acq_rel"`)
- `scope` ∈ `"gpu" | "cta" | "sys"` (default `"gpu"`)

Typical uses: histogram (`atomic_add` on counter), reduction-across-CTAs, spin-locks (`atomic_cas` + `atomic_xchg`).

Spinlock pattern (from `05-layer-norm`):
```python
while tl.atomic_cas(Lock, 0, 1) == 1:
    pass
# ... critical section ...
tl.debug_barrier()   # ensure writes visible before unlocking
tl.atomic_xchg(Lock, 0)
```

---

## 6. Tile Creation

```python
tl.arange(start, end)                # int32 tensor [start, start+1, ..., end-1]; end-start must be const and pow-2 for reductions
tl.zeros(shape, dtype)               # shape is tuple/list of constexpr
tl.zeros_like(tensor)
tl.full(shape, value, dtype)         # constant fill
```

Constraint: `tl.arange(start, end)` requires `start` and `end` to be compile-time constants, and `end - start` must be a power of 2.

```python
# Common patterns
offs_m = tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
m_i    = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')    # running max
ones   = tl.full((BLOCK,), 1.0, dtype=tl.float16)
```

---

## 7. Shape & View Operations

```python
tl.reshape(x, shape, can_reorder=False)
tl.view(x, shape)                    # cheaper reshape (no data movement guaranteed)
tl.broadcast(x, y)                   # align two tensors to common shape
tl.broadcast_to(x, shape)            # explicit broadcast
tl.trans(x, *dims)                   # transpose (aliases: x.T for 2D)
tl.permute(x, *dims)                 # reorder axes
tl.expand_dims(x, axis)              # insert size-1 dim
tl.cat(a, b, can_reorder=False)      # concat along axis 0
tl.join(a, b)                        # stack into new innermost axis of size 2
tl.split(a)                          # inverse of join: (a_even, a_odd) along innermost
tl.interleave(a, b)                  # zip along innermost axis
tl.flip(x, dim=None)                 # reverse along dim
tl.ravel(x)                          # flatten to 1D
```

Method forms on tensors: `x.reshape(...)`, `x.trans(...)`, `x.permute(...)`, `x.broadcast_to(...)`, `x.expand_dims(...)`, `x.T` (for 2D transpose), `x.view(...)`.

---

## 8. Arithmetic & Math

### 8.1 Operator Overloads

```python
c = a + b     c = a - b     c = a * b     c = a / b     # true division → fp
c = a // b    c = a % b     c = a ** b
c = a & b     c = a | b     c = a ^ b     c = ~a        # bitwise
c = -a        c = a << b    c = a >> b
```

Python scalars auto-promote: `x * 2`, `x + 0.5`.

Comparison operators (return `tl.int1` block):
```python
mask = x > y;  x >= y;  x < y;  x <= y;  x == y;  x != y
```

Logical (word-level, return bool):
```python
x.logical_and(y);  x.logical_or(y);  x.__not__()
```

### 8.2 Explicit Arithmetic

```python
tl.add(x, y);  tl.sub(x, y);  tl.mul(x, y)              # from core; rarely needed
tl.maximum(x, y, propagate_nan=PropagateNan.NONE)       # element-wise max
tl.minimum(x, y, propagate_nan=PropagateNan.NONE)       # element-wise min
tl.clamp(x, min_val, max_val, propagate_nan=...)
tl.where(cond, x, y)                                     # element-wise select
```

`PropagateNan`:
- `PropagateNan.NONE` — NaN-tolerant min/max (default; IEEE behaves min/max(nan, x) = x)
- `PropagateNan.ALL` — if either is NaN, result is NaN

### 8.3 Math (tl.math.* — prefer these over libdevice)

Fast approximate (xxf intrinsics on NVIDIA):
```python
tl.exp(x)        tl.log(x)         tl.cos(x)         tl.sin(x)
tl.exp2(x)       tl.log2(x)                                  # faster; use in softmax via log2(e)
tl.sqrt(x)       tl.rsqrt(x)                                  # approximate
tl.abs(x)        tl.floor(x)       tl.ceil(x)
tl.erf(x)        tl.fma(x, y, z)   tl.umulhi(x, y)            # high bits of unsigned mul
tl.fdiv(x, y, ieee_rounding=False)                            # fast divide
```

Precise (IEEE 754):
```python
tl.sqrt_rn(x)    # precise sqrt, round-to-nearest-even (fp32 only)
tl.div_rn(x, y)  # precise divide (fp32 only)
```

Dtype constraints: most math fns accept only fp32/fp64. **Cast fp16/bf16 → fp32 before calling `tl.exp`, `tl.log`, etc.**

Softmax trick:
```python
# Use exp2 + qk_scale * log2(e) rather than exp — one FMA saved per element
qk_scale = sm_scale * 1.44269504    # 1/ln(2)
p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
```

---

## 9. Matrix Multiply — tl.dot()

```python
result = tl.dot(input, other, acc=None,
                input_precision=None,     # "tf32" | "tf32x3" | "ieee" (NV); "ieee" | "tf32" (AMD)
                allow_tf32=None,          # deprecated
                max_num_imprecise_acc=None,
                out_dtype=tl.float32)
```

- `input`: `[M, K]` or `[B, M, K]`
- `other`: `[K, N]` or `[B, K, N]`
- `acc` (optional): same shape as result; if given, `result = input @ other + acc`
- Returns `[M, N]` or `[B, M, N]`

Supported dtype combinations (NVIDIA):

| input, other | accumulator | Notes |
|--------------|-------------|-------|
| `float16` | `float16` or `float32` | fp16 acc = fastest but less accurate |
| `bfloat16` | `float32` | training default |
| `float32` | `float32` | uses TF32 cores unless `input_precision="ieee"` |
| `float8_e5m2` | `float32` (or `float16` in some backends) | FP8 |
| `float8_e4m3fn` | `float32` | FP8 |
| `int8` | `int32` | INT8 GEMM |

Standard GEMM loop pattern:
```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
c = accumulator.to(tl.float16)   # downcast before store
```

### 9.1 tl.dot_scaled() — Microscaling FP8/FP4

For MX-format fp4/fp8 GEMM (sm_100+, native; older archs: software emulation).

```python
tl.dot_scaled(lhs, lhs_scale, lhs_format,     # formats: "e2m1" | "e4m3" | "e5m2" | "bf16" | "fp16"
              rhs, rhs_scale, rhs_format,
              acc=None, fast_math=False,
              lhs_k_pack=True, rhs_k_pack=True,
              out_dtype=tl.float32)
```

Scale tensors are `e8m0` (uint8). Group size is 32 elements for e8m0 scales.

---

## 10. Reductions

```python
tl.sum(x, axis=None, keep_dims=False, dtype=None)
tl.max(x, axis=None, return_indices=False,
       return_indices_tie_break_left=True, keep_dims=False)
tl.min(x, axis=None, return_indices=False,
       return_indices_tie_break_left=True, keep_dims=False)
tl.argmax(x, axis, tie_break_left=True, keep_dims=False)
tl.argmin(x, axis, tie_break_left=True, keep_dims=False)
tl.xor_sum(x, axis=None, keep_dims=False)      # integer xor reduction
tl.reduce_or(x, axis, keep_dims=False)         # integer or reduction
```

Rules:
- `axis=None` reduces over all dimensions.
- `keep_dims=True` keeps a size-1 dim so the result can broadcast back.
- Integer dtypes <32-bit are promoted to `int32`/`uint32` automatically (to avoid overflow).
- `bfloat16` is promoted to `float32` before max/min/sum/scan.

### 10.1 Custom Reductions with tl.reduce()

```python
@triton.jit
def _my_combine(a, b):
    return a + b

y = tl.reduce(x, axis, _my_combine, keep_dims=False)

# Multi-output (e.g. argmin): input is a tuple, combine_fn takes 4 args, returns 2-tuple
@triton.jit
def _argmin_combine(v1, i1, v2, i2):
    lt = v1 < v2
    return tl.where(lt, v1, v2), tl.where(lt, i1, i2)

min_val, min_idx = tl.reduce((vals, idxs), axis=0, combine_fn=_argmin_combine)
```

The combine function must be a `@triton.jit` function (or `@jit` module-level function).

### 10.2 Scans (Prefix)

```python
tl.cumsum(x, axis=0, reverse=False, dtype=None)
tl.cumprod(x, axis=0, reverse=False)
tl.associative_scan(input, axis, combine_fn, reverse=False)   # custom scan
```

---

## 11. Control Flow

### 11.1 Python for / if — Compile-time Specialization

When the loop bounds / condition are `constexpr`-computable, Triton unrolls/specializes:

```python
for j in tl.static_range(0, 8):             # fully unrolled
    ...

if CAUSAL:                                   # CAUSAL is tl.constexpr → branch eliminated
    ...
```

### 11.2 tl.range — Runtime Loops with Attributes

Preferred for loops that must stay as loops in IR (not unrolled), with optional pipelining / warp specialization:

```python
for k in tl.range(0, K, BLOCK_K,
                  num_stages=3,                # SW-pipeline depth (default 3)
                  loop_unroll_factor=2,         # IR-level unroll factor
                  disallow_acc_multi_buffer=False,
                  flatten=False,                # flatten nested loops into one
                  warp_specialize=False,        # enable warp specialization (Blackwell)
                  disable_licm=False):
    ...
```

Pipelining with `num_stages` overlaps load + compute + store across iterations. For matmul loops, typical values are 2–4.

### 11.3 tl.static_range — Compile-time for Loop

Like Python `range`, but guarantees compile-time unrolling. All bounds must be `constexpr`.

```python
for i in tl.static_range(0, N_STAGES):       # N_STAGES is constexpr
    ...
```

### 11.4 While Loops + tl.condition

```python
while tl.condition(not_done, disable_licm=False):
    ...
```

Used for atomic spinlocks and iterative convergence loops.

---

## 12. Debug & Introspection

```python
tl.static_print(*values, sep=" ", end="\n", file=None, flush=False)
tl.static_assert(cond, msg="")              # compile-time assert
tl.device_print(prefix, *args, hex=False)    # printf — significant overhead
tl.device_assert(cond, msg="", mask=None)   # runtime assert — significant overhead
tl.debug_barrier()                           # __syncthreads()
```

`tl.device_print` follows printf semantics but multi-program output interleaves; use at low block counts or `num_warps=1` for readable logs.

---

## 13. Compiler Hints

```python
tl.assume(cond)                                        # integer bound hints
tl.multiple_of(x, divisor)                             # x divisible by divisor
tl.max_contiguous(x, size)                             # x has `size` contiguous elements
tl.max_constancy(x, values)                            # adjacent elements equal in groups
```

Commonly used in persistent matmul for address-math optimization:
```python
offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
```

`tl.assume` for integer analysis:
```python
tl.assume(pid_m >= 0);  tl.assume(pid_n >= 0)
tl.assume(stride_am > 0);  tl.assume(stride_ak > 0)
```

---

## 14. Randomness (Philox PRNG)

```python
tl.rand(seed, offsets)         # fp32 uniform in [0, 1)
tl.randn(seed, offsets)        # fp32 normal (Box-Muller)
tl.randint(seed, offsets)      # int32 uniform
tl.rand4x(seed, offsets)       # 4 uniform values per offset (faster)
tl.randn4x(seed, offsets)      # 4 normal values per offset
tl.randint4x(seed, offsets)    # 4 int values per offset
```

Seeded dropout pattern:
```python
random = tl.rand(seed, offsets)
keep   = random > p
out    = tl.where(keep, x / (1 - p), 0.0)
```

---

## 15. @triton.jit — The JIT Decorator

```python
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    ...

# With options
@triton.jit(do_not_specialize=("stride_am",),              # don't specialize on this arg's alignment
            do_not_specialize_on_alignment=("N",),
            debug=False,                                     # insert device assertions
            noinline=False,                                  # prevent inlining in @jit callers
            launch_metadata=my_meta_fn)                      # hook for profiler metadata
def kernel(...): ...
```

Accepted inside `@triton.jit`:
- Python builtins: `float`, `int`, `getattr`, `isinstance`, `len`, `list`, `max`, `min`, `print`, `range`.
- `copy`, `math` module.
- Anything under `triton.language.*`.
- Other `@triton.jit` functions (inlined or non-inlined based on `noinline`).

Kernel launch syntax:
```python
kernel[grid](*args, **kwargs)           # calls .run() with warmup=False
kernel.warmup(grid=(1,), *args, **kwargs)   # compiles without executing
```

Specialization: Triton recompiles for new `constexpr` values and for pointer-alignment changes (16/8/4-byte). Shapes passed as runtime ints typically do **not** recompile; shapes passed as `constexpr` **do**.

---

## 16. @triton.autotune — Configuration Search

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_warps=4, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_warps=4, num_stages=4),
        # ...
    ],
    key=['M', 'N', 'K'],                    # re-benchmark when any of these changes
    prune_configs_by={                       # optional: prune before benchmarking
        'early_config_prune': prune_fn,      # List[Config], named_args, **kwargs -> List[Config]
        'perf_model': perf_fn,               # Config -> est. runtime
        'top_k': 10,
    },
    reset_to_zero=['c_ptr'],                # zero these tensors before each trial
    restore_value=['x_ptr'],                # save+restore these tensors per trial
    pre_hook=None, post_hook=None,           # custom setup / teardown
    cache_results=False,                     # write autotune results to disk
    # Deprecated: warmup, rep, use_cuda_graph, do_bench — use TRITON_PRINT_AUTOTUNING=1
)
@triton.jit
def kernel(...):
    ...
```

### 16.1 triton.Config

```python
Config(kwargs,                       # dict of meta-params for the kernel
       num_warps=4,                   # warps per CTA (1, 2, 4, 8, 16, 32)
       num_stages=3,                  # SW-pipeline depth for dot-feeding loads
       num_ctas=1,                    # CTAs per CGA (SM90+)
       maxnreg=None,                  # per-thread register cap (PTX .maxnreg)
       pre_hook=None,                 # fn(nargs) run before each bench (e.g. reshape TMA desc)
       ir_override=None)              # substitute custom *.ttgir / *.ptx / *.llir
```

Common `num_warps` / `num_stages` sweeps:
```python
configs = [
    triton.Config({'BLOCK': bs}, num_warps=nw, num_stages=ns)
    for bs in [64, 128, 256, 512]
    for nw in [2, 4, 8]
    for ns in [2, 3, 4]
]
```

### 16.2 best_config / Cached Result

After first launch with a given `key=`, the autotuner picks one `Config` and caches it. Read it via:

```python
cfg = _kernel_autotuned.best_config       # triton.Config or None
if cfg:
    bs = cfg.kwargs['BLOCK']
    nw = cfg.num_warps
    ns = cfg.num_stages
```

### 16.3 Disk Cache

Set `cache_results=True` or env `TRITON_AUTOTUNING_CACHE=1` to persist results across sessions. Cache key hashes the source, Triton commit, backend target, env vars, `key=` values, and config list.

### 16.4 Verbose Logging

```bash
TRITON_PRINT_AUTOTUNING=1 python run_bench.py ...
```

Prints each config tried and the winner for each key tuple.

---

## 17. @triton.heuristics — Derived Meta-params

For values that are pure functions of the inputs (no benchmarking needed):

```python
@triton.heuristics(values={
    'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_cols']),
})
@triton.jit
def kernel(x_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    ...
```

The heuristic runs at launch time; result becomes part of the specialization key.

---

## 18. TileBench Integration — impl_triton.py Contract

Each `benchmarks/operators/<op>/impl_triton.py` must export:

```python
def run(*tensor_inputs, autotune: bool = False, **kwargs) -> torch.Tensor:
    """Execute the operator. When autotune=True, use the autotuned kernel."""
    ...

def get_last_config() -> dict | None:
    """Return the last-used Config's kwargs as a flat dict, or None."""
    ...
```

### 18.1 Canonical Template

```python
import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def _op_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = ...                                         # the operator
    tl.store(output_ptr + offsets, y, mask=mask)


# Autotuned wrapper — attached via functional call, NOT the decorator, so the
# non-autotune path still has a fast `_op_kernel` to call directly.
_op_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [256, 512, 1024, 2048, 4096]
        for nw in [4, 8, 16]
        for ns in [2, 3]
    ],
    key=["n_elements"],
)(_op_kernel)


def run(x: torch.Tensor, autotune: bool = False) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)

    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _op_kernel_autotuned[grid](x, output, n_elements)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _op_kernel[grid](
            x, output, n_elements,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return output


def get_last_config() -> dict | None:
    cfg = _op_kernel_autotuned.best_config   # triton.Config or None
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
```

### 18.2 Why the Dual-Kernel Pattern

- The raw `_op_kernel` is what the benchmark engine calls when `autotune=False`. Its config comes from `_DEFAULT_CONFIG`.
- `_op_kernel_autotuned = triton.autotune(...)(_op_kernel)` creates an `Autotuner` wrapper that picks the best config per `key` tuple.
- `best_config` lives **on** the `Autotuner` (not module state), so `get_last_config()` reads it directly — no `global _last_config` needed.
- TileBench's engine tries `autotune=True` first and records the resulting `best_config`; then separately runs `autotune=False` to measure latency with `_DEFAULT_CONFIG`.

### 18.3 What NOT to do

```python
# WRONG — using the decorator form hides _op_kernel
@triton.autotune(configs=..., key=...)
@triton.jit
def _op_kernel(...):       # now there is no raw kernel to call with an explicit BLOCK

# WRONG — writing to module-level state in run()
_last_config = None        # NOT NEEDED for Triton; best_config is on the Autotuner
def run(x, autotune=False):
    global _last_config    # anti-pattern
```

### 18.4 Common `key=` Choices

| Operator class | `key=` |
|----------------|--------|
| Element-wise (`mul2`, `relu`, `swiglu`) | `["n_elements"]` |
| Row-wise (`rmsnorm`, `layernorm`, `softmax`) | `["n_cols"]` or `["N_SIZE"]` |
| Matmul | `["M", "N", "K"]` |
| Attention | `["SEQLEN", "DIM"]` or `["N_CTX", "HEAD_DIM"]` |
| Conv | `["N", "C", "H", "W", "K"]` |

`key=[...]` only needs the args whose *value* (not just dtype/shape) changes the optimal config. Shape-like ints are usually enough.

---

## 19. Common Kernel Patterns

### Pattern A: Element-wise (mul2, relu, vector_add, swiglu, dropout)

```python
@triton.jit
def _kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = x * 2                         # or any element-wise expression
    tl.store(out_ptr + offs, y, mask=mask)

# grid = (cdiv(n, BLOCK),)
```

Variants:
- **ReLU**: `y = tl.where(x >= 0, x, 0.0)` — or `y = tl.maximum(x, 0.0)`
- **SwiGLU**: `y = x * (1.0 / (1.0 + tl.exp(-gate.to(tl.float32)))).to(x.dtype)`
- **Dropout**: `y = tl.where(tl.rand(seed, offs) > p, x / (1 - p), 0.0)`

### Pattern B: Row-wise single-pass (softmax with row-in-one-tile)

```python
@triton.jit
def _kernel(out_ptr, in_ptr, in_stride, out_stride, n_rows, n_cols,
            BLOCK: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step  = tl.num_programs(0)
    for row in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_ptr = in_ptr + row * in_stride
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(row_ptr + cols, mask=mask, other=-float('inf'))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        y = num / den
        tl.store(out_ptr + row * out_stride + cols, y, mask=mask)

# Host:
BLOCK = triton.next_power_of_2(n_cols)
grid  = (min(NUM_SM * occupancy, n_rows), )     # persistent programs
```

### Pattern C: Row-wise two-pass (rmsnorm, layernorm, l2_norm)

Row too large for a single tile — loop over `BLOCK`-sized chunks twice:

```python
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride_row, N_SIZE, eps,
                    BLOCK_N_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr     = x_ptr   + pid * stride_row
    out_row_ptr = out_ptr + pid * stride_row
    block_N = tl.arange(0, BLOCK_N_SIZE)

    # Pass 1: sum of squares (accumulate in fp32)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offs = n_start + block_N
        mask = offs < N_SIZE
        x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var += x * x
    rstd = tl.math.rsqrt(tl.sum(var, axis=0) / N_SIZE + eps)

    # Pass 2: normalize and scale
    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offs = n_start + block_N
        mask = offs < N_SIZE
        x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_row_ptr + offs, x * rstd * w, mask=mask)
```

### Pattern D: Matmul (BLOCK_M × BLOCK_K × BLOCK_N tiles with GROUP_SIZE_M grouping)

```python
@triton.autotune(configs=..., key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_SIZE_M: tl.constexpr):
    # L2-cache-friendly block ordering
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0);  tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0);  tl.assume(stride_bk > 0)

    # Pointer setup
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # K loop
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)

    # Store
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

Grid: `(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), )`.

### Pattern E: Flash Attention (online softmax + streaming K/V)

```python
@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, L, O,
                stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
                stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
                stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
                stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
                BS, HEAD, SEQLEN,
                BLOCK_M: tl.constexpr, DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                IS_CAUSAL: tl.constexpr):
    start_m     = tl.program_id(0)
    off_bs_head = tl.program_id(1)
    qkv_base    = off_bs_head * stride_q_head

    Q_block = tl.make_block_ptr(base=Q + qkv_base, shape=(SEQLEN, DIM),
                                strides=(stride_q_seqlen, stride_q_dim),
                                offsets=(start_m * BLOCK_M, 0),
                                block_shape=(BLOCK_M, DIM), order=(1, 0))
    K_block = tl.make_block_ptr(base=K + qkv_base, shape=(DIM, SEQLEN),
                                strides=(stride_k_dim, stride_k_seqlen),
                                offsets=(0, 0),
                                block_shape=(DIM, BLOCK_N), order=(0, 1))
    V_block = tl.make_block_ptr(base=V + qkv_base, shape=(SEQLEN, DIM),
                                strides=(stride_v_seqlen, stride_v_dim),
                                offsets=(0, 0),
                                block_shape=(BLOCK_N, DIM), order=(1, 0))

    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    m_i   = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i   = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc   = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504   # 1 / ln(2)
    q = tl.load(Q_block)
    q = (q * qk_scale).to(tl.float16)

    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block)
        v = tl.load(V_block)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float('-inf'))
        qk += tl.dot(q, k)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p     = tl.math.exp2(qk - m_new[:, None])
        acc   = acc * alpha[:, None]
        acc  += tl.dot(p.to(tl.float16), v)
        l_i   = l_i * alpha + tl.sum(p, 1)
        m_i   = m_new

        K_block = tl.advance(K_block, (0, BLOCK_N))
        V_block = tl.advance(V_block, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(L + off_bs_head * SEQLEN + off_m, m_i + tl.math.log2(l_i))

    O_block = tl.make_block_ptr(base=O + qkv_base, shape=(SEQLEN, DIM),
                                strides=(stride_o_seqlen, stride_o_dim),
                                offsets=(start_m * BLOCK_M, 0),
                                block_shape=(BLOCK_M, DIM), order=(1, 0))
    tl.store(O_block, acc.to(tl.float16))
```

Key ideas:
- Running max `m_i`, running sum `l_i`, running accumulator `acc`.
- Rescale `acc *= exp2(m_old - m_new)` before adding new `p @ v`.
- Convert to `log2`-scale softmax for 1 FMA/element instead of 2.
- Use `make_block_ptr` + `tl.advance` for clean K/V streaming.

### Pattern F: Persistent Kernel (one program handles many output tiles)

Launch fewer programs than output tiles, loop inside:

```python
# Grid: (min(NUM_SM, num_output_tiles), )
@triton.jit
def _persistent_kernel(..., num_tiles, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for tile in tl.range(start_pid, num_tiles, num_progs, num_stages=3):
        pid_m = tile // tl.cdiv(N, BLOCK_N)
        pid_n = tile %  tl.cdiv(N, BLOCK_N)
        # ... compute tile, store ...
```

Amortizes launch overhead and enables TMA + warp specialization on Hopper/Blackwell.

### Pattern G: Spin-Locked Parallel Reduction (layernorm backward dw/db)

```python
# Multiple rows write partial sums into GROUP_SIZE_M shared buffers under a per-group lock.
lock_id = row % GROUP_SIZE_M
Lock   = lock_ptr  + lock_id
Count  = Lock + GROUP_SIZE_M

while tl.atomic_cas(Lock, 0, 1) == 1:
    pass                                     # spin until acquired
count = tl.load(Count)
if count == 0:
    tl.atomic_xchg(Count, 1)
else:
    partial_dw += tl.load(DW, mask=mask)     # accumulate
    partial_db += tl.load(DB, mask=mask)
tl.store(DW, partial_dw, mask=mask)
tl.store(DB, partial_db, mask=mask)
tl.debug_barrier()                           # make writes visible
tl.atomic_xchg(Lock, 0)                      # release
```

### Pattern H: Using `tl.make_tensor_descriptor` (TMA, Hopper+)

```python
@triton.jit
def _kernel(in_out_ptr, M, N,
            M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(in_out_ptr,
                                     shape=[M, N],
                                     strides=[N, 1],
                                     block_shape=[M_BLOCK, N_BLOCK])
    m_off = tl.program_id(0) * M_BLOCK
    n_off = tl.program_id(1) * N_BLOCK
    value = desc.load([m_off, n_off])
    desc.store([m_off, n_off], tl.abs(value))

# Host side
def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)
```

---

## 20. Performance Methodology

### 20.1 Occupancy / Resource Tradeoffs

- `num_warps` ↑ → more threads/CTA, more parallel work, fewer live registers per thread.
- `num_stages` ↑ → deeper pipelining, more shared memory for in-flight buffers, better latency hiding, but caps occupancy.
- `num_ctas` > 1 → thread-block clusters (SM90+); enables CGA-level DSMem sharing.
- `maxnreg` → caps register pressure to boost occupancy (rarely needed; autotuner usually handles).

Rule of thumb: for memory-bound ops, prefer large `BLOCK_SIZE` + `num_warps=8/16` to saturate HBM. For compute-bound GEMMs, sweep `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_SIZE_M)` jointly.

### 20.2 Saturating HBM Bandwidth

- Work per program × memory bandwidth per program ≈ peak HBM (8 TB/s on B200).
- Flush L2 between runs (TileBench's `--flush-l2` or `triton.runtime.driver.active.clear_cache(cache)`).
- Sweep problem sizes past L2 (~50 MB on B200) to measure HBM, not cache.

### 20.3 Compute-Bound Checks

- Use TF32 / FP16 / BF16 / FP8 for GEMM. `input_precision="tf32"` on fp32 tensors.
- Verify arithmetic intensity: `flops / bytes > peak_tflops / peak_bw`.
- For MMA-heavy loops, ensure `num_stages ≥ 2` to hide mainloop loads.

### 20.4 triton.testing.do_bench

```python
ms = triton.testing.do_bench(lambda: kernel[grid](...),
                              warmup=25,       # ms of warmup
                              rep=100,         # ms of timed runs
                              quantiles=[0.5, 0.2, 0.8])   # returns [median, p20, p80]
# grad-clearing for autograd: grad_to_none=[tensor]
```

`do_bench_cudagraph` captures a graph of `rep / est_ms` kernel calls and replays. Use when launch overhead dominates.

TileBench's `core/timer.py` uses Proton (`triton.profiler as proton`) for more accurate latency measurement.

### 20.5 Common `perf_report` Pattern (from tutorials)

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], x_vals=[2**i for i in range(12, 28)],
        x_log=True, line_arg='provider',
        line_vals=['triton', 'torch'], line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s', plot_name='vector-add',
        args={},
    ))
def benchmark(size, provider):
    ...
benchmark.run(print_data=True, show_plots=True)
```

---

## 21. Debugging Checklist

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `CompileTimeAssertionFailure` | `tl.static_assert` failed | Inspect constexpr values; usually shape mismatch |
| `OutOfResources` during autotune | Config needs more regs/smem than available | Reduce `BLOCK_*`, `num_stages`, or add `maxnreg` |
| Runtime `CUDA error: misaligned address` | Bad strides / non-16B-aligned ptr with TMA | Make sure leading strides % 16 == 0 |
| All-zero output | Forgot `mask=` on store; or accumulator never updated | Check mask; print with `tl.device_print` |
| NaN output | Divide by zero in softmax (empty row); `tl.exp` on unbounded fp16 | Subtract max before exp; cast to fp32 first |
| Non-deterministic output | Race on atomic_add with FP (FP add non-associative) | Use fp32 for atomics; or restructure |
| Slow vs PyTorch | `BLOCK` too small; `num_stages=1`; wrong `order` on block_ptr | Autotune; check Proton/NSight; inspect TTIR |
| fp16 argmax wrong | fp16 can't represent integers > 2048 | Cast to fp32 on host before launch |

### 21.1 Dumping IRs

```bash
TRITON_CACHE_DIR=/tmp/triton_cache MLIR_ENABLE_DUMP=1 python ...
# Cached kernels live in $TRITON_CACHE_DIR with .ttir, .ttgir, .llir, .ptx files per compile key.
```

### 21.2 Interpreter Mode (pure Python, for debugging)

```bash
TRITON_INTERPRET=1 python my_script.py
```

Runs kernels under pure-Python emulation. Use `print()`, `pdb`, etc. Very slow; don't use for perf.

---

## 22. Critical Constraints & Gotchas

### 22.1 constexpr-ness of Shapes

`tl.arange`, `tl.zeros`, `tl.full`, `tl.dot` block dims, `make_block_ptr.block_shape` all require **compile-time constant** sizes. Declare every shape-parameter as `tl.constexpr` in the kernel signature.

### 22.2 `tl.advance` Returns a New Pointer

```python
# WRONG
tl.advance(K_block, (0, BLOCK_N))           # result discarded!

# RIGHT
K_block = tl.advance(K_block, (0, BLOCK_N))
```

Triton's `must_use_result` decorator on `advance` will emit a warning if you forget.

### 22.3 Arithmetic on Blocks Returns New Blocks (Immutable)

```python
# WRONG
acc += something                             # fine — this is re-binding
x_ptr += BLOCK                               # fine — x_ptr is reassigned

# But:
x[0] = 1                                     # NOT SUPPORTED — tiles are immutable
```

### 22.4 Always Mask or Pad Non-Pow-2 Trailing Dims

```python
cols = tl.arange(0, BLOCK)                   # BLOCK must be pow-2
mask = cols < n_cols                         # guard OOB on the actual problem size
x = tl.load(ptr + cols, mask=mask, other=0.0)
```

Pick `BLOCK = triton.next_power_of_2(n_cols)` on the host side.

### 22.5 Compute in fp32, Store in Source Dtype

```python
x = tl.load(x_ptr + ..., mask=...).to(tl.float32)   # upcast for accumulation
...
tl.store(out_ptr + ..., result.to(orig_dtype), mask=...)
# Or let tl.store auto-cast via the pointer's element_ty:
# tl.store(out_ptr + ..., result, mask=...)   # auto-casts if out_ptr is fp16*
```

### 22.6 bfloat16 Is Promoted Silently in Reductions

`tl.max`, `tl.min`, `tl.sum`, `tl.cumsum`, `tl.cumprod` on `bfloat16` → promoted to `float32` automatically. This is almost always what you want (precision), but it means the accumulator dtype is **not** `bfloat16`.

### 22.7 `num_stages` on Loops vs on Kernels

- Kernel-level `num_stages` (in `Config(num_stages=...)` or launch kwarg) only pipelines loads that feed `tl.dot`.
- Loop-level `tl.range(..., num_stages=N)` pipelines **most** loads in that loop.

For non-matmul loops (rmsnorm, softmax), pass `num_stages` via `tl.range` inside the kernel, or let `Config(num_stages=...)` govern a nested dot if present.

### 22.8 Host-side `triton.cdiv` vs In-Kernel `tl.cdiv`

```python
# Host
grid = (triton.cdiv(N, BLOCK), )

# In kernel
num_blocks = tl.cdiv(N, BLOCK)
```

Both accept runtime values and return the ceil-division.

### 22.9 `triton.next_power_of_2` Is Host-Only

```python
# Host side — compute BLOCK before launch
BLOCK = triton.next_power_of_2(n_cols)
```

There's no in-kernel equivalent; pre-compute on the host.

### 22.10 Tensor Pointers vs Block Pointers — Don't Mix

- Tensor-of-pointers style: `tl.load(ptr + offsets, mask=..., other=...)`
- Block pointer style: `tl.load(block_ptr, boundary_check=..., padding_option=...)`

**Never** pass `mask`/`other` with a block pointer, and **never** pass `boundary_check` with a plain tensor-of-pointers.

### 22.11 Float8 GEMM — `ct.mma`-Style Restrictions in Triton

- `tl.dot` with fp8 inputs typically requires fp32 accumulator.
- Input tensor layout for fp8 may require transposing on load (use `order=(0, 1)` vs `(1, 0)` on `make_block_ptr`).
- Blackwell supports non-transposed V for fp8 fmha; Hopper usually requires `.T` on V.

### 22.12 Kernel Must Not Mutate Input Tensors Unless Documented

TileBench's autotuner benchmarks each config by calling `run()` multiple times. If `run()` writes to an input tensor (in-place ops), use `restore_value=['x_ptr']` in `@triton.autotune` so the tensor is restored between trials.

---

## 23. TileBench-Specific Workflow

### 23.1 File Skeleton

```
benchmarks/operators/<op>/
├── config.yaml      # peak_tflops, bytes_expr, flops_expr, case_grid, verify
├── impl_torch.py    # reference: def run(*tensors) -> Tensor
├── impl_triton.py   # def run(*, autotune=False) -> Tensor; def get_last_config() -> dict|None
└── impl_cutile.py   # (cuTile counterpart)
```

And `data/tensors.py::GENERATORS['<op>']` must produce input tensors matching the `config.yaml` case grid.

### 23.2 Running & Profiling

```bash
# Single operator
PYTHONPATH=. python scripts/run_bench.py --operator rmsnorm \
  --warmup 30 --repeat 200 --use-cuda-graph --flush-l2

# All operators
PYTHONPATH=. python scripts/run_bench_all.py

# Visualize (roofline)
PYTHONPATH=. python scripts/visualize.py --operator rmsnorm --gpu B200
```

Output written to:
- `results/logs/time_measurement_logs/<op>_results.json`
- `results/logs/autotune_logs/<op>_autotune.json`
- `results/csv/<op>_summary.csv`

### 23.3 Engine Behavior

For each test case:
1. Generate inputs (via `data/tensors.py` generator).
2. Run `impl_torch.run(...)` for the reference output.
3. For each backend: call `run(*, autotune=True)`, then `get_last_config()`, record config.
4. Verify correctness against torch reference (per-dtype tolerance).
5. Call `run(*, autotune=False)` inside a timing loop (Proton-wrapped).

The autotune pass primes `best_config`; the timed pass uses `_DEFAULT_CONFIG`. If you want the timed path to use the autotuned config, set `_DEFAULT_CONFIG` to match or gate on a different flag.

### 23.4 config.yaml Example

```yaml
operator: rmsnorm
peak_tflops:
  fp16: 989
  bf16: 989
  fp32: 60.9
bytes_expr: "2 * batch * M * K * dtype_size + K * dtype_size"
flops_expr: "2 * batch * M * K"
verify:
  atol: 1.0e-2
  rtol: 1.0e-2
case_defaults:
  batch: 1
  M: 2048
case_grid:
  K:
    expr: "[512*i for i in range(1, 21)]"
  dtype: ["fp16", "bf16", "fp32"]
```

- `bytes_expr` / `flops_expr` can reference any numeric param in the case (`batch`, `M`, `K`, `dtype_size`, `n`, `math.*`).
- `peak_tflops` keys must cover every `dtype` in the grid.
- `verify.atol/rtol` are optional overrides; otherwise per-dtype defaults apply.

---

## 24. Quick Reference Tables

### 24.1 tl.load() Parameters

| Param | Tensor-of-ptrs | Block ptr | Default |
|-------|----------------|-----------|---------|
| `pointer` | required | required | — |
| `mask` | bool tensor | **None** | None |
| `other` | scalar/tensor | **None** | undefined |
| `boundary_check` | **empty** | tuple of axes | `()` |
| `padding_option` | **empty** | `""/"zero"/"nan"` | `""` |
| `cache_modifier` | `""/".ca"/".cg"/".cv"` | same | `""` |
| `eviction_policy` | `""/"evict_first"/"evict_last"` | same | `""` |
| `volatile` | bool | bool | False |

### 24.2 tl.store() Parameters

| Param | Tensor-of-ptrs | Block ptr | Default |
|-------|----------------|-----------|---------|
| `pointer` | required | required | — |
| `value` | required | required | — |
| `mask` | bool tensor | **None** | None |
| `boundary_check` | **empty** | tuple | `()` |
| `cache_modifier` | `""/".wb"/".cg"/".cs"/".wt"` | same | `""` |
| `eviction_policy` | `""/"evict_first"/"evict_last"` | same | `""` |

### 24.3 tl.dot() Dtype Matrix

| input, other | acc | Output (default) |
|--------------|-----|------------------|
| float16 | float16 | float16 (or float32) |
| float16 | float32 | float32 |
| bfloat16 | float32 | float32 |
| float32 (TF32) | float32 | float32 |
| float8_e5m2 | float32 | float32 |
| float8_e4m3fn | float32 | float32 |
| int8 | int32 | int32 |

### 24.4 `triton.Config` Fields

| Field | Type | Typical values |
|-------|------|----------------|
| `kwargs` | dict[str, Any] | `{'BLOCK_M': 128, ...}` |
| `num_warps` | int | 1, 2, 4, 8, 16, 32 |
| `num_stages` | int | 1, 2, 3, 4 |
| `num_ctas` | int | 1 (default); 2, 4, 8 on SM90+ |
| `maxnreg` | int or None | usually None |
| `pre_hook` | `fn(nargs)` or None | used for TMA desc block_shape reshape |
| `ir_override` | path or None | rarely |

### 24.5 @triton.autotune Args

| Arg | Type | Purpose |
|-----|------|---------|
| `configs` | `list[triton.Config]` | search space |
| `key` | `list[str]` | arg names that re-trigger tuning on value change |
| `prune_configs_by` | dict | optional `early_config_prune`, `perf_model`, `top_k` |
| `reset_to_zero` | `list[str]` | zero-out these tensors before each trial |
| `restore_value` | `list[str]` | snapshot+restore these tensors per trial |
| `pre_hook` / `post_hook` | callable | custom setup/teardown |
| `cache_results` | bool | persist tuning to disk |

### 24.6 `tl.range` Attributes

| Attribute | Type | Effect |
|-----------|------|--------|
| `num_stages` | int | SW-pipeline depth for loop-body loads |
| `loop_unroll_factor` | int (≥2) | IR-level unroll |
| `disallow_acc_multi_buffer` | bool | disable acc multi-buffering |
| `flatten` | bool | flatten loop nest |
| `warp_specialize` | bool | enable warp specialization (Blackwell matmul) |
| `disable_licm` | bool | no hoisting |

---

## 25. Cheatsheet — Minimal Kernels

### Vector Add

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) +
                              tl.load(y_ptr + offs, mask=mask),
             mask=mask)
```

### Fused Softmax (single-pass)

```python
@triton.jit
def softmax_kernel(out_ptr, in_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK); mask = cols < n_cols
    x = tl.load(in_ptr + row * row_stride + cols, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x); den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * row_stride + cols, num / den, mask=mask)
```

### RMSNorm (two-pass)

```python
@triton.jit
def rms_kernel(x_ptr, w_ptr, out_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row   = x_ptr   + row * stride
    out_row = out_ptr + row * stride
    block_N = tl.arange(0, BLOCK)

    var = tl.zeros((BLOCK,), tl.float32)
    for n0 in range(0, N, BLOCK):
        offs = n0 + block_N; mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        var += x * x
    rstd = tl.math.rsqrt(tl.sum(var, axis=0) / N + eps)

    for n0 in range(0, N, BLOCK):
        offs = n0 + block_N; mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_row + offs, x * rstd * w, mask=mask)
```

### Matmul with Grouped Scheduling

See Pattern D in §19 — use `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_SIZE_M=8, num_warps=8, num_stages=3` as a strong starting point for fp16 on Blackwell.

### Flash Attention Forward

See Pattern E in §19. Starting configs: `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2` (scale up for Hopper/Blackwell).

---

## 26. When cuTile and Triton Disagree

TileBench benchmarks both; when writing dual impls:

| Aspect | Triton | cuTile |
|--------|--------|--------|
| Loading | `tl.load(ptr + offs, mask=..., other=...)` | `ct.load(array, index, shape, padding_mode=...)` |
| Tile shape | Any positive int, pow-2 for reductions | **Must** be power of 2 |
| Dynamic indices | mask+load (flexible) | `ct.gather` / `ct.scatter` |
| Accumulator | `tl.zeros(..., dtype=tl.float32)` | `ct.full(..., 0.0, dtype=np.float32)` |
| Matmul | `tl.dot(a, b, acc)` | `ct.mma(a, b, acc)` / `ct.matmul(a, b)` |
| Autotune | `@triton.autotune` w/ `Config` | `ct_experimental.autotune_launch()` |
| Config readback | `kernel_autotuned.best_config` | module-level `_last_autotune_config` |
| Unsupported op | fall through (just raise if impossible) | `raise NotImplementedError` |

If Triton supports an op and cuTile doesn't, `impl_cutile.py::run` raises `NotImplementedError` and `get_last_config` returns `None`. TileBench's engine catches this and marks the backend as skipped for that case.

---

## 27. Environment Variables (Most Useful)

| Variable | Effect |
|----------|--------|
| `TRITON_INTERPRET=1` | Run kernels in pure Python (debugging) |
| `TRITON_PRINT_AUTOTUNING=1` | Log every config tried during autotune + winners |
| `TRITON_AUTOTUNING_CACHE=1` | Persist autotune results across sessions |
| `TRITON_CACHE_DIR=/path` | Where compiled kernels / IR are cached |
| `MLIR_ENABLE_DUMP=1` | Dump MLIR IR to cache dir |
| `TRITON_ALWAYS_COMPILE=1` | Force recompilation, ignore cache |
| `TRITON_DEBUG=1` | Insert device asserts; slower but catches OOB |
| `TRITON_F32_DEFAULT="ieee"` | Force IEEE f32 (disable TF32) for all `tl.dot` fp32 |

---

## 28. Further Reading

Within this repo:
- `lib/Triton/triton-3.6.0/python/tutorials/` — 11 canonical tutorials (vector-add, softmax, matmul, dropout, layernorm, fmha, extern fns, grouped GEMM, persistent matmul, block-scaled matmul, PDL)
- `lib/Triton/triton-3.6.0/python/triton/language/core.py` — Full in-kernel primitive source
- `lib/Triton/triton-3.6.0/python/triton/language/standard.py` — Higher-level helpers (sort, topk, sigmoid, softmax, argmax, etc.)
- `lib/Triton/triton-3.6.0/python/triton/language/math.py` — Math functions (exp, log, rsqrt, erf, ...)
- `lib/Triton/triton-3.6.0/python/triton/runtime/autotuner.py` — `Autotuner` + `Config` source
- `lib/Triton/triton-3.6.0/python/triton/runtime/jit.py` — `@triton.jit` + `JITFunction`
- `lib/Triton/triton-3.6.0/python/triton/testing.py` — `do_bench`, `perf_report`, `Benchmark`

Existing TileBench impls worth cribbing from:
- `benchmarks/operators/mul2/impl_triton.py` — minimal element-wise
- `benchmarks/operators/rmsnorm/impl_triton.py` — two-pass row-wise
- `benchmarks/operators/softmax/impl_triton.py` — single-pass softmax
- `benchmarks/operators/flash_attention/impl_triton.py` — FMHA fwd with block pointers
- `benchmarks/operators/matmul_int8/impl_triton.py` — INT8 GEMM
- `benchmarks/operators/streamk_matmul/impl_triton.py` — StreamK + persistent


---


# cuTile (cuda.tile) API Reference

The cuTile DSL is newer than Triton and likely sparse in your training data. Treat the API reference below as authoritative; do NOT invent attributes by analogy with Triton (e.g. `tl.range` has no `ct.range` equivalent --- use plain Python `for` loops). When writing `impl_cutile.py`, every `ct.*` symbol you use must appear in this reference.

**IMPORTANT — autotune sections of this reference do NOT apply.** The reference below discusses `CutileAutotuner`, `ct_experimental.autotune_launch`, and `ct.tune.exhaustive_search`. In THIS pipeline you must NOT use any of them — see the 'No autotune' rule in the TileBench framework conventions above. Use the reference for kernel-body syntax (ct.load, ct.store, ct.mma, ct.bid, padding_mode, etc.); ignore everything about tuners and search spaces.

# cuTile (cuda.tile) Comprehensive Programming Guide

The `cuda.tile` Python DSL compiles Python into optimized CUDA via MLIR, targeting NVIDIA GPUs (Blackwell B200 / sm_100). All computation operates on immutable, multi-dimensional **Tiles** with power-of-2 dimensions.

---

## 1. Import Boilerplate

Every `impl_cutile.py` MUST follow this structure:

```python
from types import SimpleNamespace
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_last_autotune_config: dict | None = None
```

- `ct_experimental` is optional; always guard with try/except
- `_last_autotune_config` is module-level state for the engine to read via `get_last_config()`
- Inside kernels: use `np.float32`, `np.int32`, `np.bool` etc. for dtype args, or `x.dtype` to reference a tensor's dtype

---

## 2. Data Types

### 2.1 DType Constants

| Integer Types | Float Types | Special Types |
|---------------|-------------|---------------|
| `ct.bool_` (8-bit) | `ct.float16` (IEEE 754 half) | `ct.tfloat32` (tensor float, 19-bit) |
| `ct.int8` / `ct.uint8` | `ct.float32` (IEEE 754 single) | `ct.float8_e4m3fn` (FP8, 1+4+3) |
| `ct.int16` / `ct.uint16` | `ct.float64` (IEEE 754 double) | `ct.float8_e5m2` (FP8, 1+5+2) |
| `ct.int32` / `ct.uint32` | `ct.bfloat16` (1+8+7 bits) | |
| `ct.int64` / `ct.uint64` | | |

DType properties: `.name` (str), `.bitwidth` (int).

Inside kernels, you can also use numpy dtypes: `np.float32`, `np.float16`, `np.int32`, etc.

### 2.2 DType Usage in Kernels

```python
# Use ct.* types or np.* types interchangeably in kernels
tile = ct.full((64,), 0.0, dtype=ct.float32)    # ct type
tile = ct.full((64,), 0.0, dtype=np.float32)     # numpy type
tile = ct.astype(x, ct.bfloat16)                 # explicit cast
tile = ct.astype(x, x_array.dtype)               # match input dtype
```

---

## 3. Core Classes

### 3.1 Array (Global Memory)

`Array` wraps a PyTorch tensor passed to the kernel. Available properties:

```python
array.dtype     # DType — element data type (compile-time constant)
array.shape     # tuple[int,...] — dimensions (compile-time per axis)
array.strides   # tuple[int,...] — strides per dimension
array.ndim      # int — number of dimensions (compile-time constant)
```

**Array.slice()** — create a sub-view:

```python
sub = array.slice(axis, start, stop)
# axis: const int (negative counts from last dim)
# start/stop: int or 0D tile (inclusive/exclusive)
# Returns: Array view (same memory, restricted range)
```

### 3.2 Tile (Local Data)

`Tile` is an immutable multi-dimensional collection local to a block. Every dimension must be a **power of 2**.

Properties:
```python
tile.dtype    # DType — element type
tile.shape    # tuple[const int,...] — shape
tile.ndim     # int — number of dimensions
```

Methods:
```python
tile.item()                         # extract scalar (0D Tile) from 1-element tile
tile.reshape(shape)                 # reshape
tile.permute(axes)                  # permute dimensions
tile.transpose(axis0=None, axis1=None)  # transpose two dims (auto for 2D)
tile.astype(dtype)                  # type cast
tile.extract(index, shape)          # extract sub-tile
tile[np.newaxis, :]                 # expand_dims via indexing
tile[:, None]                       # add dim via slicing
```

Operator overloads on Tile:
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Bitwise: `&`, `|`, `^`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Reverse ops: `radd`, `rsub`, `rmul`, etc. (for `scalar op tile`)

### 3.3 Scalar

`Scalar = int | float` — Python scalars auto-promote to tiles in binary operations.

---

## 4. Enumerations

### 4.1 PaddingMode

Controls out-of-bounds behavior for `ct.load()`:

| Value | Description | Use Case |
|-------|-------------|----------|
| `ct.PaddingMode.UNDETERMINED` | Undefined OOB (default) | When OOB never happens |
| `ct.PaddingMode.ZERO` | OOB = 0 | Sum, L2 norm, RMSNorm, LayerNorm |
| `ct.PaddingMode.NEG_ZERO` | OOB = -0.0 | Rare; IEEE semantics |
| `ct.PaddingMode.NAN` | OOB = NaN | Detection of OOB access |
| `ct.PaddingMode.POS_INF` | OOB = +inf | min reductions |
| `ct.PaddingMode.NEG_INF` | OOB = -inf | max, argmax, softmax |

### 4.2 RoundingMode

Controls float rounding for `ct.truediv`, `ct.sqrt`, `ct.sum`, `ct.add`, etc.:

| Value | Alias | Description |
|-------|-------|-------------|
| `RMd.RN` | `NEAREST_EVEN` | Round to nearest, ties to even (default) |
| `RMd.RZ` | `ZERO` | Round towards zero (truncate) |
| `RMd.RM` | `NEGATIVE_INF` | Round towards -inf |
| `RMd.RP` | `POSITIVE_INF` | Round towards +inf |
| `RMd.APPROX` | — | Approximate (faster, less accurate) |
| `RMd.FULL` | — | Full precision |

### 4.3 MemoryOrder (for atomics)

| Value | Description |
|-------|-------------|
| `ct.MemoryOrder.RELAXED` | No ordering guarantee |
| `ct.MemoryOrder.ACQUIRE` | Acquire semantics |
| `ct.MemoryOrder.RELEASE` | Release semantics |
| `ct.MemoryOrder.ACQ_REL` | Combined acquire+release (default for atomics) |

### 4.4 MemoryScope (for atomics)

| Value | Description |
|-------|-------------|
| `ct.MemoryScope.BLOCK` | Same CTA block |
| `ct.MemoryScope.DEVICE` | Same GPU (default for atomics) |
| `ct.MemoryScope.SYS` | Entire system (all GPUs + host) |

---

## 5. Kernel Definition & Launch

### 5.1 @ct.kernel Decorator

```python
@ct.kernel
def my_kernel(x, y, output, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    # ...

# With optimization hints:
@ct.kernel(occupancy=2, num_ctas=4, opt_level=3)
def my_kernel(x, y, output, N: ConstInt, TILE: ConstInt):
    # ...

# Target-specific hints:
from cuda.tile import ByTarget
@ct.kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
def my_kernel(x, y, output, N: ConstInt, TILE: ConstInt):
    # ...
```

Parameters:
- `num_ctas` — CTAs per CGA, power of 2 in [1, 16]. Default: auto.
- `occupancy` — Active CTAs per SM, [1, 32]. Default: auto.
- `opt_level` — Optimization level [0, 3]. Default: 3.

Kernel parameter rules:
- No annotation → runtime value (torch tensor, Python scalar like `float`, `int`)
- `ConstInt` / `ConstBool` → compile-time constant (embedded in binary, triggers recompilation on change)
- `ct.Constant[int]`, `ct.Constant[bool]`, `ct.Constant[float]` are all available

### 5.2 @ct.function Decorator

For helper functions callable from kernels:

```python
@ct.function
def helper(x, y):
    return x + y

@ct.function(host=True)       # callable from both host and tile code
def cdiv_helper(a, b):
    return (a + b - 1) // b
```

### 5.3 ct.launch()

```python
ct.launch(stream, grid, kernel, kernel_args)
```

- `stream` — CUDA stream (`torch.cuda.current_stream()`)
- `grid` — `(x,)`, `(x, y)`, or `(x, y, z)` tuple of block counts
- `kernel` — `@ct.kernel` decorated function
- `kernel_args` — tuple matching kernel signature order

### 5.4 Block/Grid Information

```python
ct.bid(axis)          # int32 — block ID along axis (0, 1, or 2)
ct.num_blocks(axis)   # int32 — total blocks along axis

ct.num_tiles(array, axis, shape, order="C")
# Number of tiles in the tile space of array along axis.
# shape: tile shape tuple.
# order: "C" (default), "F" (reversed), or tuple permutation.
```

---

## 6. Memory Operations

### 6.1 ct.load()

```python
tile = ct.load(array, index, shape, *,
               order="C",
               padding_mode=PaddingMode.UNDETERMINED,
               latency=None,
               allow_tma=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Source array (torch tensor) |
| `index` | tuple[int\|Tile,...] | Tile-space coordinates |
| `shape` | tuple[const int,...] | Tile shape (all dims must be power of 2); `()` for 0D scalar tile |
| `order` | `"C"` / `"F"` / tuple | Axis mapping: `"C"`=identity, `"F"`=reversed, tuple=explicit permutation |
| `padding_mode` | PaddingMode | OOB element handling |
| `latency` | const int | DRAM traffic hint, 1 (low) to 10 (high) |
| `allow_tma` | const bool | Enable/disable TMA (default True) |

```python
# Examples
tile = ct.load(x, index=(bid,), shape=(TILE,))                           # 1D
tile = ct.load(x, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)  # 2D padded
tile = ct.load(Q, index=(b, h, i, 0), shape=(1,1,M,D))                  # 4D
tile = ct.load(K, index=(b, h, 0, j), shape=(1,1,D,N), order=(0,1,3,2)) # transposed load
scalar = ct.load(x, index=(i,), shape=())                                # scalar (0D tile)
```

### 6.2 ct.store()

```python
ct.store(array, index, tile, *,
         order="C",
         latency=None,
         allow_tma=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Destination array |
| `index` | tuple[int\|Tile,...] | Tile-space coordinates |
| `tile` | Tile or scalar | Data to store |
| `order` | `"C"` / `"F"` / tuple | Axis mapping |
| `latency` | const int | Latency hint |
| `allow_tma` | const bool | Enable/disable TMA |

OOB writes are silently ignored (no-op).

**CRITICAL**: indices must be **static** (compile-time known or block-uniform). For runtime-computed destinations, use `ct.scatter()`.

### 6.3 ct.gather()

```python
result = ct.gather(array, indices, *,
                   padding_value=0,
                   check_bounds=True,
                   latency=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Source array |
| `indices` | Tile or tuple[Tile\|int,...] | Per-element indices (length = array rank); single Tile for 1D arrays |
| `padding_value` | scalar | Value for OOB indices (default 0) |
| `check_bounds` | bool | Enable bounds checking (default True) |
| `latency` | int | Latency hint |

- All index tiles must be same shape or broadcastable. Result shape = broadcasted shape.
- Negative indices treated as OOB.

```python
# 1D gather
val = ct.gather(dest_loc, token_id)                     # scalar
# Multi-dim gather
vals = ct.gather(kv, (token_id, head_id, offsets))       # tile
# With custom padding
vals = ct.gather(x, (idx,), padding_value=-1)            # -1 for OOB
```

### 6.4 ct.scatter()

```python
ct.scatter(array, indices, value, *,
           check_bounds=True,
           latency=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Destination array |
| `indices` | Tile or tuple[Tile\|int,...] | Per-element destination indices |
| `value` | Tile or scalar | Data to scatter (broadcastable to indices shape) |
| `check_bounds` | bool | Enable bounds checking (default True) |
| `latency` | int | Latency hint |

OOB writes are silently ignored.

```python
ct.scatter(out, (dest_index, head_id, offsets), kv_vals)
```

---

## 7. Atomic Operations

All atomics return the **old value** before the operation. All support these common keyword args:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_bounds` | bool | True | Bounds checking |
| `memory_order` | MemoryOrder | ACQ_REL | Ordering semantics |
| `memory_scope` | MemoryScope | DEVICE | Scope of ordering |

```python
ct.atomic_add(array, indices, update)     # array[idx] += update; return old
ct.atomic_max(array, indices, update)     # array[idx] = max(old, update); return old
ct.atomic_min(array, indices, update)     # array[idx] = min(old, update); return old
ct.atomic_and(array, indices, update)     # array[idx] &= update; return old
ct.atomic_or(array, indices, update)      # array[idx] |= update; return old
ct.atomic_xor(array, indices, update)     # array[idx] ^= update; return old
ct.atomic_xchg(array, indices, update)    # array[idx] = update; return old
ct.atomic_cas(array, indices, expected, desired)
    # if array[idx] == expected: array[idx] = desired; return old
```

OOB atomics: no-op, returns `expected` (for CAS) or `update`.

```python
# Example: histogram
old = ct.atomic_add(histogram, (bin_idx,), 1,
                    memory_order=ct.MemoryOrder.RELAXED,
                    memory_scope=ct.MemoryScope.DEVICE)
```

---

## 8. Tile Creation (Factory Functions)

```python
ct.full(shape, fill_value, dtype)       # filled tile
ct.zeros(shape, dtype)                   # all zeros
ct.ones(shape, dtype)                    # all ones
ct.arange(size, dtype=...)               # [0, 1, ..., size-1]
```

```python
# Examples
acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
zero = ct.zeros((TILE,), dtype=ct.float16)
one = ct.ones((64,), dtype=ct.int32)
idx = ct.arange(128, dtype=np.int32)

# Common indexing patterns
offs_m = ct.arange(TILE_M, dtype=np.int32)[:, None]   # [TILE_M, 1] column
offs_n = ct.arange(TILE_N, dtype=np.int32)[None, :]   # [1, TILE_N] row
```

---

## 9. Shape & View Operations

```python
ct.reshape(x, shape)              # reshape; one dim can be -1
ct.permute(x, axes)               # permute dimensions
ct.transpose(x, axis0=None, axis1=None)  # transpose two axes; auto for 2D
ct.expand_dims(x, axis)           # insert size-1 dimension
ct.broadcast_to(x, shape)         # broadcast (NumPy rules)
ct.cat((x, y), axis)              # concatenate two tiles along axis
ct.extract(x, index, shape)       # extract sub-tile from tile
ct.bitcast(x, dtype)              # reinterpret bits as different type (same bitwidth)
```

```python
# Examples
q = ct.load(Q, ..., shape=(1,1,M,D)).reshape((M, D))   # 4D -> 2D
acc = ct.reshape(acc, (1, 1, M, D))                      # 2D -> 4D
k_t = ct.transpose(k)                                    # transpose 2D tile
p = ct.permute(tile, (0, 2, 1))                          # reorder 3D axes
offs = ct.expand_dims(ct.arange(M), 1)                   # [M] -> [M, 1]
big = ct.cat((left, right), axis=1)                       # concat along cols
sub = ct.extract(tile, (0, 0), shape=(32, 32))            # top-left 32x32
i32_bits = ct.bitcast(float_tile, ct.int32)               # reinterpret float as int
```

---

## 10. Type Conversion

```python
ct.astype(x, dtype)               # cast tile to dtype
x.astype(dtype)                   # method form

# Inside kernels, numpy dtypes or ct dtypes both work:
y = ct.astype(x, np.float32)
y = ct.astype(x, ct.bfloat16)
y = x.astype(array.dtype)         # match input tensor's dtype
```

Common pattern — **compute in fp32, store in original dtype**:
```python
xj = ct.astype(ct.load(x, ...), np.float32)    # upcast for accumulation
yj = ct.astype(result, x.dtype)                 # downcast for storage
```

---

## 11. Arithmetic Operations

### 11.1 Operator Overloads (preferred for simple cases)

```python
c = a + b          c = a - b          c = a * b
c = a / b          c = a // b         c = a % b
c = a ** b
c = a & b          c = a | b          c = a ^ b
c = ~a                                c = -a
```

Python scalars auto-promote: `x * 2`, `x + 0.5`, `3 - x`.

### 11.2 Explicit Functions (when you need extra control)

All support `rounding_mode` and/or `flush_to_zero` where applicable:

```python
ct.add(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.sub(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.mul(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.truediv(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.floordiv(x, y)
ct.mod(x, y)
ct.pow(x, y)
ct.negative(x)              # unary negate (same as -x)
ct.abs(x)                   # absolute value

ct.minimum(x, y, *, flush_to_zero=False)   # element-wise min
ct.maximum(x, y, *, flush_to_zero=False)   # element-wise max

ct.cdiv(x, y)     # ceiling division: ceil(x/y). Works on host AND in kernels.
```

```python
# Example: controlled division
acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
```

### 11.3 Bitwise Operations

```python
ct.bitwise_and(x, y)        # x & y
ct.bitwise_or(x, y)         # x | y
ct.bitwise_xor(x, y)        # x ^ y
ct.bitwise_not(x)           # ~x
ct.bitwise_lshift(x, n)     # x << n
ct.bitwise_rshift(x, n)     # x >> n
```

---

## 12. Comparison Operations

### 12.1 Operator Overloads (return bool tiles)

```python
mask = x > y       mask = x >= y      mask = x < y
mask = x <= y      mask = x == y      mask = x != y
```

### 12.2 Explicit Functions

```python
ct.greater(x, y)             # x > y
ct.greater_equal(x, y)       # x >= y
ct.less(x, y)                # x < y
ct.less_equal(x, y)          # x <= y
ct.equal(x, y)               # x == y
ct.not_equal(x, y)           # x != y
```

---

## 13. Math Functions

### 13.1 Exponential & Logarithmic

```python
ct.exp(x)           # e^x
ct.exp2(x, *, flush_to_zero=False)   # 2^x (faster; use for softmax with log2 scaling)
ct.log(x)           # ln(x)
ct.log2(x)          # log2(x)
ct.log10(x)         # log10(x)
ct.log1p(x)         # log(1 + x) — more accurate near x=0
```

### 13.2 Trigonometric

```python
ct.sin(x)           ct.cos(x)           ct.tan(x)
ct.sinh(x)          ct.cosh(x)          ct.tanh(x)
```

### 13.3 Power & Root

```python
ct.sqrt(x, *, rounding_mode=None, flush_to_zero=False)    # sqrt(x)
ct.rsqrt(x, *, flush_to_zero=False)                        # 1/sqrt(x)
ct.pow(x, y)                                                # x^y
```

### 13.4 Rounding

```python
ct.floor(x)         # round down
ct.ceil(x)          # round up
```

---

## 14. Reduction Operations

All reductions support:
- `axis=None` — reduce all elements to scalar
- `axis=int` — reduce along one axis
- `axis=tuple` — reduce along multiple axes
- `keepdims=False` — whether to preserve reduced dimensions

```python
ct.sum(x, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False)
ct.prod(x, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False)
ct.max(x, axis=None, *, keepdims=False, flush_to_zero=False)
ct.min(x, axis=None, *, keepdims=False, flush_to_zero=False)
ct.argmax(x, axis=None, *, keepdims=False)
ct.argmin(x, axis=None, *, keepdims=False)
```

```python
# Examples
total = ct.sum(tile)                                # scalar sum
row_sum = ct.sum(tile, axis=1, keepdims=False)      # sum along cols
max_val = ct.max(tile, axis=-1, keepdims=True)      # keep dim for broadcast
idx = ct.argmax(tile)                                # index of max element
```

### 14.1 Custom Reduction

```python
ct.reduce(x, axis, func, identity, *, keepdims=False)
```

- `x` — Tile or tuple of Tiles
- `func(a, b) -> combined` — reduction function (binary associative)
- `identity` — identity element (scalar or tuple of scalars)

```python
# Custom sum (equivalent to ct.sum)
result = ct.reduce(x, axis=0, func=lambda a, b: a + b, identity=0)

# Multi-output reduction: simultaneous min and argmin
min_val, min_idx = ct.reduce(
    (values, indices), axis=0,
    func=lambda v1, i1, v2, i2: (
        ct.where(v1 < v2, v1, v2),
        ct.where(v1 < v2, i1, i2)
    ),
    identity=(float('inf'), 0)
)
```

---

## 15. Scan (Prefix) Operations

```python
ct.cumsum(x, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False)
ct.cumprod(x, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False)
```

- `axis` — axis along which to scan (default 0)
- `reverse` — if True, scan from end to start

```python
# Prefix sum
prefix = ct.cumsum(tile, axis=1)           # [a, a+b, a+b+c, ...]
# Suffix product
suffix = ct.cumprod(tile, axis=0, reverse=True)
```

---

## 16. Conditional & Selection

```python
ct.where(condition, x, y)
# condition: bool Tile of shape S
# x, y: Tiles of shape S and same dtype (or scalars)
# Returns: element from x where True, y where False
```

```python
# ReLU
y = ct.where(x >= 0, x, 0.0)

# Causal mask for attention
mask = ct.where(offs_m >= offs_n, 0.0, -np.inf)
qk = qk + mask

# Clamp
y = ct.maximum(ct.minimum(x, upper), lower)
```

---

## 17. Matrix Multiply

### 17.1 ct.mma() — Matrix Multiply-Accumulate (Fused)

```python
result = ct.mma(x, y, acc)
# x: [M, K] or [B, M, K]
# y: [K, N] or [B, K, N]
# acc: [M, N] or [B, M, N]
# Returns: (x @ y) + acc with acc's dtype preserved
```

Supported input/accumulator dtype combinations:

| Input (x, y) | Accumulator (acc) |
|---------------|-------------------|
| float16 | float16 or float32 |
| bfloat16 | float32 |
| float32 | float32 |
| float64 | float64 |
| tfloat32 | float32 |
| float8_e4m3fn | float16 or float32 |
| float8_e5m2 | float16 or float32 |
| int8 / uint8 | int32 |

```python
# GEMM pattern
acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
for k_block in range(num_k_tiles):
    a = ct.load(A, index=(bid_m, k_block), shape=(TILE_M, TILE_K))
    b = ct.load(B, index=(k_block, bid_n), shape=(TILE_K, TILE_N))
    acc = ct.mma(a, b, acc)
```

### 17.2 ct.matmul() — Simple Matrix Multiply

```python
result = ct.matmul(x, y)   # or x @ y
# Auto-promotes dtypes and infers result type
# Supports 1D, 2D, 3D (batch) inputs
```

Difference from `ct.mma()`: `matmul` doesn't require an accumulator, auto-promotes types, and returns a fresh result. `mma` is fused and preserves accumulator dtype — preferred for performance-critical paths.

---

## 18. Control Flow

### 18.1 Python for-loop (compile-time unrolled)

```python
num_tiles = ct.cdiv(N, TILE_SIZE)
for j in range(0, num_tiles):
    xj = ct.load(x, index=(row, j), shape=(1, TILE_SIZE), ...)
    # ... process tile
```

The loop is fully unrolled at compile time. `range()` bounds can use `ct.cdiv()` or any expression yielding a compile-time int.

### 18.2 Python if/else (compile-time)

```python
if CAUSAL:   # ConstBool — resolved at compile time, generates only one branch
    mask = offs_m >= offs_n
```

### 18.3 Runtime Conditionals

`ct.where()` for element-wise (see Section 16).

For block-level branching, use standard Python `if` on runtime scalars — but note this creates both branches in the compiled code and selects at runtime.

---

## 19. Synchronization (Advanced)

Token-based synchronization for explicit memory ordering (low-level API, rarely needed for typical kernels):

```python
from cuda_tile.dialects.cuda_tile_ops import (
    make_token, join_tokens,
    load_ptr_tko, store_ptr_tko,
    load_view_tko, store_view_tko,
)

token = make_token()
data, token = load_ptr_tko(result_type, ptr,
                            memory_ordering_semantics=MemoryOrderingSemantics.WEAK,
                            input_token=token, return_token=True)
token = store_ptr_tko(ptr, data, input_token=token)
combined = join_tokens(token1, token2)
```

These are the low-level MLIR bindings. The high-level `ct.load`/`ct.store` handle synchronization automatically in most cases.

---

## 20. TensorView & PartitionView (Advanced)

For complex memory access patterns beyond simple tile-space indexing:

```python
from cuda_tile.dialects.cuda_tile_ops import (
    make_tensor_view, make_partition_view,
    load_view_tko, store_view_tko,
    get_tensor_shape,
)

# Create tensor view with dynamic shape
tv = make_tensor_view(base_ptr, Float32,
                       shape=[N, M],       # int or scalar Tile for dynamic
                       strides=[M, 1])

# Partition into tiles
pv = make_partition_view(tv, tile_shape=[64, 64],
                          padding_value=PaddingValue.ZERO)

# Load/store via views
data = load_view_tko(pv, indices=[i, j], return_token=False)
store_view_tko(data, pv, indices=[i, j])
```

---

## 21. Debug & Utility

### 21.1 printf

```python
ct.printf(format_str, *tile_args)
# format_str: C-style printf format (specifiers: %d %u %x %f %e %g etc.)
# *tile_args: only Tile values (not Python scalars)
```

**Significant overhead** — use for debugging only. Multi-block outputs interleave. Use `opt_level=0` for serial output.

```python
ct.printf("bid=%d val=%f\n", ct.bid(0), tile)
ct.printf("x[0]=%f x[1]=%f\n", tile)  # prints per-element
```

### 21.2 assert_

```python
ct.assert_(cond_tile)
ct.assert_(cond_tile, "message")
# Asserts all elements of cond_tile are True. Significant overhead.
```

---

## 22. Compiler Hints (Advanced)

Optimization hints that help the compiler generate better code:

```python
# Hint: value is divisible by divisor
x = ct.assume_div_by(x, divisor=16)
x = ct.assume_div_by(x, divisor=4, every=2, along=0)  # every 2nd element along axis 0

# Hint: elements within groups are identical
x = ct.assume_same_elements(x, group_size=[1, 32])

# Hint: value is bounded
x = ct.assume_bounded(x, lb=0, ub=1023)

# Prevent compiler from optimizing across this point
x = ct.optimization_barrier(x)
```

---

## 23. Autotune Pattern

### 23.1 autotune_launch()

```python
result = ct_experimental.autotune_launch(
    stream,                                              # CUDA stream
    grid_fn=lambda cfg: (grid_x, grid_y, 1),            # cfg -> grid tuple
    kernel=my_kernel,                                     # @ct.kernel function
    args_fn=lambda cfg: (x, y, output, cfg.tile),        # cfg -> kernel args tuple
    hints_fn=lambda cfg: {"occupancy": cfg.occupancy},   # cfg -> hints dict
    search_space=_SEARCH_SPACE,                          # iterable of config objects
    # Optional:
    key=None,                   # hashable cache key (auto-generated from tensor shapes/dtypes)
    max_iter=60,                # max configs to try
    compiler_time_limit_sec=10, # timeout per config compilation
    seed=None,                  # random seed for sampling
    force_retune=False,         # ignore cache and retune
)
```

Returns `TunedResult`:
- `result.tuned_config` — winning `SimpleNamespace`
- `result.grid` — grid tuple used
- `result.cache_hit` — bool
- `result.tuning_record` — list of `(config, time_ms)` pairs

### 23.2 clear_autotune_cache()

```python
ct_experimental.clear_autotune_cache()                    # clear all
ct_experimental.clear_autotune_cache(kernel=my_kernel)    # clear for one kernel
ct_experimental.clear_autotune_cache(key=my_key)          # clear for one key
```

### 23.3 Standard Implementation Template

```python
_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]

def run(x: torch.Tensor, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    output = torch.empty_like(x)
    n = x.numel()
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (ct.cdiv(n, cfg.tile), 1, 1),
            kernel=my_kernel,
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile": result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        ct.launch(stream, (ct.cdiv(n, cfg.tile), 1, 1),
                  my_kernel, (x, output, cfg.tile))
    return output

def get_last_config() -> dict | None:
    return _last_autotune_config
```

---

## 24. run() and get_last_config() Contract

Every `impl_cutile.py` must export:

```python
def run(*inputs, autotune=False, **kwargs) -> torch.Tensor:
    """Execute the operator."""
    ...

def get_last_config() -> dict | None:
    """Return last autotune config dict, or None."""
    return _last_autotune_config
```

- `_last_autotune_config` is module-level, written only in the autotune branch
- `autotune_launch()` has no kernel-level cache (unlike Triton); `_last_autotune_config` persists the result
- If an operator cannot be implemented: `run()` raises `NotImplementedError`, `get_last_config()` returns `None`

---

## 25. Common Kernel Patterns

### Pattern A: Element-wise (mul2, relu, vector_add, dropout, swiglu)

```python
@ct.kernel
def kernel(x, out, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    y_tile = x_tile * 2  # or any element-wise op
    ct.store(out, index=(bid,), tile=y_tile)

# Grid: (cdiv(n_elements, tile), 1, 1)
```

Variants:
- **ReLU**: `ct.where(x >= 0, x, 0.0)`
- **SwiGLU**: `x * (1.0 / (1.0 + ct.exp(-gate)))`
- **Dropout**: `ct.load(mask, ...) * x * scale`

### Pattern B: Row-wise single-pass (softmax, argmax)

Load entire row (padded to power-of-2), one CTA per row:

```python
@ct.kernel
def kernel(x, out, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.NEG_INF)
    max_val = ct.max(tile)
    exp_tile = ct.exp(tile - max_val)
    out_tile = exp_tile / ct.sum(exp_tile)
    ct.store(out, index=(row, 0), tile=out_tile)
```

Host: `TILE = 1 << (n_cols - 1).bit_length()`

### Pattern C: Row-wise two-pass (rmsnorm, layernorm, l2_norm)

Two loops over tiles when row doesn't fit in one tile:

```python
@ct.kernel
def kernel(x, w, out, eps, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # Pass 1: accumulate
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(ct.load(x, index=(row, j), shape=(1, TILE),
                       allow_tma=False, latency=1,
                       padding_mode=ct.PaddingMode.ZERO), np.float32)
        acc = acc + xj * xj

    rstd = ct.rsqrt(ct.sum(acc, axis=1, keepdims=False) / N + eps)

    # Pass 2: normalize
    for j in range(0, num_tiles):
        xj = ct.astype(ct.load(x, ..., padding_mode=ct.PaddingMode.ZERO), np.float32)
        wj = ct.astype(ct.load(w, ..., padding_mode=ct.PaddingMode.ZERO), np.float32)
        ct.store(out, index=(row, j), tile=ct.astype(xj * rstd * wj, x.dtype))
```

### Pattern D: Flash Attention (online softmax + MMA)

```python
@ct.kernel(occupancy=2)
def fmha(Q, K, V, Out, qk_scale: float,
         D: ConstInt, H: ConstInt, M: ConstInt, N: ConstInt, CAUSAL: ConstBool):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch, head = bid_y // H, bid_y % H
    qk_scale *= 1.0 / math.log(2)  # log2 scale for exp2

    m_i = ct.full((M, 1), -np.inf, dtype=np.float32)    # running max
    l_i = ct.full((M, 1), 0.0, dtype=np.float32)        # running sum
    acc = ct.full((M, D), 0.0, dtype=np.float32)         # accumulator

    q = ct.load(Q, ...).reshape((M, D))
    for j in range(0, Tc):
        k = ct.load(K, ..., order=(0,1,3,2)).reshape((D, N))
        qk = ct.mma(q, k, ct.full((M, N), 0., dtype=np.float32))
        # Apply causal mask if needed
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        p = ct.exp2(qk * qk_scale - m_ij, flush_to_zero=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha
        v = ct.load(V, ...).reshape((N, D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
```

### Pattern E: Dynamic indexing (gather/scatter)

```python
@ct.kernel
def kernel(kv, dest_loc, out, DIM: ConstInt):
    token_id, head_id = ct.bid(0), ct.bid(1)
    dest = ct.gather(dest_loc, token_id)
    offsets = ct.arange(DIM, dtype=np.int32)
    vals = ct.gather(kv, (token_id, head_id, offsets))
    ct.scatter(out, (dest, head_id, offsets), vals)
```

### Pattern F: NotImplementedError

```python
def run(*args, **kwargs):
    raise NotImplementedError("cuTile does not support <reason>")

def get_last_config() -> dict | None:
    return None
```

---

## 26. Critical Constraints & Gotchas

### Tile dimensions must be powers of two
All dimensions in `ct.load/store` `shape` must be powers of 2. Handle non-pow2 problem sizes by:
- Round up TILE_SIZE: `block = 1 << (n - 1).bit_length()`
- Use `padding_mode` for OOB reads
- OOB stores are silently ignored

### ct.store() requires static indices
Use `ct.scatter()` for runtime-computed destination indices. Use `ct.gather()` for runtime-computed source indices.

### Tiles are immutable
```python
# WRONG: x_tile += 1
# RIGHT:
x_tile = x_tile + 1
```

### Always accumulate in fp32
```python
acc = ct.full(..., 0.0, dtype=np.float32)
xj = ct.astype(ct.load(...), np.float32)
result = ct.astype(acc, x.dtype)
```

### Grid tuple
`ct.launch()` accepts 1-, 2-, or 3-tuple. Convention: always use 3-tuple with trailing 1s:
```python
grid = (num_blocks, 1, 1)      # 1D
grid = (grid_x, grid_y, 1)     # 2D
```

### flush_to_zero for exp2
Use `flush_to_zero=True` with `ct.exp2()` in softmax to avoid denormals:
```python
p = ct.exp2(qk, flush_to_zero=True)
```

### Kernel vs host code
Inside `@ct.kernel`: Python `for`/`if` on compile-time values are unrolled/specialized. Array `.shape`, `.dtype` produce compile-time constants. All operations return tiles.

Outside kernel: standard Python/PyTorch. Use `ct.cdiv()` for ceiling division (it works both host and device side).

### Autotune has no persistent cache
`ct_experimental.autotune_launch()` caches in-memory per session (not on disk). `_last_autotune_config` module-level variable is the only way to expose configs to the benchmark engine.

### fp16 argmax precision
fp16 can only exactly represent integers up to 2048. For argmax on large dimensions, cast to fp32 first:
```python
x2d = x2d.float()  # on host, before passing to kernel
```

---

## 27. Quick Reference Tables

### ct.load() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | Array | required | Source tensor |
| `index` | tuple | required | Tile-space coordinates |
| `shape` | tuple[const int] | required | Tile shape (pow2 dims); `()` for scalar |
| `order` | `"C"`/`"F"`/tuple | `"C"` | Axis permutation |
| `padding_mode` | PaddingMode | UNDETERMINED | OOB behavior |
| `latency` | const int | None | DRAM hint 1-10 |
| `allow_tma` | const bool | True | TMA enable |

### ct.store() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | Array | required | Destination tensor |
| `index` | tuple | required | Tile-space coordinates (static only!) |
| `tile` | Tile/scalar | required | Data to write |
| `order` | `"C"`/`"F"`/tuple | `"C"` | Axis permutation |
| `latency` | const int | None | DRAM hint 1-10 |
| `allow_tma` | const bool | True | TMA enable |

### MMA Supported Dtypes

| x, y dtype | acc dtype | Notes |
|------------|-----------|-------|
| float16 | float16, float32 | Most common |
| bfloat16 | float32 | Training |
| float32 | float32 | Uses TF32 cores |
| float64 | float64 | Double precision |
| tfloat32 | float32 | Explicit TF32 |
| float8_e4m3fn | float16, float32 | FP8 inference |
| float8_e5m2 | float16, float32 | FP8 |
| int8/uint8 | int32 | Quantized |


---

# Operator description: `reverse_array`

# Reverse Array

Implement a program that reverses an array in-place. The program should perform an in-place reversal of `input`.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored back in `input`

## Example 1

```text
Input: [1.0, 2.0, 3.0, 4.0]
Output: [4.0, 3.0, 2.0, 1.0]
```

## Example 2

```text
Input: [1.5, 2.5, 3.5]
Output: [3.5, 2.5, 1.5]
```

## Constraints

- $1 \leq N$


---

# `config.yaml` for `reverse_array`

```yaml
# Reverse array — pure data movement (no arithmetic)
# No verify section needed: exact element reordering, per-dtype defaults suffice.

benchmark:
  warmup: 20
  repeat: 100
  use_cuda_graph: true
  flush_l2: true
  autotune: false

case_grid:
  n:
    expr: "[1000000 * i for i in range(1, 21)]"   # 1M, 2M, ..., 20M (20 points)
  dtype: ["fp16", "bf16", "fp32", "int8"]

# ---------------------------------------------------------------------------
# Derived-metric expressions
#   reverse is pure memory-bound: no meaningful FLOPs, use element count as proxy
#   1 read + 1 write per element
# ---------------------------------------------------------------------------
metrics:
  flops_expr: "n"
  bytes_expr: "n * dtype_size * 2"

  plots:
    - latency_ms
    - bandwidth_GBs
    - speedup

```

---

# PyTorch reference: `impl_torch.py`

```python
import torch


def run(input: torch.Tensor, N: int, **kwargs):
    return input.flip(0).contiguous()

```

---

## Output format (STRICT)

Return EXACTLY 2 fenced code blocks, in this order, with these titles:

    ```python title="impl_triton.py"
    # full Python file content here
    ```

    ```python title="impl_cutile.py"
    # full Python file content here
    ```

Emit ONLY the `impl_triton.py`, `impl_cutile.py` files. No other code blocks. No prose between or
after the blocks beyond a 1-2 sentence summary of your approach. Do
NOT include test code, do NOT include PyTorch reference code (that file is
provided by the framework).
