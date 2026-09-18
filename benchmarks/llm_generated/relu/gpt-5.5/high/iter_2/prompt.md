# Task: improve operator `relu` for TileBench (iteration 2)

Your previous iteration (1) did not yet meet the stopping criterion (`stop_score` ≥ 80% on the **single largest case per dtype**) for `cutile`. Read the trajectory below carefully — if your last iteration **regressed** vs the best verify-clean iter so far, you should consider going back to that approach as your starting point and trying a different optimization. Then re-emit `impl_cutile.py`.

## Frozen backends

- `triton` froze at iter 0 with stop_score=80.4%. Do NOT regenerate it; focus only on `impl_cutile.py`.

## Iteration trajectory so far

| iter | triton cfg | triton score | triton speedup_vs_torch | triton verify | cutile cfg | cutile score | cutile speedup_vs_torch | cutile verify |
|---|---|---|---|---|---|---|---|---|
| 0 | {BLOCK_SIZE:2048, num_warps:4, num_stages:2} | 80.4% | 1.18× | ✓ | — | 0.0% | — | ✗4 |
| 1 | (frozen) | (frozen) | — | — | {TILE:2048, occupancy:8, allow_tma:False} | 77.2% | 1.18× | ✓ |

_`cfg` is the configuration your kernel actually used, as returned by `get_last_config()`. `speedup_vs_torch` is `torch_latency / kernel_latency` — values >1 mean your kernel beat the PyTorch reference; <1 means torch is still faster._

**Best verify-clean `cutile` so far: iter 1 with stop_score=77.2%.**

## Feedback from iteration 1

### `cutile` performance — stop_score=**77.2%** (target ≥ 80%), report_mean=77.2%

Per-(backend,dtype,case) detail (sorted by worst first):
- `cutile` / `int8` / {'n': 20971520, 'dtype': 'int8'}: 50.9% of roofline (bandwidth), 1.12× torch, cfg={TILE:2048, occupancy:8, allow_tma:False}
- `cutile` / `fp16` / {'n': 20971520, 'dtype': 'fp16'}: 82.9% of roofline (bandwidth), 1.18× torch, cfg={TILE:2048, occupancy:8, allow_tma:False}
- `cutile` / `bf16` / {'n': 20971520, 'dtype': 'bf16'}: 83.9% of roofline (bandwidth), 1.08× torch, cfg={TILE:2048, occupancy:8, allow_tma:False}
- `cutile` / `fp32` / {'n': 20971520, 'dtype': 'fp32'}: 91.4% of roofline (bandwidth), 0.97× torch, cfg={TILE:2048, occupancy:8, allow_tma:False}

→ Action: focus on the worst-performing combinations. If the case is bandwidth-bound, optimize memory coalescing / tile layout / async copies. If compute-bound, ensure Tensor Cores (tl.dot/ct.mma) and high arithmetic intensity per load.

---

## Your previous `impl_cutile.py` (iteration 1)

```python
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _relu_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    zero = ct.zeros((TILE,), dtype=x.dtype)

    # Match torch.relu/threshold semantics: values <= 0 become +0, NaNs pass through.
    y_tile = ct.where(x_tile <= zero, zero, x_tile)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        allow_tma=False,
    )


def run(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _relu_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "allow_tma": False,
    })
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

# Operator description: `relu`

# ReLU Activation

Implement a program that applies the Rectified Linear Unit (ReLU) activation element-wise to a 1D array. ReLU outputs the input value if it is non-negative, and zero otherwise.

The input consists of one array:

- `x`: A 1D array.

The output should be written to the `output` array, which has the same length as `x`.

The operation is defined mathematically as:

$$
output[i] = \max(0, x[i])
$$

where $i$ ranges from $0$ to $n-1$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`

## Example 1

```text
Input: x = [-2.0, 0.0, 1.5, -0.5, 3.0]
Output: [0.0, 0.0, 1.5, 0.0, 3.0]
```

## Example 2

```text
Input: x = [-1, -2, -3, 4]
Output: [0, 0, 0, 4]
```

## Constraints

- $1 \leq n$


---

# `config.yaml` for `relu`

```yaml
benchmark:
  warmup: 20
  repeat: 100
  use_cuda_graph: true
  flush_l2: true
  autotune: false

case_grid:
  # 1M to 20M elements, step 1M
  n:
    expr: "[1024 * 1024 * i for i in range(1, 21)]"
  dtype: ["fp16", "bf16", "fp32", "int8"]

# ---------------------------------------------------------------------------
# Derived-metric expressions
#   Available variables: n (problem size), dtype_size (bytes per element)
# ---------------------------------------------------------------------------
metrics:
  # relu: 1 comparison per element (memory-bound; use n as FLOP proxy)
  flops_expr: "n"
  # 1 read + 1 write per element
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


def run(x: torch.Tensor):
    return torch.relu(x)

```

---

## Output format (STRICT)

Return EXACTLY 1 fenced code block, in this order, with this title:

    ```python title="impl_cutile.py"
    # full Python file content here
    ```

Emit ONLY the `impl_cutile.py` file. No other code blocks. No prose between or
after the block beyond a 1-2 sentence summary of your approach. Do
NOT include test code, do NOT include PyTorch reference code (that file is
provided by the framework).
