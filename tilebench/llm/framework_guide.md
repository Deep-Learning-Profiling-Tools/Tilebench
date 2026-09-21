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

The refinement loop runs the full iteration budget (default 10 iters).
There is no roofline-based early stopping; both Triton and cuTile are
regenerated every iteration. After the budget is exhausted, the best
verify-clean iteration per backend is promoted as the final result.
Each iteration's `stop_score` (capped `roofline_pct` on the single
largest case per dtype, averaged across the supported dtypes) is
reported as feedback so the next iteration can see how close to peak
the previous attempt got.

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
