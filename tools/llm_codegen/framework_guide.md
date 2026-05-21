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
def run(*args, autotune: bool = False, **kwargs):
    """Execute the operator. Must produce a single torch.Tensor (or tuple
    of tensors) that bit-equivalently matches impl_torch.run() within the
    tolerances in config.yaml's `verify:` section."""

def get_last_config() -> dict | None:
    """Return the autotune-winner config as a flat dict, or None if no
    autotune has run yet. Engine uses this to log per-case best cfgs."""
```

When called with `autotune=True`, `run()` must perform autotune internally.
When called with `autotune=False`, `run()` must use SOME default config —
since you are NOT required to write `_DEFAULT_CONFIG`, the simplest path
is to ALWAYS run autotune. The harness's autotune cache makes repeated
runs cheap (autotune sweeps once per shape, then caches).

## Autotune convention

### Triton

```python
import triton
import triton.language as tl

@triton.jit
def _op_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ... kernel body ...

_op_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [512, 1024, 2048]      # keep cfg space small (<= 30 cfgs)
        for nw in [2, 4, 8]              # autotune budget is 15min per backend
        for ns in [2, 3]
    ],
    key=["n_elements"],                  # cache key: which problem dims trigger re-tune
)(_op_kernel)

def run(x, autotune: bool = False):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _op_kernel_autotuned[grid](x, output, n_elements)
    return output

def get_last_config() -> dict | None:
    cfg = getattr(_op_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
```

**Rules:**
- `triton.autotune` caches `best_config` on the kernel object automatically.
- Do NOT use a module-level `_last_config` global — read `kernel.best_config`.
- Keep total cfg space ≤ 30; bigger spaces blow the 15-minute autotune cap.
- `num_warps` and `num_stages` are **optional** — include them only if your
  kernel actually benefits from a non-default value (e.g. a software-pipelined
  matmul gains from `num_stages` sweep; a one-pass elementwise rarely does).
  Reasonable ranges when you do sweep them: `num_warps` ∈ {2, 4, 8},
  `num_stages` ∈ {1, 2, 3, 4}.

### cuTile

```python
from types import SimpleNamespace
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

# Module-level dict — DO NOT use `global` keyword; use .clear() + .update().
_last_autotune_config: dict = {}

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]            # keep search space small
    for occ in [4, 8, 16]                  # nw * occ ≈ 64 on B200
]

@ct.kernel
def _op_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(output, index=(bid,), tile=x_tile)   # ... your compute ...

_tuner = CutileAutotuner(_op_kernel)

def run(x, autotune: bool = False):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()
    cfg = _tuner.tune_or_cached(
        shape_key=(n_elements,),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(n_elements, cfg.tile), 1, 1),
        args_fn=lambda cfg: (x, output, cfg.tile),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )
    _last_autotune_config.clear()
    _last_autotune_config.update({"tile": cfg.tile, "occupancy": cfg.occupancy})
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream,
              (ct.cdiv(n_elements, cfg.tile), 1, 1),
              kernel,
              (x, output, cfg.tile))
    return output

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```

**Rules:**
- `_last_autotune_config` must be a module-level `dict` mutated via
  `.clear()` + `.update()`. Never use `global _last_autotune_config`.
- cuTile tile-shape dimensions MUST be powers of 2.
- For non-power-of-2 problem dims: use `padding_mode=ct.PaddingMode.ZERO`
  on `ct.load`, and `ct.store` silently drops OOB writes.
- For max-reduction kernels needing to neutralize OOB: use
  `padding_mode=ct.PaddingMode.NEG_INF`.
- `ct.store()` only accepts static indices. For runtime-computed scatter
  indices, use `ct.scatter()`.

## Choosing meaningful tile / BLOCK sizes

A common LLM mistake is to throw every small tile size into the autotune
search space "to be thorough". Tiles below the thresholds below are
provably suboptimal on B200 (each thread loads / computes too few elements
to amortise launch overhead, miss memory coalescing windows, or fall off
the Tensor Core path). Including them wastes autotune budget on configs
that cannot win.

**Lower bounds — do NOT sweep below these.** All values must be powers of 2.

| Operator shape | Triton `BLOCK_SIZE` / cuTile `tile` | Triton matmul `BLOCK_M`,`BLOCK_N` | Triton matmul `BLOCK_K` | cuTile matmul `tm`,`tn` | cuTile matmul `tk` |
|---|---|---|---|---|---|
| 1D elementwise / pointwise | ≥ **512** | — | — | — | — |
| Row-reduction / softmax / norm (per-row inner tile) | ≥ **256** along reduction dim | — | — | — | — |
| Stencil / 2D conv (sliding window) | ≥ **256** along the spatial inner dim | — | — | — | — |
| Matmul / attention (Tensor Core) | — | ≥ **64** | ≥ **32** | ≥ **64** | ≥ **32** |

**Upper bounds — do NOT sweep above these.** Larger tiles run out of
registers / shared memory on B200 (sm_100, 228 KB shmem, 64 K registers
per CTA).

| Operator shape | Reasonable upper bound |
|---|---|
| 1D elementwise / pointwise | `BLOCK_SIZE` ≤ **8192** for fp16/fp32; ≤ **16384** for int8 |
| Matmul tile area | `BLOCK_M × BLOCK_N` ≤ **256 × 256** for fp16, ≤ **128 × 256** for fp32, ≤ **256 × 256** for fp8/int8 |
| Matmul K-tile | `BLOCK_K` ≤ **128** for fp16/fp8/int8, ≤ **64** for fp32 |

**Recommended sweep ranges (start here, prune if 30-cfg budget is tight):**

| Use case | Recommended config space |
|---|---|
| 1D pointwise (Triton) | `BLOCK_SIZE` ∈ {512, 1024, 2048, 4096}, `num_warps` ∈ {4, 8} |
| 1D pointwise (cuTile) | `tile` ∈ {512, 1024, 2048, 4096}, `occupancy` ∈ {4, 8, 16} |
| Per-row reduction (Triton, 1 CTA per row) | `BLOCK_N` ∈ {256, 512, 1024, 2048}, `num_warps` ∈ {2, 4, 8} |
| Matmul (Triton, fp16/bf16) | `BLOCK_M`,`BLOCK_N` ∈ {64, 128, 256}, `BLOCK_K` ∈ {32, 64, 128}, `num_warps` ∈ {4, 8}, `num_stages` ∈ {2, 3, 4} — prune to ≤ 30 cfgs |
| Matmul (cuTile, fp16/bf16) | `tm`,`tn` ∈ {64, 128, 256}, `tk` ∈ {32, 64, 128}, `occupancy` ∈ {4, 8, 16}, `group_size_m` ∈ {8} |

**Why this matters.** A `BLOCK_SIZE=128` config for a 20M-element fp32
pointwise has only 32 elements per warp (1 element per thread with
`num_warps=4`), missing vectorised loads, missing memory coalescing,
and forcing 156× more CTAs than `BLOCK_SIZE=2048`. The autotune
sweep will dutifully benchmark this config and reject it, but every
small config you include costs roughly 1-15 seconds of compile +
benchmark time per problem shape — for matmul, that adds tens of
minutes per iteration with no chance of a win.

## Multi-dtype support

If `config.yaml`'s `case_grid.dtype` lists multiple dtypes (e.g.
`["fp16", "bf16", "fp32"]`), your `run()` function must work for **all
of them** in a single `run()` call. Two acceptable patterns:

1. **Single kernel, dtype-polymorphic** — Triton's `tl.dot` and cuTile's
   `ct.mma` adapt to input dtype automatically. The cleanest path.
2. **Per-dtype branches inside `run()`** — if e.g. fp8 needs different
   cast logic, branch on `a.dtype` in the Python wrapper, NOT in two
   separate kernels.

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
\textbf{not} as a substitute for the operator), stream / event
management (`torch.cuda.current_stream`, `torch.cuda.synchronize`), and
dtype-only casts on metadata (\textbf{not} on the data path that should
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
skip the kernel launch. Triton's per-cfg compilation cache and cuTile's
autotune cache are fine (they cache the **compiled kernel + best cfg**, not
the **output tensor**).

## Common pitfalls (the harness rejects these)

1. **Importing modules you didn't list**: stay within `torch`, `triton`,
   `triton.language`, `cuda.tile`, `cuda.tile_experimental` (optional),
   `core.cutile_autotune`, `math`, `numpy as np`.
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
2. Calls `impl_torch.run(*inputs)` → reference output.
3. Calls `impl_triton.run(*inputs, autotune=True)` → triton output.
4. `verify(triton_output, ref_output, atol, rtol)` with per-dtype tolerances
   (see `core/verifier.py`). On failure, no timing is collected for that backend.
5. If verify passes: times `impl_triton.run(*inputs, autotune=True)` over
   `repeat=100` iterations using NVIDIA Proton. Mean latency is recorded.
6. Same flow for `impl_cutile.run()`.

The metric for stopping is the **geometric mean of `roofline_pct` across
all `(backend, dtype, case)` combinations** ≥ 0.80, where `roofline_pct`
= measured_FLOPS_per_sec / `min(peak_compute_dtype, peak_bw × AI)`.

## TL;DR checklist before you return code

- [ ] Two files: `impl_triton.py` and `impl_cutile.py`
- [ ] Each exports `run(...)` and `get_last_config() -> dict | None`
- [ ] `run()` signature matches `impl_torch.run()` exactly
- [ ] Triton autotune via `triton.autotune` decorator, ≤ 30 cfgs
- [ ] cuTile autotune via `CutileAutotuner`, ≤ 30 cfgs
- [ ] No `_DEFAULT_CONFIG`, no `global` keyword for `_last_autotune_config`
- [ ] Multi-dtype handled in a single `run()`
- [ ] OOB / non-pow-2 handled with masks (Triton) or `padding_mode` (cuTile)
- [ ] No in-place mutation of inputs
