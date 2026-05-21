# Task: implement operator `vector_add` for TileBench

Follow the framework conventions below. Then implement the operator described.

---

# TileBench Framework Conventions

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
- Always include `num_warps` (∈ {2,4,8}) and `num_stages` (∈ {1,2,3,4}) in cfgs.

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


---

# Operator description: `vector_add`

# Vector Addition

Implement a program that performs element-wise addition of two 1D arrays of the same length.

The input consists of two arrays:

- `x`: A 1D array.
- `y`: A 1D array of the same length as `x`.

The output should be written to the `output` array, which has the same length as the inputs.

The operation is defined mathematically as:

$$
output[i] = x[i] + y[i]
$$

where $i$ ranges from $0$ to $n-1$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`

## Example 1

```text
Input: x = [1, 2, 3, 4], y = [5, 6, 7, 8]
Output: [6, 8, 10, 12]
```

## Example 2

```text
Input: x = [-1.5, 2.0, 0.0], y = [0.5, -2.0, 3.0]
Output: [-1.0, 0.0, 3.0]
```

## Constraints

- $1 \leq n$
- `x` and `y` have the same length


---

# `config.yaml` for `vector_add`

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
  # vector_add: 1 add per element (memory-bound)
  flops_expr: "n"
  # 2 reads + 1 write per element
  bytes_expr: "n * dtype_size * 3"

  plots:
    - latency_ms
    - bandwidth_GBs
    - speedup

```

Available case variables: see `case_grid` and `case_defaults` above. Your kernels must work for all combinations.

---

# PyTorch reference: `impl_torch.py`

Your generated kernels must match this reference's outputs within the tolerances in `verify:` (or per-dtype defaults if absent).

```python
def run(x, y):
    return x + y

```

---

## Output format (STRICT)

Return EXACTLY two fenced code blocks, in this order, with these titles:

    ```python title="impl_triton.py"
    # full Python file content here
    ```

    ```python title="impl_cutile.py"
    # full Python file content here
    ```

No other code blocks. No prose between or after the two blocks beyond a 1-2
sentence summary of your approach. Do NOT include test code, do NOT include
PyTorch reference code (that file is provided by the framework).
