---
name: tilelang-reference
dsl: tilelang
version: "0.1.11"
kind: reference
derived_from: ~/.claude/skills/tilelang-guide/SKILL.md@58216107 + installed-package introspection
revised: 2026-10-05
---

# TileLang 0.1.11 API Reference

TileLang (pip `tilelang`) is a TVM-based tile-level DSL for GPU kernels. Everything below was checked against the installed `tilelang==0.1.11` package (`inspect.signature`, docstrings, and `tilelang/language/*.py`, `tilelang/jit/*.py`, `src/op/copy.cc`, `cuda/pipeline.py`). The wheel vendors its own TVM fork; the TIR namespace is `tvm.tirx`. The pin `apache-tvm-ffi==0.1.11` is load-bearing: 0.1.12 makes `import tilelang` abort at C++ level.

Conventions: `import tilelang`, `import tilelang.language as T`. Autotuning (`tilelang.autotune`, `AutoTuner`, `set_autotune_inputs`, config sweeps) is not permitted under this protocol and is not documented here.

## 1. Kernel definition: `tilelang.jit`

```
tilelang.jit(func=None, *, out_idx: list[int] | int | None = None,
             target: str | Target | None = None, target_host=None,
             execution_backend=None, verbose=None,
             pass_configs: dict[str, Any] | None = None,
             debug_root_path: str | None = None,
             compile_flags: list[str] | str | None = None)
```

Two authoring styles; the mode is inferred automatically (`JITImpl._infer_jit_mode`):

| Mode | How it is recognised | What a call returns |
|---|---|---|
| **lazy** | the decorated function defines an inner `@T.prim_func` and returns it | a compiled `JITKernel`; you invoke it separately |
| **eager** | the body re-annotates parameters (`A: T.Tensor(...)`) and opens `T.Kernel`; no `return` of a PrimFunc | compiles (cached) and executes immediately |

Rules:
- `out_idx` is accepted **only in lazy mode**. In eager mode it raises `ValueError("out_idx is only supported in lazy mode. In eager mode, use T.empty() to declare output tensors instead.")`.
- Lazy mode: the decorated function's arguments are compile-time constants (shapes, tile sizes, dtype strings); one kernel is compiled per distinct argument tuple and cached in-process.
- Eager mode: arguments are the tensors plus any scalar keyword arguments; a compiled kernel is cached per key (see §2).
- `target`: `"auto"` (default when `None`) resolves to the CUDA target of the visible torch device; the arch string is `sm_<major><minor>` with suffix `a` when `major >= 9` (`contrib/nvcc.get_target_arch`). Explicit targets: `"cuda"`, `{"kind": "cuda", "arch": "sm_90"}`, `"hip"`, `"llvm"`, `"metal"`, `"cutedsl"`.
- `execution_backend`: `"tvm_ffi"` (default), `"cython"`, `"nvrtc"`, `"torch"` (Metal only), `"cutedsl"`.
- `pass_configs`: a `{tilelang.PassConfigKey.X: value}` dict (§10). `T.annotate_pass_configs({...})` and `T.annotate_compile_flags([...])` set the same things from inside the body; external values override body values.
- `debug_root_path`: writes `tilelang_jit_kernel_<name>.c` and `tilelang_jit_program_<name>.py` for inspection.

`tilelang.compile(func: PrimFunc, out_idx=None, execution_backend=None, target=None, target_host=None, verbose=None, pass_configs=None, compile_flags=None) -> JITKernel` compiles a bare `@T.prim_func` without the decorator.

### 1.1 Calling a compiled kernel (`JITKernel`)

`kernel(*tensors)` calls `kernel.torch_function`. The adapter checks `len(inputs) == len(params) - len(out_idx)`; for every index in `out_idx` it allocates `torch.empty(shape, dtype, device)` (device = the first tensor input's device, else torch's current device) and returns it (one tensor, or a list for several). Output shapes that contain dynamic variables are resolved from the input tensors' shapes/strides that bind the same variable. 0-dimensional outputs are rejected. Positional order is the PrimFunc parameter order with the `out_idx` parameters removed. Useful members: `kernel.get_kernel_source()`, `kernel.show_source("kernel"|"host"|"both")`, `kernel.out_idx`, `kernel.params`.

### 1.2 Minimal lazy kernel

```python
@tilelang.jit(out_idx=[-1])
def scale(N, block, dtype="float32"):
    @T.prim_func
    def kernel(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
            for i in T.Parallel(block):
                if bx * block + i < N:
                    y[bx * block + i] = x[bx * block + i] * 2
    return kernel

k = scale(N=10000, block=1024)      # JITKernel, compiled once per (N, block, dtype)
y = k(x)                            # y allocated by out_idx=[-1]
```

### 1.3 Minimal eager kernel

```python
@tilelang.jit
def scale(x, y, dtype: str, block: int = 1024, threads: int = 128):
    n = T.dynamic("n")              # runtime-symbolic length
    x: T.Tensor((n,), dtype)
    y: T.Tensor((n,), dtype)
    with T.Kernel(T.ceildiv(n, block), threads=threads) as bx:
        for i in T.Parallel(block):
            if bx * block + i < n:
                y[bx * block + i] = x[bx * block + i] * 2

scale(x, y, dtype="float16")        # executes; y is caller-allocated
```

Eager-mode outputs are either caller-allocated tensors written in place (as above) or declared with `T.empty(shape, dtype)` and **returned** from the function (every `T.empty` tensor must be returned, else `RuntimeError("Not all tensor allocated from T.empty are returned")`).

## 2. Symbolic vs constant shapes and the compile cache

| Primitive | Semantics |
|---|---|
| `T.dynamic(name, dtype="int32")` | Runtime-symbolic variable. One compiled kernel serves every value. `"M, N"` or `"M N"` returns a tuple. |
| `T.symbolic(...)` | Deprecated alias of `T.dynamic` since v0.1.9 (still works, warns). |
| `T.const(name, dtype="int32")` | Eager mode only. Value is read from the matching tensor argument's shape at call time and baked in as a compile-time constant. Raises `JITNoBuilderError` outside eager `@tilelang.jit`. |

Eager cache key (`JITFunc.parse_args`): **phase 1** = all non-tensor arguments (every scalar/str keyword, including tile sizes and dtype strings); **phase 2** = the concrete values of every `T.const` variable, taken from the tensor arguments' shapes/strides (`TirTemplate._parse_phase2_key`). `T.dynamic` dimensions are not part of the key. Consequences:
- A `T.const` dimension recompiles once per distinct value; a `T.dynamic` dimension does not.
- Any scalar argument forks compilation, so pass problem sizes through tensor shapes (`T.dynamic`) rather than as Python ints unless constant folding is intended.
- Lazy mode keys on the decorator-function arguments (`(args, sorted kwargs)`).

## 3. `T.Kernel`: grid, block, indices

```
T.Kernel(*blocks, threads: int | list[int] | tuple | None = None,
         cluster_dims: int | tuple[int, int, int] | None = None,
         is_cpu: bool = False, prelude: str | None = None)
```

- `blocks`: 1 to 3 grid extents (gridDim.x, y, z). The context yields **one `Var` for a 1-D grid** and a **list `[bx, by, bz]`** otherwise (`_normalize_bindings`): `with T.Kernel(gx) as bx:` / `with T.Kernel(gx, gy) as (bx, by):`.
- `threads`: blockDim.x as an int, or a list/tuple for 2-D/3-D blocks. **Defaults to 128 when omitted** (`kernel.py`). `-1` skips the threadIdx.x binding.
- Thread index: `T.get_thread_binding(dim=0)` / `T.get_thread_bindings()`; block index also via `T.get_block_binding(dim)` / `T.get_block_bindings()`; `T.get_block_extents()`.
- `cluster_dims`: thread-block cluster launch; requires a target that supports clusters (the device context states whether it does).
- Lane/warp helpers: `T.get_lane_idx()`, `T.get_warp_idx()`, `T.get_warp_group_idx()`.

## 4. Tensors, dtypes, on-chip buffers

**Global tensors (kernel parameters).** `T.Tensor(shape, dtype="float32")` (contiguous, row-major strides); `T.StridedTensor(shape, strides, dtype)`; `T.Buffer(shape, dtype, ..., scope="global")`. Both call and subscript forms are accepted: `T.Tensor((M, N), "float16")` and `T.Tensor[[M, N], T.float16]`. Scalar parameters are annotated with a dtype object, e.g. `n: T.int32`.

**Dtype names.** Strings or the `T.<name>` dtype objects: `"bool"`, `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`, `"float16"`, `"bfloat16"`, `"float32"`, `"float64"`, `"tfloat32"`, `"float8_e4m3fn"`, `"float8_e4m3fnuz"`, `"float8_e5m2"`, `"float8_e5m2fnuz"`, `"float8_e8m0fnu"`, `"float4_e2m1fn"` (packed) and `"float4_e2m1_unpacked"`. Vector forms `float16x2`, `float32x4`, … also exist. torch dtypes map by `str(t.dtype).removeprefix("torch.")` (`torch.float16 -> "float16"`, `torch.bfloat16 -> "bfloat16"`); `T.float8_e4m3fn`/`T.float8_e5m2` are the CUDA fp8 names. Narrow float types are only usable where the target supports them.

**Allocation (inside `T.Kernel`).**

| Call | Scope | Use |
|---|---|---|
| `T.alloc_shared(shape, dtype, scope="shared.dyn")` | shared memory | block-wide tiles, `T.copy` staging, GEMM operands |
| `T.alloc_fragment(shape, dtype)` | `local.fragment` (registers, thread-distributed) | GEMM accumulators, reduction inputs/outputs, elementwise tiles |
| `T.alloc_local(shape, dtype)` | `local` (per-thread) | small per-thread arrays |
| `T.alloc_var(dtype, init=None, scope="local.var")` | single scalar | counters, running max/sum |
| `T.alloc_barrier(arrive_count: int | list[int])` | mbarrier(s) | manual async copy / MMA synchronisation |
| `T.alloc_tmem(shape, dtype)` | `shared.tmem` | tensor-memory accumulator; `shape` must be 2-D. Requires a target with TMEM (the device context states whether it has it); column count must be a power of two, ≥ 32, ≤ 512 |
| `T.alloc_reducer(shape, dtype, op="sum"|"max"|"min")` | thread-private partials | accumulate inside `T.Parallel`, then `T.finalize_reducer(reducer)` |
| `T.empty(shape, dtype)` | global (eager output) | must be returned from the eager function |

Fragment/shared buffers have fixed, compile-time shapes. Layouts are inferred (`LayoutInference`); `T.annotate_layout({buf: layout})` overrides, `T.reshape(buf, shape)` / `T.view(buf, shape=None, dtype=None)` give views without copies.

## 5. `T.copy` and tile data movement

```
T.copy(src, dst, *, coalesced_width=None, disable_tma=False,
       eviction_policy: "evict_normal"|"evict_first"|"evict_last"|None = None,
       prefer_instruction: str | None = None, annotations=None, loop_layout=None)
```

Operand forms (either side): a whole `Buffer`; a `BufferRegion` slice `A[r0:r1, c0:c1]`; or the head-address sugar `A[i, j]` (a scalar `BufferLoad`), whose extents are taken from the other side. Extent rules (`copy_op._normalize_copy_regions`):
- `Buffer -> shape`, `BufferRegion -> [r.extent]` (so `A[y0:y1]` contributes `y1 - y0`), `BufferLoad -> extents inferred from its encoded region`.
- Two whole `Buffer`s must have structurally equal shapes.
- Missing extents are treated as length-matched from the tail; this is a limited sugar, not broadcasting. When the two sides' extents differ, the lowering picks one side as the base range and "may generate unexpected code" (docstring). Keep both sides the same extent; do not use a shorter slice on one side to bound a copy.
- Two scalar `BufferLoad`s lower to a plain store `dst[...] = src[...]`.

Out-of-bounds semantics (`src/op/copy.cc::MakePredicate`, SIMT path): per-dimension predicates against the **buffer's real shape** are generated only when the analyzer cannot prove the access in bounds (no cost for divisible tiles). OOB **reads** become `if_then_else(pred, load, 0)` — destination lanes are **zero-filled**; OOB **writes** are skipped. A tile pre-filled with another value (e.g. `-inf`) and then written by `T.copy` on a ragged edge therefore ends with `0`, not the pre-fill, in the OOB lanes. For a non-zero pad value, write the guarded element loop yourself (§7) or mask after the copy.

`T.annotate_safe_value({buf: value})` sets the fill value used by the `LegalizeSafeMemoryAccess` pass, which guards **direct global** `BufferLoad`/`BufferStore` (and cp.async fills) that may be out of bounds; shared/local accesses are at most warned about, never guarded. This pass runs after tile-op lowering (`cuda/pipeline.py`), so it is not a guarantee that a `T.copy` pad value changes; verify on the target before relying on it. `PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS` turns the pass off.

Options: `disable_tma=True` forces a non-TMA lowering; `prefer_instruction="tma"|"cp_async"|"sync"` picks the CUDA lowering class (TMA keeps synchronous copy semantics with an auto-allocated barrier); `eviction_policy` is an L2 hint; `coalesced_width` sets the vector width of the SIMT loop; `loop_layout` (a `T.Fragment`) is only valid for the SIMT path.

Related: `T.async_copy(src, dst)` (cp.async, no wait inserted; synchronise explicitly), `T.tma_copy(src, dst, *, barrier=...)` (user-managed TMA; `barrier` required for loads; wait with `T.mbarrier_wait_parity`, stores with `T.tma_store_wait()`), `T.transpose(src, dst)`, `T.fill(buf_or_region, value)`, `T.clear(buf)` (fill with zero).

## 6. `T.gemm`

```
T.gemm(A, B, C, transpose_A=False, transpose_B=False,
       policy=T.GemmWarpPolicy.Square, clear_accum=False, k_pack=1, mbar=None)
```

- `A`, `B`: shared or fragment buffers (or regions of them). `C`: the accumulator, a fragment buffer (or a TMEM buffer on targets that have it). `C` must be 2-D; `A`/`B` may have leading unit dimensions.
- Shape contract with `M, N = C.shape`: `A` is `(M, K)` or, with `transpose_A=True`, `(K, M)`; `B` is `(K, N)` or, with `transpose_B=True`, `(N, K)`. `K` must match (asserted at trace time).
- `clear_accum=True` zeroes `C` before the multiply; otherwise `C += A @ B`. Accumulate in a wider dtype than the inputs when precision matters (e.g. `float32` accumulator for `float16`/`bfloat16` inputs).
- `policy`: `GemmWarpPolicy.Square` (0), `FullRow` (1), `FullCol` (2): how warps are partitioned over the `M x N` tile.
- `k_pack`: ROCm only.
- `mbar`: required when the GEMM lowers to the asynchronous tensor-memory MMA on a target that has it; for the ordinary synchronous lowering leave it `None`.
- `T.gemm` is synchronous: the compiler inserts the matching wait after the issued MMA. Manual asynchronous variants exist (`T.wgmma_gemm` with `T.wait_wgmma`/`T.warpgroup_wait`; `T.tcgen05_gemm` with `T.mbarrier_wait_parity`) and are target-specific; use them only when the device context says the target supports the corresponding instruction class.
- Instruction selection is automatic from the target (`TL_DISABLE_WGMMA`, `TL_DISABLE_TMA_LOWER`, `TL_DISABLE_WARP_SPECIALIZED` in §10 constrain it).

## 7. Loops and control flow

- **`T.Parallel(*extents, coalesced_width=None, loop_layout=None, prefer_async=None, annotations=None)`**: a nested parallel loop distributed over the block's threads and auto-vectorised. The elementwise idiom is `for i, j in T.Parallel(BM, BN): C[i, j] = f(A[i, j])`. Every parallel nest needs a layout after inference; it is inferred unless `loop_layout` (a `T.Fragment` with `InputDim == nest depth`) is given. Only the outermost loop of a nest carries the annotation.
- **`T.Pipelined(start, stop=None, num_stages=0, order=None, stage=None, sync=None, group=None)`**: a software-pipelined loop (typically the K loop). `num_stages=0` **disables pipelining**; `num_stages=k` lets the compiler multi-buffer the loads against the compute (`k` buffers, more shared memory). For manual schedules give `order`/`stage` (depth = `max(stage) + 1`) and do not also set `num_stages`.
- `T.serial(start, stop=None, step=None)`, `T.unroll(start, stop=None, step=None, explicit=False, unroll_factor=None)`, `T.vectorized(start, stop=None)`. Plain `range(...)` inside a kernel body is rewritten to `T.serial`.
- `T.Persistent(domain, wave_size, index, group_size=8)`: persistent-block tile loop.
- Python `if`/`elif`/`else` and `while` lower to predication/branches; conditions may use loop variables, thread bindings, and `T.dynamic` variables. `T.loop_break()` exists.
- Expression-level selection: `T.if_then_else(cond, t, f)` evaluates only the taken branch (use it to guard OOB reads; cannot be vectorised when lanes disagree); `T.Select(cond, t, f)` may evaluate both.
- `T.ceildiv(a, b)`, `T.floordiv`, `T.floormod`, `T.truncdiv`, `T.truncmod`.
- `T.sync_threads()` (`__syncthreads`), `T.sync_warp(mask=None)`, `T.barrier_arrive(mbar)`, `T.barrier_wait(mbar, parity)` / `T.mbarrier_wait_parity(mbar, parity)`; `T.ws(*warp_group_idx)` opens a warp-specialised region.

Guarded element loop (the explicit alternative to relying on `T.copy` padding):

```python
for i in T.Parallel(BLOCK):
    idx = start + i
    x_local[i] = T.if_then_else(idx < n, x[idx], -T.infinity(dtype))
```

## 8. Reductions and scans

```
T.reduce_sum(buffer, out, dim=-1, clear=True, batch=1)
T.reduce_max / T.reduce_min / T.reduce_absmax(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False)
T.reduce_abssum(buffer, out, dim=-1, batch=1)          # no clear parameter
T.reduce_bitand / T.reduce_bitor / T.reduce_bitxor(buffer, out, dim=-1, clear=True, batch=1)
```

- Operands are fragment or shared buffers (shared inputs/outputs are staged through fragments internally). The reduction is over dimension `dim` of `buffer`.
- **Out-shape contract**: `out.shape` must equal `buffer.shape` with `dim` removed, or with `dim` set to 1; anything else raises `ValueError` at trace time.
- `clear=True` initialises `out` (`-inf` for max, `+inf` for min, 0 for sum) before reducing; `clear=False` accumulates into the existing contents of `out`, which the caller must have initialised.
- `nan_propagate` applies to `float16`/`bfloat16` max/min only (`__hmax_nan`/`__hmin_nan` vs `__hmax`/`__hmin`).
- Scalar-per-thread warp reductions: `T.warp_reduce_sum/max/min/bitand/bitor(value)`.
- Reducer buffers: `T.alloc_reducer` + `T.fill(reducer, identity)` + updates inside `T.Parallel` (`reducer[...] += v` for sum, `reducer[...] = T.max(reducer[...], v)` for max) + `T.finalize_reducer(reducer)`.
- Scans: `T.cumsum(src, dst=None, dim=0, reverse=False)` and `T.cummax(...)`; `dst=None` is in place; accept buffers or regions.

## 9. Math, casting, atomics, RNG, annotations

- Elementwise math on expressions: `T.exp`, `T.exp2`, `T.exp10`, `T.log`, `T.log2`, `T.log10`, `T.log1p`, `T.sqrt`, `T.rsqrt`, `T.abs`, `T.tanh`, `T.sigmoid`, `T.erf`, `T.pow(x, y)`, `T.floor`, `T.ceil`, `T.round`, `T.trunc`, `T.sin`, `T.cos`, `T.tan`, `T.isnan`, `T.isinf`, `T.isfinite`; two-argument `T.max(a, b)`, `T.min(a, b)`; constants `T.infinity(dtype)`, `T.max_value(dtype)`, `T.min_value(dtype)`. Bit ops: `T.bitwise_and/or/xor/not`, `T.shift_left`, `T.shift_right`, `T.popcount`, `T.clz`. `float16`/`bfloat16` intrinsics have native device implementations (`hexp`, `__habs`, `hrsqrt`, … in `tl_templates/cuda/common.h`); `PassConfigKey.TL_ENABLE_FAST_MATH` switches fp32 transcendental calls to the fast variants.
- Casting: `T.cast(value, dtype)` (value-first; optional PTX `round="rs"`, `sat`), `T.Cast(dtype, value)` (dtype-first TIR node), `expr.astype(dtype)` (TVM expression method), `T.reinterpret(value, dtype)` (bit cast).
- Atomics on global/shared buffers: `T.atomic_add(dst, value, memory_order=None, return_prev=False, use_tma=False)`, `T.atomic_max`, `T.atomic_min`, `T.atomic_addx2`, `T.atomic_addx4`, `T.atomic_load(src, memory_order="seq_cst")`, `T.atomic_store(dst, src, memory_order="seq_cst")`. `dst` may be a scalar element or a whole buffer/region (tile atomic); `use_tma=True` needs a target with bulk-reduce support.
- RNG (CUDA cuRAND): `T.rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t")`, `T.rng_rand() -> uint32`, `T.rng_rand_float(bit=32|64, dist="uniform"|"normal")`.
- Warp intrinsics: `T.shfl_sync`, `T.shfl_up`, `T.shfl_down`, `T.shfl_xor`, `T.ballot`, `T.any_sync`, `T.all_sync`.
- Kernel-level annotations (inside `T.Kernel`): `T.use_swizzle(panel_size, order="row"|"col")` (threadblock rasterisation), `T.annotate_layout({buf: Layout|Fragment|callable})`, `T.annotate_safe_value({buf: value})`, `T.annotate_min_blocks_per_sm(n)` (`__launch_bounds__` second argument), `T.annotate_l2_hit_ratio({global_buf: ratio})`, `T.annotate_restrict_buffers(*bufs)` (drop `__restrict__` for aliasing parameters).
- Debug: `T.print(buffer_or_expr, msg="")`, `T.device_assert(cond)`.

## 10. Pass configuration (`tilelang.PassConfigKey`)

Pass as `@tilelang.jit(pass_configs={tilelang.PassConfigKey.KEY: value})` or `T.annotate_pass_configs({...})` in the body. Keys present in 0.1.11 that affect generated code:

| Key | Effect |
|---|---|
| `TL_DISABLE_WARP_SPECIALIZED` | disable the producer/consumer warp-specialised lowering |
| `TL_DISABLE_TMA_LOWER` | never lower copies through TMA |
| `TL_DISABLE_WGMMA` | never lower `T.gemm` through warpgroup MMA |
| `TL_ENABLE_FAST_MATH` | fast fp32 transcendental intrinsics |
| `TL_DISABLE_SAFE_MEMORY_ACCESS` | skip `LegalizeSafeMemoryAccess` (no automatic global bounds guards) |
| `TL_DISABLE_OUT_OF_BOUND_WARNING` | silence shared/local OOB warnings |
| `TL_DISABLE_THREAD_STORAGE_SYNC` | do not auto-insert `__syncthreads` for shared-memory hazards |
| `TL_DISABLE_VECTORIZE_256` / `TIR_DISABLE_VECTORIZE` | limit or disable vectorisation |
| `TL_ENABLE_ASYNC_COPY` / `TL_ENABLE_LOWER_LDGSTG` | enable cp.async / LDG-STG lowering paths |
| `TL_PTXAS_REGISTER_USAGE_LEVEL`, `TL_ENABLE_PTXAS_VERBOSE_OUTPUT`, `TL_DEVICE_COMPILE_FLAGS` | ptxas / nvcc flags |
| `TL_CONFIG_INDEX_BITWIDTH` | index bit width (default narrowing to 32) |
| `TL_DISABLE_SHARED_MEMORY_REUSE`, `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` | shared-memory allocation merging |
| `TL_ENABLE_DUMP_IR`, `TL_DUMP_IR_DIR`, `CUDA_KERNELS_OUTPUT_DIR` | debugging output |

Whether a disabled feature would otherwise be used depends on the target; the device context states which instruction classes (TMA, warpgroup MMA, tensor memory, clusters) the target supports.

## 11. Legality checklist

- Eager mode: no `out_idx`; outputs are caller tensors or returned `T.empty` tensors. Lazy mode: inner `@T.prim_func` returned; outputs via `out_idx`.
- Shared/fragment/local buffer shapes are compile-time constants; global tensor shapes may use `T.dynamic` variables.
- `T.copy` sides must have matching extents; OOB reads zero-fill, OOB writes are skipped, predicates are against the real buffer shape.
- `T.gemm`: `C` 2-D fragment (or TMEM) accumulator, `A`/`B` shared or fragment, shapes consistent with the transpose flags, `K` equal.
- `reduce_*`: `out.shape` = input shape minus `dim` (or `dim` set to 1); initialise `out` yourself when `clear=False`.
- `T.Pipelined(..., num_stages=0)` means no pipelining; do not combine `num_stages` with manual `order`/`stage`.
- `T.Kernel` without `threads` runs 128 threads; a 1-D grid yields a bare `Var`, multi-D yields a list.
- Shared/local OOB is never guarded automatically; guard with `if` / `T.if_then_else` or pad tiles.
- Narrow floats (fp8, fp4), TMEM, clusters, TMA, warpgroup MMA are capability-dependent; the device context states what the target supports.
- `T.const` and every scalar argument fork the compile cache; `T.dynamic` does not.

## 12. Example: tiled GEMM (lazy, 2-D grid, pipelined K loop)

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, BM, BN, BK, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), dtype)
            B_s = T.alloc_shared((BK, BN), dtype)
            C_f = T.alloc_fragment((BM, BN), accum_dtype)
            T.clear(C_f)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[k * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * BM, bx * BN])
    return kernel
```

## 13. Example: row softmax-style reduction (eager, dynamic rows)

```python
@tilelang.jit
def row_max(x, out, dtype: str, BN: int, threads: int = 128):
    M = T.dynamic("M")
    x: T.Tensor((M, BN), dtype)
    out: T.Tensor((M,), dtype)
    with T.Kernel(M, threads=threads) as row:
        x_f = T.alloc_fragment((BN,), dtype)
        m_f = T.alloc_fragment((1,), dtype)
        T.copy(x[row, 0], x_f)
        T.reduce_max(x_f, m_f, dim=0, clear=True)
        out[row] = m_f[0]
```
