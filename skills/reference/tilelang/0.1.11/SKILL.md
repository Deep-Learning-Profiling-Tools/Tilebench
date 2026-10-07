---
name: tilelang-reference
dsl: tilelang
version: "0.1.11"
kind: reference
upstream: https://github.com/tile-ai/tilelang (tag v0.1.11, commit cd37ed5fc35ae7a60a1277c8eb49028174ac51e6)
upstream_license: MIT, Copyright (c) Tile-AI
revised: 2026-10-06
---

# TileLang 0.1.11 Language Reference

TileLang (`tilelang`) is a Python-embedded tile-level DSL that compiles through a TVM-based compiler to GPU kernels. A kernel is written at the level of one **thread block**: it allocates tiles in shared memory and registers, moves tiles with `T.copy`, computes on them with tile operators (`T.gemm`, `T.reduce_*`, element-wise `T.Parallel` loops), and writes results back; the compiler maps the work onto the block's threads.

Scope: the API of TileLang **0.1.11** (`tilelang.__version__ == "0.1.11"`). Every `T.*` / `tilelang.*` name below was checked by introspection of the installed package and located in the v0.1.11 sources; treat unlisted names as unavailable (section 19 lists known absent ones). Hardware figures are not part of this document. Tile sizes in snippets are arbitrary illustrations.

`tilelang.autotune` and `tilelang.autotuner.AutoTuner` exist in 0.1.11 but are not documented here: the benchmark's generated files must use fixed literal configurations (no autotuning or runtime configuration search).

---

## 1. Imports and Namespaces

```python
import torch
import tilelang
import tilelang.language as T
```

| Namespace | Purpose |
|-----------|---------|
| `tilelang` | host API: `tilelang.jit`, `tilelang.compile`, `tilelang.JITKernel`, `tilelang.PassConfigKey` |
| `tilelang.language` (`T`) | everything used inside a kernel (types, allocation, copies, loops, math, GEMM, ...) |
| `tilelang.layout` | layout helpers for `T.annotate_layout` |

---

## 2. Data Types

### 2.1 Spelling a dtype

A dtype can be given as a string (`"float16"`), a TileLang dtype object (`T.float16`) or a torch dtype (`torch.float16`); TileLang normalizes all three. `T.float16` etc. are `str` subclasses: `T.float32 == "float32"` is `True`.

| Kind | Names (all on `T`) |
|------|---------------------|
| Boolean | `T.bool` |
| Signed int | `T.int8`, `T.int16`, `T.int32`, `T.int64` |
| Unsigned int | `T.uint8`, `T.uint16`, `T.uint32`, `T.uint64` |
| Float | `T.float16`, `T.bfloat16`, `T.float32`, `T.float64` |
| FP8 | `T.float8_e4m3fn`, `T.float8_e5m2`, `T.float8_e4m3`, `T.float8_e4m3fnuz`, `T.float8_e5m2fnuz`, `T.float8_e8m0fnu`, ... |
| FP6 / FP4 | `T.float6_e2m3fn`, `T.float6_e3m2fn`, `T.float4_e2m1fn` |
| Aliases | `T.half` = float16, `T.float` = float32, `T.double` = float64, `T.short` = int16, `T.int` = int32, `T.long` = int64, `T.uint` = uint32 |
| Vector packs | `<base>x2`, `x4`, `x8`, `x16`, `x32`, `x64`, e.g. `T.float16x2`, `T.int8x4` |

Availability of FP8/FP6/FP4 operations depends on target and backend support (docs, Type System).

### 2.2 Casts

```python
y = T.cast(x, T.float32)        # explicit conversion of an expression
y = x.astype(T.float32)         # method form on an expression / buffer element
```

`T.cast(value, dtype, round="", sat=True, rbits=None)` also carries optional PTX rounding hints; only `round=""` (default) and `"rs"` (stochastic, needs `rbits`) are lowered by the CUDA backend.

`T.copy` converts element types when source and destination dtypes differ (e.g. a `float32` fragment copied into a `float16` global tensor). Make conversions inside expressions explicit with `T.cast` / `.astype`.

---

## 3. Programming Model

### 3.1 Block-level programs

`with T.Kernel(...)` opens a launch: the grid extents give `gridDim`, `threads=` gives `blockDim`. Code inside the `with` block describes the work of **one block**; tile operators (`T.copy`, `T.gemm`, `T.reduce_*`, `T.fill`) and `T.Parallel` loops are distributed over the block's threads by the compiler. Any other statement at kernel scope (for example `Y[bx] = acc[0]`, including plain statements inside `T.serial` loops) is executed by **every thread** of the block.

```python
T.Kernel(*blocks, threads=None, cluster_dims=None, is_cpu=False, prelude=None)
```

- `blocks`: 1 to 3 grid extents (`gridDim.x`, `.y`, `.z`), ints or expressions such as `T.ceildiv(N, block_N)`.
- `threads`: an int (`blockDim.x`) or a list/tuple of up to 3 ints; **defaults to 128** when omitted on GPU.
- `cluster_dims`: SM90+ thread-block clusters, e.g. `2` or `(2, 1, 1)`.
- The `as` target is a single block-index variable for a 1-D grid, otherwise a list: `as bx` or `as (bx, by)` or `as (bx, by, bz)`.

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    ...            # bx = blockIdx.x, by = blockIdx.y
```

Thread / block indices when thread-level code is needed:

```python
tx = T.get_thread_binding()        # threadIdx.x (dim=0); T.get_thread_binding(1) -> threadIdx.y
bx = T.get_block_binding()         # blockIdx.x
```

Several `with T.Kernel(...)` blocks in one function are compiled as separate kernels that execute sequentially (README).

### 3.2 Two ways to define a kernel

`@tilelang.jit` accepts two styles and infers which one is used.

**Lazy style** (section 17.1) — the decorated function takes compile-time values and returns a `@T.prim_func`. Calling it returns a compiled `JITKernel` (cached per distinct argument tuple), which is then called with tensors.

**Eager style** (section 17.3) — the decorated function takes the tensors themselves, declares their shapes by annotation inside the body and declares outputs with `T.empty`. Calling it compiles (first use of a specialization) and **runs** the kernel, returning the `T.empty` outputs.

```python
@tilelang.jit(out_idx=[-1])                       # lazy
def add_one(N, block=128, dtype=T.float32):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                B[bx * block + i] = A[bx * block + i] + 1
    return main

kernel = add_one(4096)                            # JITKernel
b = kernel(a)

@tilelang.jit                                     # eager
def add_one_eager(A, block=128, dtype=T.float32):
    N = T.const("N")                              # bound from A.shape[0]
    A: T.Tensor((N,), dtype)                      # or A: T.Tensor[[N], dtype]
    B = T.empty((N,), dtype)
    with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
        for i in T.Parallel(block):
            B[bx * block + i] = A[bx * block + i] + 1
    return B

b = add_one_eager(a)                              # compiles if needed, runs, returns B
kernel = add_one_eager.compile(a)                 # or .compile(N=4096): JITKernel, not run
```

`out_idx` is only valid in lazy mode; in eager mode `T.empty` defines the outputs (passing `out_idx` raises `ValueError`).

### 3.3 `@tilelang.jit` parameters

```python
tilelang.jit(func=None, *, out_idx=None, target=None, target_host=None,
             execution_backend=None, verbose=None, pass_configs=None,
             debug_root_path=None, compile_flags=None)
```

| Parameter | Meaning |
|-----------|---------|
| `out_idx` | `int` or `list[int]` (negative allowed): parameters that are **outputs**; they are allocated by the call and returned (lazy mode only) |
| `target` | `"auto"` (default, from `TILELANG_TARGET`; detects CUDA, then HIP, then Metal), `"cuda"`, or a TVM target config dict `{"kind": "cuda", "arch": ...}` |
| `execution_backend` | `"auto"` (default) resolves to `"tvm_ffi"` for CUDA; also `"cython"`, `"nvrtc"`, `"cutedsl"`, `"torch"` (Metal only); `"dlpack"` is an alias of `"tvm_ffi"` |
| `pass_configs` | dict of compiler options keyed by `tilelang.PassConfigKey` members or their string values, e.g. `{tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}` |
| `compile_flags` | extra device-compiler flags (`str` or `list[str]`) |
| `debug_root_path` | directory where the generated kernel source and the TIR script are written at each compilation |
| `verbose` | verbose compilation (default from `TILELANG_VERBOSE`) |

`tilelang.compile(func, out_idx=None, execution_backend=None, target=None, target_host=None, verbose=None, pass_configs=None, compile_flags=None)` compiles an already-built `T.prim_func` into a `JITKernel` with the same options.

### 3.4 Kernel parameters

| Annotation | Meaning |
|------------|---------|
| `X: T.Tensor(shape, dtype)` | global-memory tensor, **contiguous row-major** (strides derived from `shape`) |
| `X: T.StridedTensor(shape, strides, dtype)` | global tensor with explicit strides |
| `n: T.int32`, `alpha: T.float32`, `flag: T.bool` | runtime scalar, passed as a Python number at call time |

`T.Tensor[(M, N), dtype]` (subscript form) is equivalent to `T.Tensor((M, N), dtype)`. `T.Buffer(...)` still works but is deprecated in favour of `T.Tensor(...)`.

### 3.5 Compile-time and symbolic sizes

| Size given as | Where | Effect |
|---------------|-------|--------|
| Python `int` | lazy-style parameter | constant baked into the kernel; a new value means a new compilation |
| `T.const("M, N")` | eager style only | names bound from the actual tensor shapes at call time and **specialized**: every new value compiles a new kernel |
| `T.dynamic("M")` | lazy or eager | runtime symbol: one compiled kernel serves every size; the value is bound from tensor shapes at call time and passed to the CUDA kernel as an `int` parameter |

```python
M = T.dynamic("M")               # tirx.Var, int32 by default
M, N = T.dynamic("M, N")         # several at once (comma or space separated)
M, N, K = T.const("M, N, K")     # eager style only
```

- `T.dynamic(name, dtype="int32")`. `T.symbolic` is a deprecated alias (since v0.1.9) of `T.dynamic`.
- A `T.const` name must appear **directly** as a dimension of some tensor shape or stride (`(M, K)`, not only `M * 2`), otherwise compilation raises `RuntimeError`.
- Symbols that appear only in expressions of other dims are resolved when a single unknown remains (host-side linear solving).

---

## 4. Calling a Compiled Kernel from Python

### 4.1 Argument order and outputs

A `JITKernel` is called with positional arguments in `@T.prim_func` parameter order, **skipping** the parameters listed in `out_idx`:

- The number of arguments must equal `len(params) - len(out_idx)`, otherwise `ValueError("Kernel expected N inputs, but M are provided.")`.
- `out_idx` parameters are allocated with `torch.empty` (uninitialized) on PyTorch's current CUDA device, with shapes taken from the declaration (symbolic dims are resolved from the inputs). 0-d outputs are not supported.
- Return value: the tensor when one output is declared, a `list` of tensors for several, and an empty list `[]` when `out_idx` is not used. Without `out_idx` the caller passes every tensor, including outputs, and the kernel writes into them.

### 4.2 Streams and devices

The default (`tvm_ffi`) adapter launches on **PyTorch's current CUDA stream** and device, read at call time, so `with torch.cuda.stream(s): kernel(...)` launches on `s` (`tilelang/jit/adapter/tvm_ffi.py`). All tensor arguments must be on the same device.

### 4.3 Kernel objects and caching

- Lazy style: `f(args)` returns the same `JITKernel` object for the same argument tuple (in-memory cache); compiled binaries are also cached on disk under `TILELANG_CACHE_DIR` (default `~/.tilelang/cache`).
- Eager style: `f(tensors, **params)` keys its cache on the non-tensor arguments plus the `T.const` values.

### 4.4 Wrapping a kernel in a plain Python function

```python
@tilelang.jit(out_idx=[-1])
def scale_add(block=256, dtype=T.float32):
    N = T.dynamic("N")

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype), alpha: T.float32,
             Z: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                Z[bx * block + i] = alpha * X[bx * block + i] + Y[bx * block + i]
    return main

_scale_add_kernel = scale_add()          # compiled once; N is bound at every call

def run(x, y, alpha=2.0):
    return _scale_add_kernel(x.contiguous(), y.contiguous(), alpha)
```

### 4.5 Host-side argument checks

The generated host stub validates every call: argument count, tensor rank, dtype (exact match; only `float8_e4m3`/`float8_e5m2` variants and `bool` have tolerances), each shape dimension against its constant or symbol, strides (a `T.Tensor` must be contiguous: a transposed or strided view fails), `byte_offset == 0`, device type, and equal device ids across tensors. Scalar parameters must have the declared kind (`T.int*` requires an integer, `T.bool` a boolean). Violations raise `RuntimeError` naming the parameter (e.g. `input A dtype mismatch, expected float32`).

---

## 5. Memory Scopes and Allocation

All allocations are made inside `T.Kernel`.

```python
T.alloc_shared(shape, dtype, scope="shared.dyn")       # block-shared tile (staging for T.gemm, data exchange)
T.alloc_fragment(shape, dtype, scope="local.fragment") # register tile of the block, split across its threads
                                                       # by layout inference (accumulators, reduction tiles)
T.alloc_local(shape, dtype, scope="local")             # thread-private array: every thread owns a full copy
T.alloc_var(dtype, init=None)                          # thread-private scalar (scope "local.var")
```

- Fragment elements are addressed with tile coordinates (`frag[i, j]`) inside `T.Parallel` loops and tile operators.
- `T.alloc_var` returns a 1-element buffer that is used like a scalar: `cnt = T.alloc_var(T.int32, init=0)`, then `cnt += 1`, `Y[0] = cnt`. Pass `init` (or assign) before the first read.
- `T.alloc_shared` with `dtype="bool"` is placed in scope `"shared"` instead of `"shared.dyn"`.
- `T.alloc_global(shape, dtype)` allocates a global workspace outside the PyTorch allocator; its docstring recommends allocating workspaces in PyTorch and passing them as arguments instead.
- `T.alloc_tmem(shape, dtype)` (Blackwell Tensor Memory) and `T.alloc_barrier(arrive_count)` (mbarriers) are covered in section 9.4.

Views that share storage (no copy; the total size in bits must match):

```python
T.reshape(buf, new_shape)              # same dtype
T.view(buf, shape=None, dtype=None)    # new shape and/or dtype
```

---

## 6. Data Movement

### 6.1 `T.copy`

```python
T.copy(src, dst, *, coalesced_width=None, disable_tma=False,
       eviction_policy=None, prefer_instruction=None, annotations=None, loop_layout=None)
```

`src` and `dst` may each be a whole buffer, a region (slice) or a single element used as a **start corner**:

```python
T.copy(A[by * BM, k * BK], A_shared)         # global -> shared; extent taken from A_shared's shape
T.copy(A_shared, A_frag)                     # shared -> fragment
T.copy(X[bx, 0:N], row)                      # explicit slice region
T.copy(A[bx * bm:(bx + 1) * bm, :], A_frag)  # 2-D slice
T.copy(C_frag, C[by * BM, bx * BN])          # fragment -> global; extent taken from C_frag
```

Rules (from `tilelang/language/copy_op.py`):
- Extents come from buffer shapes, slice extents, or the other operand when one side is a start corner. Both sides should describe the same extents; otherwise one side is picked as the iteration space ("may generate unexpected code"), and size-1 dimensions can cause errors.
- If both sides are single elements without region info, the copy lowers to a plain store `dst[...] = src[...]`.
- Supported directions include global <-> shared, global <-> fragment, shared <-> fragment, and Tensor Memory -> fragment on Blackwell.
- `T.copy` has synchronous semantics: after the statement `dst` is ready. The compiler may lower it to plain loads/stores, `ldmatrix`/`stmatrix`, `cp.async` or TMA bulk copies depending on the target, and inserts the needed waits/barriers. On sm_100 a global -> shared tile copy can lower to TMA; `disable_tma=True` opts out for one copy.
- `eviction_policy`: `"evict_normal"`, `"evict_first"`, `"evict_last"`. `prefer_instruction` (CUDA): `"tma"`, `"cp_async"`, `"sync"`.

### 6.2 Out-of-bounds global accesses

The `LegalizeSafeMemoryAccess` pass guards accesses to **global** buffers whose indices may exceed the declared shape:
- out-of-bounds **loads** return the safe value, which is **zero** for the buffer's dtype;
- out-of-bounds **stores** are skipped.

A partial edge tile copied from global memory therefore contains zeros in its out-of-range part. That is neutral for sums and GEMM, but not for `max`, `min`, products, or anything where 0 is not the identity: overwrite the padding explicitly (section 17.2). Accesses to shared, fragment and local buffers are **not** guarded.

### 6.3 `T.fill` / `T.clear`

```python
T.fill(buffer, value)   # buffer, region or element; e.g. T.fill(m, -T.infinity(T.float32))
T.clear(buffer)         # == T.fill(buffer, 0)
```

### 6.4 Explicitly asynchronous copies

For manual pipelines only: `T.async_copy(src, dst)` issues `cp.async` without a wait (call `T.ptx_wait_group(n)` before reading `dst`; compilation fails if `cp.async` is not possible), and `T.tma_copy(src, dst, barrier=...)` issues a TMA load signalled on an mbarrier from `T.alloc_barrier`.

---

## 7. Loops and Control Flow

### 7.1 `T.Parallel` — element-wise tile loops

```python
T.Parallel(*extents, coalesced_width=None, loop_layout=None, prefer_async=None, annotations=None)
```

```python
for i, j in T.Parallel(BM, BN):
    C_frag[i, j] = T.max(C_frag[i, j], 0)
for i in T.Parallel(BM):
    acc[i] = acc[i] * scale[i]
```

Builds a nest of parallel loops whose iterations are mapped onto the block's threads (vectorized where possible). Iterations must be independent. A lower-rank fragment may be read with a subset of the indices (`row_stat[i]` inside a `(i, j)` loop). `loop_layout=` takes a `T.Fragment` whose input dimension equals the nest depth.

### 7.2 `T.Pipelined` — software-pipelined loops

```python
T.Pipelined(start, stop=None, num_stages=0, order=None, stage=None, sync=None, group=None)
```

```python
for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
    T.copy(A[by * BM, k * BK], A_s)
    T.copy(B[k * BK, bx * BN], B_s)
    T.gemm(A_s, B_s, C_f)
```

- One argument means `range(0, start)`.
- `num_stages` is the number of buffers between producers (copies) and consumers; **`num_stages=0` (the default) disables pipelining**.
- `order` / `stage` give a manual schedule, one entry per scheduled statement (copies, fills, GEMMs, reductions, stores); scalar alias bindings such as `base = k * BK` get no entry. Do not combine manual `order`/`stage` with `num_stages`; the depth is then `max(stage) + 1`.

### 7.3 Sequential loops

```python
T.serial(start, stop=None, step=None, *, annotations=None)      # plain for-loop
T.unroll(start, stop=None, step=None, *, explicit=False, unroll_factor=None, annotations=None)
T.vectorized(start, stop=None, *, annotations=None)
```

- `for i in range(n)` and `range(a, b, s)` inside a kernel map to `T.serial`.
- `T.Serial`, `T.Unroll`, `T.Vectorized` are aliases of the lower-case forms.
- `T.unroll(..., unroll_factor=f)` requires `explicit=False` (the default).

### 7.4 Conditionals and other Python syntax

- `if` / `elif` / `else` with device conditions (`i < N`); conditions on plain Python values are folded at compile time. Ternary `a if c else b` is supported.
- `while cond:` with a device condition; `break` and `continue` inside `T.serial` / `T.unroll` / `T.Parallel` / `while`.
- `T.any_of(c1, c2, ...)` / `T.all_of(c1, c2, ...)` combine predicates.
- `T.ceildiv(a, b)` (alias `T.cdiv`) is ceiling division, used for grid extents and trip counts.
- Not supported inside kernels: iterating over Python lists, `enumerate`, `zip`, chained assignment `a = b = c`, `len()` (use `buf.shape[d]`), `isinstance`, defining functions or classes (use `T.macro`, section 14).
- A plain Python name assigned inside the kernel (`base = k * BK`) is a scalar binding, not storage; use `T.alloc_var` for a mutable per-thread scalar.

---

## 8. Element-wise Math and Expressions

### 8.1 Operators

`+ - * / % //` and comparisons (`< <= > >= == !=`) apply to device expressions; Python numbers are promoted. Bitwise operations: `T.bitwise_and`, `T.bitwise_or`, `T.bitwise_xor`, `T.bitwise_not`, `T.shift_left`, `T.shift_right`.

- Integer `//` and `%` are **floor** division / modulo (`-7 // 3 == -3`, `-7 % 3 == 2`).
- Integer `/` is rejected at compile time ("TVM supports multiple types of integer divisions"); use `//`, `T.floordiv`, `T.floormod`, `T.truncdiv` or `T.truncmod`.
- Float `/` is ordinary division.

### 8.2 Functions

```python
T.exp(x)  T.exp2(x)  T.exp10(x)  T.log(x)  T.log2(x)  T.log10(x)  T.log1p(x)
T.sqrt(x)  T.rsqrt(x)  T.sin(x)  T.cos(x)  T.tan(x)  T.tanh(x)  T.sigmoid(x)  T.erf(x)
T.abs(x)  T.floor(x)  T.ceil(x)  T.round(x)  T.trunc(x)  T.pow(x, y)
T.max(a, b)  T.min(a, b)           # two operands
T.clamp(x, lo, hi)                 # T.min(T.max(x, lo), hi)
T.if_then_else(cond, a, b)         # element-wise select
T.isnan(x)  T.isinf(x)  T.isfinite(x)
```

Value helpers (dtype argument required):

```python
T.infinity(T.float32)       # +inf; -T.infinity(T.float32) for -inf
T.max_value(T.float16)      # largest finite value of a dtype
T.min_value(T.int32)        # smallest value of a dtype
```

Fast-math intrinsics with reduced accuracy: `T.__exp`, `T.__log`, `T.__log2`, `T.__log10`, `T.__exp10`, `T.__sin`, `T.__cos`, `T.__tan`. IEEE helpers with explicit rounding mode (`"rn"`, `"rz"`, `"ru"`, `"rd"`): `T.ieee_add`, `T.ieee_sub`, `T.ieee_mul`, `T.ieee_fmaf`, `T.ieee_fdiv`, `T.ieee_fsqrt`, `T.ieee_frcp`, `T.ieee_frsqrt`.

---

## 9. Matrix Multiply — `T.gemm`

### 9.1 Signature

```python
T.gemm(A, B, C, transpose_A=False, transpose_B=False,
       policy=T.GemmWarpPolicy.Square, clear_accum=False, k_pack=1, mbar=None)
```

Computes `C += op(A) @ op(B)` for one tile (`C = op(A) @ op(B)` when `clear_accum` is true; `clear_accum` may be a device expression such as `k == 0`).

- `A`: `[M, K]`, or `[K, M]` with `transpose_A=True`; `B`: `[K, N]`, or `[N, K]` with `transpose_B=True`; `C`: `[M, N]`. Shapes are checked when the program is built (`T.gemm M/K/N shape check failed`).
- `C` is 2-D; `A`/`B` may have extra leading dimensions of extent 1, and their offset along the second-to-last dimension must be 0.
- `A`, `B`: shared tiles or fragments (any combination); `C`: a whole fragment (or Tensor Memory, section 9.4).
- `policy` (`T.GemmWarpPolicy.Square`, `.FullRow`, `.FullCol`) chooses how the block's warps are split over `M` and `N`.
- `k_pack` is for ROCm only.
- `T.gemm` is synchronous: when it lowers to an asynchronous instruction the matching wait is inserted.

### 9.2 Constraints of the MMA path (fragment accumulator)

From the CUDA MMA lowering (`src/cuda/op/gemm.cc`, `tilelang/cuda/intrinsics/macro/mma_macro_generator.py`):
- `M` must be divisible by 16 and `N` by 8; the warps (`threads / 32`) must be partitionable over the tile (`M >= 16 * m_warps`, `N >= 8 * n_warps`, `m_warps * n_warps == num_warps`), otherwise compilation fails.
- The K chunk of one instruction is `min(256 / bits(A dtype), block_K)`: 16 for fp16/bf16, 8 for fp32, 32 for int8/fp8; `block_K` must be at least that.
- Operand dtype names known to the MMA emitter: `float16`, `bfloat16`, `float32`, `float64`, `int8`, `uint8`, `int4`, fp8 (`e4m3*`, `e5m2*`), fp6, fp4; target support for each still applies.
- **`float32` operands with a `float32` accumulator use TF32 tensor-core MMA** (`kTensorFloat32`), i.e. inputs are rounded to TF32.

### 9.3 Instruction selection on NVIDIA targets

`T.gemm` picks, in order (`src/cuda/op/gemm.cc`):
1. TCGEN5MMA when the target is sm_100–sm_110, `C` is in Tensor Memory (`shared.tmem`), `A` is shared (or TMEM), `B` is shared, and the shape/dtype is supported;
2. WGMMA when the target is Hopper (sm_90–sm_99) and its constraints hold;
3. otherwise warp-level MMA (`mma.sync`).

Consequently on **B200 (sm_100)** `T.gemm` into a **fragment** accumulator lowers to warp-level `mma.sync` (the generated source contains `tl::mma_sync<...>`); WGMMA is never selected on sm_100. `T.wgmma_gemm` (Hopper) and `T.tcgen05_gemm` (Blackwell) force a path and fail compilation instead of falling back.

### 9.4 Blackwell Tensor Memory GEMM (TCGEN5MMA)

TCGEN5MMA writes its accumulator to **Tensor Memory (TMEM)**. The upstream `examples/gemm_sm100/README.md` labels SM100 support a preview.

```python
@tilelang.jit(out_idx=[-1])
def matmul_tmem(M, N, K, block_M=128, block_N=128, block_K=64):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.bfloat16), B: T.Tensor((N, K), T.bfloat16),
             C: T.Tensor((M, N), T.bfloat16)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            C_tmem = T.alloc_tmem([block_M, block_N], T.float32)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, mbar=mbar, clear_accum=k == 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
    return main
```

- `T.alloc_tmem(shape, dtype)`: 2-D shape; TMEM is 128 lanes x 512 columns of 32-bit cells, allocated by columns; the column count must be a power of two, >= 32 and <= 512. It is freed automatically at the end of the allocation block (`T.deallocate_tmem` releases it earlier). Only TCGEN5MMA and TMEM load/store instructions access TMEM, so results are moved to a fragment with `T.copy` before further processing.
- `mbar` (from `T.alloc_barrier(1)`) is **required** when `T.gemm` lowers to TCGEN5MMA; `T.gemm` then inserts the `T.mbarrier_wait_parity` itself.
- The explicit form `T.tcgen05_gemm(A, B, C_tmem, transpose_A, transpose_B, policy, clear_accum, *, mbar, use_2cta=False)` issues without waiting; the program waits with `T.mbarrier_wait_parity(mbar, k % 2)`.
- Supported operand/accumulator combinations (`src/op/tcgen5_meta.h`): fp16/bf16 -> fp32 with `K % 16 == 0`; fp8/fp6/fp4 -> fp32 or fp16 with `K % 32 == 0`; int8/uint8 -> int32 with `K % 32 == 0`; `M` a multiple of 32, 64 or 128 depending on the variant. `float32`/TF32 operands are not in the TCGEN5MMA table.

---

## 10. Reductions and Scans

### 10.1 Tile reductions

```python
T.reduce_sum(buffer, out, dim=-1, clear=True, batch=1)
T.reduce_max(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False)
T.reduce_min(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False)
T.reduce_absmax(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False)
T.reduce_abssum(buffer, out, dim=-1, batch=1)            # always clears
T.reduce_bitand / T.reduce_bitor / T.reduce_bitxor (buffer, out, dim=-1, clear=True, batch=1)
```

- `buffer` and `out` must each be a **fragment or shared** buffer (shared operands are staged through fragments automatically); other scopes raise `ValueError`.
- `out.shape` must equal `buffer.shape` with `dim` removed, or with `dim` set to 1 (e.g. `[BM, BN] -> [BM]` or `[BM, 1]` for `dim=1`).
- `clear=True` initializes `out` first (`0` for sum, `-inf` for max, `+inf` for min); `clear=False` combines with the existing contents of `out`.
- `nan_propagate=True` (fp16/bf16 max/min, CUDA) propagates NaNs; by default NaN inputs are ignored in favour of the other operand.
- `dim` may be negative.

Warp-level reductions of a per-thread scalar (result is available in every lane of the warp): `T.warp_reduce_sum(v)`, `T.warp_reduce_max(v)`, `T.warp_reduce_min(v)`, `T.warp_reduce_bitand(v)`, `T.warp_reduce_bitor(v)`.

### 10.2 Scans

```python
T.cumsum(src, dst=None, dim=0, reverse=False)   # inclusive prefix sum
T.cummax(src, dst=None, dim=0, reverse=False)   # inclusive prefix maximum
```

- `dst=None` scans in place; otherwise `dst` must have the same shape as `src`. `src` may be a buffer or a region slice.
- A fragment `src` is staged through shared memory automatically.

```python
@tilelang.jit(out_idx=[1])
def row_cumsum(M, N):
    @T.prim_func
    def main(X: T.Tensor((M, N), T.float32), Y: T.Tensor((M, N), T.float32)):
        with T.Kernel(M, threads=128) as bm:
            row = T.alloc_shared((N,), T.float32)
            T.copy(X[bm, 0:N], row)
            T.cumsum(row, dim=0)                    # in place
            T.copy(row, Y[bm, 0:N])
    return main
```

---

## 11. Atomics

```python
T.atomic_add(dst, value, memory_order=None, return_prev=False, use_tma=False)
T.atomic_max(dst, value, memory_order=None, return_prev=False)
T.atomic_min(dst, value, memory_order=None, return_prev=False)
T.atomic_addx2(dst, value, return_prev=False)    # 2-wide packed add
T.atomic_addx4(dst, value, return_prev=False)    # 4-wide packed add
T.atomic_load(src, memory_order="seq_cst")
T.atomic_store(dst, src, memory_order="seq_cst")
```

- **Element form**: `dst` is one element (`Y[i]`) and `value` a scalar expression; `return_prev=True` returns the old value; `memory_order` is one of `"relaxed"`, `"consume"`, `"acquire"`, `"release"`, `"acq_rel"`, `"seq_cst"`.
- **Tile form**: `dst`/`value` are buffers or regions (e.g. `T.atomic_add(Out[r0, c0], frag)`): element-wise atomic accumulation of a tile; `return_prev` is not supported there.
- `use_tma=True` (sm90+) performs the tile add with a TMA reduction.
- Floating-point atomic accumulation is order-dependent.

```python
@tilelang.jit
def column_sums(M, N, BM=32, BN=64):
    @T.prim_func
    def main(X: T.Tensor((M, N), T.float32), Y: T.Tensor((N,), T.float32)):   # Y zero-initialized by caller
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            x_frag = T.alloc_fragment((BM, BN), T.float32)
            part = T.alloc_fragment((BN,), T.float32)
            T.copy(X[by * BM, bx * BN], x_frag)
            T.reduce_sum(x_frag, part, dim=0)
            for j in T.Parallel(BN):
                T.atomic_add(Y[bx * BN + j], part[j])
    return main
```

---

## 12. Layout and Scheduling Annotations

Written in the kernel body (`T.annotate_restrict_buffers` before `T.Kernel`, the others inside it).

```python
T.use_swizzle(panel_size, order="row", enable=True)   # threadblock rasterization (block-index remapping)
T.annotate_layout({A_shared: tilelang.layout.make_swizzled_layout(A_shared)})
T.annotate_min_blocks_per_sm(n)        # second argument of __launch_bounds__
T.annotate_restrict_buffers(x, y)      # omit __restrict__ for parameters that may alias
```

- `T.use_swizzle` remaps block indices in panels of `panel_size` (`order="row"` or `"column"`); `enable=False` makes it a no-op.
- `T.annotate_layout` maps buffers to `T.Layout` / `T.Fragment` objects (a fragment buffer requires a `T.Fragment`) or to an index-mapping callable. Layouts not annotated are inferred.
- Compiler-wide switches go in `pass_configs` (`tilelang.PassConfigKey`), e.g. `TL_DISABLE_WARP_SPECIALIZED`, `TL_ENABLE_FAST_MATH`, `TL_DISABLE_SAFE_MEMORY_ACCESS` (disables the out-of-bounds guards of section 6.2). `TL_DISABLE_TMA_LOWER` is marked deprecated in 0.1.11.

---

## 13. Synchronization and Thread-level Helpers

Tile operators and `T.Pipelined` insert the barriers they need. For hand-written thread-level code:

```python
T.sync_threads()                 # __syncthreads()
T.sync_warp()                    # __syncwarp()
T.shfl_xor(value, delta)         # also T.shfl_down, T.shfl_up, T.shfl_sync
T.get_lane_idx()  T.get_warp_idx()
```

---

## 14. Macros — `T.macro`

Inside kernels Python functions cannot be defined; `@T.macro` defines a reusable block that is inlined where it is called (like a `__device__` function). It can return an expression or emit statements.

```python
@T.macro
def square(x):
    return x * x

@tilelang.jit(out_idx=[1])
def squares(N, block=128):
    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                Y[bx * block + i] = square(X[bx * block + i])
    return main
```

---

## 15. Debugging and Inspection

```python
T.print(obj=None, msg="", warp_group_id=0, warp_id=0)   # inside a kernel
```

- A shared or fragment buffer is printed by a single thread; a scalar expression is printed by every thread that executes the statement (guard it with `if`). Output lines carry `BlockIdx` / `ThreadIdx`.
- `T.device_assert(cond, msg)` is a device-side assertion (CUDA).

Host side:

```python
kernel.get_kernel_source()      # generated CUDA C++ of a JITKernel
kernel.get_host_source()        # host stub with argument checks
jit_fn.get_kernel_source(*args) # compile (if needed) and return the source
```

`@tilelang.jit(debug_root_path="dir")` writes the CUDA source and the TIR script at each compilation.

Environment: `TILELANG_CACHE_DIR` (on-disk kernel cache, default `~/.tilelang/cache`), `TILELANG_DISABLE_CACHE=1`, `TILELANG_PRINT_ON_COMPILATION` (default `1`: logs each compilation), `TILELANG_TARGET` / `TILELANG_EXECUTION_BACKEND` / `TILELANG_VERBOSE` (defaults for the `jit` options).

---

## 16. Autotuning

`tilelang.autotune` and `tilelang.autotuner.AutoTuner` exist in 0.1.11. The benchmark's generated files must use fixed literal configurations (no autotuning or runtime configuration search).

---

## 17. Generic Examples (API illustration only)

The three programs below were compiled and run with TileLang 0.1.11 on an sm_100 GPU and compared with PyTorch.

### 17.1 Vector add (lazy style, `out_idx`)

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def vector_add(N, block=256, dtype=T.float32):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                C[bx * block + i] = A[bx * block + i] + B[bx * block + i]
    return main

a = torch.randn(1000, device="cuda")
b = torch.randn(1000, device="cuda")
kernel = vector_add(1000)          # JITKernel; N = 1000 is not a multiple of 256 (tail stores are skipped)
c = kernel(a, b)                   # C allocated by the call and returned
```

### 17.2 Row-wise maximum with a partial edge tile (dynamic `M`, output passed in)

```python
@tilelang.jit
def row_max(N, block_M=4, block_N=128, dtype=T.float32):
    M = T.dynamic("M")

    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M,), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bm:
            x_frag = T.alloc_fragment((block_M, block_N), dtype)
            m_part = T.alloc_fragment((block_M,), dtype)
            m_acc = T.alloc_fragment((block_M,), dtype)
            T.fill(m_acc, -T.infinity(dtype))
            for kn in T.serial(T.ceildiv(N, block_N)):
                T.copy(X[bm * block_M, kn * block_N], x_frag)     # out-of-range columns read as 0
                for i, j in T.Parallel(block_M, block_N):
                    x_frag[i, j] = T.if_then_else(kn * block_N + j < N, x_frag[i, j], -T.infinity(dtype))
                T.reduce_max(x_frag, m_part, dim=1)
                for i in T.Parallel(block_M):
                    m_acc[i] = T.max(m_acc[i], m_part[i])
            T.copy(m_acc, Y[bm * block_M])
    return main

x = torch.randn(37, 1000, device="cuda")
y = torch.empty(37, device="cuda")
row_max(1000)(x, y)                # no out_idx: the kernel writes into y
```

### 17.3 Tiled GEMM (eager style)

```python
@tilelang.jit
def matmul(A, B, block_M=128, block_N=128, block_K=32, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C

a = torch.randn(256, 192, device="cuda", dtype=torch.float16)
b = torch.randn(192, 320, device="cuda", dtype=torch.float16)
c = matmul(a, b)                   # compiles for (M, N, K) = (256, 320, 192), runs, returns C
```

---

## 18. Constraints and Pitfalls

### 18.1 Kernel-scope statements run on every thread
Only tile operators and `T.Parallel` loops are distributed. A statement such as `Y[0] = v` or `T.atomic_add(cnt[0], 1)` at kernel scope executes once per thread; guard thread-level work with `if T.get_thread_binding() == 0:` when it must run once.

### 18.2 Partial tiles are zero-padded
Out-of-range global loads return 0 and out-of-range global stores are dropped (section 6.2). Replace padding with the reduction identity (`-T.infinity(dtype)` for max, `T.infinity(dtype)` for min, 1 for products) before reducing, or bound loops explicitly. Shared/fragment indexing is not guarded.

### 18.3 `T.Pipelined` without `num_stages`
`T.Pipelined(n)` with the default `num_stages=0` is not pipelined. Do not combine `num_stages` with manual `order`/`stage`.

### 18.4 `T.const` specializes, `T.dynamic` does not
`T.const` (eager) and Python-int sizes (lazy) compile a new kernel for every new size; `T.dynamic` compiles once.

### 18.5 `out_idx` and argument counts
Parameters in `out_idx` are not passed by the caller (a wrong count raises `ValueError`); they are uninitialized (`torch.empty`), so every element must be written. Eager style uses `T.empty` instead.

### 18.6 Contiguity and dtype are checked
`T.Tensor` arguments must be contiguous and exactly of the compiled dtype; call `.contiguous()` / `.to(dtype)` in the Python wrapper, or declare `T.StridedTensor` (sections 3.4, 4.5).

### 18.7 GEMM shapes and precision
MMA path: `M % 16 == 0`, `N % 8 == 0`, `block_K` at least the instruction K, warps must fit the tile; fp32 operands run as TF32. On sm_100 a fragment accumulator gives `mma.sync`; TCGEN5MMA needs a TMEM accumulator and `mbar` (sections 9.2-9.4).

### 18.8 Reduction shapes and scopes
Reduction operands must be fragment or shared buffers; `out` has the input shape with `dim` removed or set to 1; `clear=False` accumulates into `out` across iterations.

### 18.9 Integer division
`/` on integers is a compile error; `//` and `%` are floor semantics; `T.truncdiv` / `T.truncmod` give C semantics.

### 18.10 `T.where` is not a select
`T.where` is a block predicate statement; element-wise selection is `T.if_then_else(cond, a, b)`.

### 18.11 Deprecated spellings
`T.symbolic` -> `T.dynamic`; `T.Buffer(...)` -> `T.Tensor(...)`.

### 18.12 No postponed annotations in kernel modules
With `from __future__ import annotations` in the defining module, a `@T.prim_func` whose parameters are annotated `T.Tensor(...)` fails to build (observed with Python 3.10: `TypeError: Forward references must evaluate to types. Got buffer.`). Do not use that import in files that define kernels.

### 18.13 Common failure messages

| Message | Cause |
|---------|-------|
| `Kernel expected N inputs, but M are provided.` | wrong number of call arguments (outputs in `out_idx` must be omitted) |
| `... dtype mismatch, expected float32` / `strides[...] violates packed ABI constraint` | tensor dtype differs / tensor not contiguous |
| `T.gemm M shape check failed` (or K / N) | operand tile shapes inconsistent with `transpose_A` / `transpose_B` |
| `M must be divisible by 16` / `N must be divisible by 8` / `m_warp * n_warp must equal num_warps` | GEMM tile too small or misaligned for the block's warps |
| `block_K (...) must be >= micro_size_k (...)` | `block_K` below the MMA instruction K |
| `Invalid reduce output shape` | reduction output shape does not match `dim` |
| `TVM supports multiple types of integer divisions` | integer `/` |

---

## 19. Names That Do Not Exist in TileLang 0.1.11

Verified absent in the installed 0.1.11 package (some exist in other DSLs or in upstream documentation of other versions):

| Absent name | Use instead |
|-------------|-------------|
| `T.dyn` (spelling used in the upstream language-basics guide) | `T.dynamic` |
| `T.reduce_prod`, `T.cumprod` | - |
| `T.argmax`, `T.reduce_argmax` | - |
| `T.atomic_sub`, `T.atomic_and`, `T.atomic_or` | `T.atomic_add` with a negated value (for subtraction) |
| `T.sort`, `T.topk` | - |
| `T.zeros`, `T.arange` | `T.alloc_fragment` + `T.fill` / `T.clear`; `T.Parallel` loop indices |
| `T.program_id` | the `T.Kernel(...) as bx` variables, `T.get_block_binding` |
| `T.load`, `T.store` | indexing (`A[i]`, `A[i] = v`) and `T.copy` |
| `tilelang.AutoTuner` | the class is `tilelang.autotuner.AutoTuner` |

---

## 20. License Notice

Parts of this reference (API descriptions and the tiled-GEMM / TCGEN5MMA examples) are adapted from TileLang v0.1.11 (https://github.com/tile-ai/tilelang), distributed under the MIT License:

MIT License

Copyright (c) Tile-AI.
**During the period from December 1, 2024, to Mar 14, 2025, this project is
subject to additional collaboration terms with Microsoft Corporation.**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
