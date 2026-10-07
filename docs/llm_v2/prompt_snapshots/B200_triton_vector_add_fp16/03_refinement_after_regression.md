# API reference: triton 3.6.0

# Triton 3.6.0 Language Reference

Triton is a Python DSL that compiles through MLIR/LLVM to GPU kernels. Every kernel is an SPMD program: a **grid** of independent **programs** that operate on **block tensors** whose shapes are compile-time constants. This document covers the API surface of Triton **3.6.0** as installed (`triton.__version__ == "3.6.0"`); every symbol named below was confirmed by introspection of that package. All snippets are valid inside a `@triton.jit` function unless marked *host*.

Scope: syntax, programming model, semantics, dtype rules, memory/synchronisation/launch semantics, legal shapes and capability-dependent constraints. It is not a tutorial and not a device datasheet: SM counts, cache sizes, bandwidths and shared-memory budgets are in the Device Context document, not here. `@triton.autotune`, `triton.Config` sweeps and `triton.testing.do_bench` exist in 3.6.0 but are **not permitted in this task setting** and are not documented here.

---

## 1. Imports and Namespaces

```python
import torch
import triton
import triton.language as tl
```

Optional, host side only:

```python
from triton.tools.tensor_descriptor import TensorDescriptor   # host-built TMA descriptor (section 4.5)
from triton.runtime import driver                             # device queries: driver.active.get_device_capability()
```

| Namespace | Purpose |
|-----------|---------|
| `triton` | Host API: `jit`, `heuristics`, `cdiv`, `next_power_of_2`, `set_allocator`, `compile`, `constexpr_function`; errors `CompilationError`, `OutOfResources`, `TritonError` |
| `triton.language` (`tl`) | Every in-kernel primitive: `load`, `store`, `dot`, `arange`, `program_id`, `sum`, ... |
| `triton.language.math` (`tl.math`) | Math functions; every `tl.math.f` is also exported as `tl.f` |
| `triton.language.extra.libdevice` | CUDA libdevice bindings (`from triton.language.extra import libdevice`) |
| `triton.runtime` | `driver`, `JITFunction` |

---

## 2. Data Types

### 2.1 Dtype Constants (all on `tl`)

| Integer | Float | FP8 |
|---------|-------|-----|
| `tl.int1` (bool) | `tl.float16` | `tl.float8e4nv` (E4M3, NVIDIA OCP) |
| `tl.int8` / `tl.uint8` | `tl.bfloat16` | `tl.float8e5` (E5M2) |
| `tl.int16` / `tl.uint16` | `tl.float32` | `tl.float8e4b15` (E4M3 with bias 15, legacy) |
| `tl.int32` / `tl.uint32` | `tl.float64` | `tl.float8e4b8`, `tl.float8e5b16` (AMD gfx942 formats) |
| `tl.int64` / `tl.uint64` | | |

Capability-dependent availability on NVIDIA (`backends/nvidia/compiler.py`):
- `float8e5` and `float8e4b15` are always accepted by the compiler; `float8e4nv` is added to the supported set only when the target capability is >= 8.9.
- On capability >= 9.0, `float8e4b15` is a *deprecated* `tl.dot` operand: it is up-cast to `float16` with a warning.
- `float8e4b8` / `float8e5b16` are gfx942 types; on NVIDIA they are up-cast to `float16` with a warning.

Dtype predicates (usable at compile time, e.g. on `x.dtype`):

```python
tl.float16.primitive_bitwidth   # 16
tl.int32.is_int_signed()        # True
tl.float16.is_floating()        # True
tl.float8e5.is_fp8()            # True
# also: is_int(), is_int_unsigned(), is_fp16(), is_bf16(), is_fp32(), is_fp64(), is_bool(), is_ptr()
```

### 2.2 Dtype of a Pointer or Tensor

```python
x_ptr.dtype.element_ty          # element dtype behind a pointer argument
x.dtype                         # dtype of a loaded block tensor
x.type.scalar                   # scalar dtype of a block
```

### 2.3 Casts

```python
y = x.to(tl.float32)                                  # numeric cast
y = x.to(tl.int32, bitcast=True)                      # reinterpret bits (same bitwidth)
y = x.to(tl.bfloat16, fp_downcast_rounding="rtne")    # "rtne" | "rtz"; only for float down-casts
y = tl.cast(x, tl.float16)                            # functional form, same keywords
```

`tl.store` casts `value` to `pointer.dtype.element_ty` implicitly; `tl.load(..., other=v)` casts `other` to the pointer's element type.

### 2.4 Integer-valued fp16

`float16` represents integers exactly only up to 2048. Index-like values must not be carried in fp16 (`int32` or `float32` instead).

---

## 3. The Kernel Programming Model

### 3.1 One Program, One Block of Work

A kernel launched on grid `(G0, G1, G2)` runs `G0 * G1 * G2` programs. Each program (1) reads its coordinates with `tl.program_id(axis)`, (2) builds pointer offsets, (3) `tl.load`s blocks, (4) computes, (5) `tl.store`s. Programs cannot communicate except through global memory (atomics, section 5.4).

### 3.2 Block Tensors

- **Shape**: a tuple of compile-time integers. Block pointers (`make_block_ptr`) and tensor descriptors carry a constexpr `block_shape` as well.
- **Dtype**: one scalar dtype per tensor.
- **Layout**: chosen by the compiler; not visible to the program.
- Tensors are immutable values: `x[0] = 1` is not supported; every operation returns a new tensor (section 11.2).

`tl.arange` extents must be powers of two (section 6), so block dimensions are in practice powers of two; non-power-of-two problem sizes are handled with masks (section 4.3) or boundary checks (section 4.4).

### 3.3 Program Id and Grid Size

```python
pid = tl.program_id(axis)       # int32 scalar, axis in {0, 1, 2}
n   = tl.num_programs(axis)     # grid extent along axis
```

### 3.4 Launch Syntax (host)

```python
kernel[grid](arg0, arg1, ..., META=value, num_warps=4, num_stages=3)
```

- `grid` is a `tuple[int, ...]` (1 to 3 entries) or a callable `Callable[[dict], tuple[int, ...]]` that receives the dict of launch keyword arguments and returns the tuple.
- Every parameter annotated `: tl.constexpr` must be passed as a keyword argument.
- Compile options accepted as launch keywords (NVIDIA backend `CUDAOptions`): `num_warps` (default 4, must be a power of two), `num_stages` (default 3), `num_ctas` (default 1; `> 1` requires capability >= 9.0 and raises otherwise), `maxnreg` (PTX `.maxnreg` register cap per thread, default `None`), `enable_fp_fusion` (default `True`).
- `kernel.warmup(*args, grid=..., **kwargs)` compiles without executing.

```python
# host
grid = (triton.cdiv(n, 1024),)
kernel[grid](x, y, n, BLOCK=1024, num_warps=4)
```

### 3.5 Kernel Parameter Rules

| Annotation | Meaning | Specialisation |
|------------|---------|----------------|
| none | runtime scalar or pointer | specialised on 16-byte divisibility (`tt.divisibility=16`) of pointers and integers |
| `: tl.constexpr` | compile-time constant | new value = new compilation |
| `: tl.const` | read-only pointer (`const T*`) | as a pointer |
| `arg=default` | Python default | applied when omitted |

`tl.constexpr` is required for every value that feeds a shape: `tl.arange` bounds, `tl.zeros`/`tl.full` shapes, `block_shape` of block pointers and tensor descriptors, `tl.dot` operand shapes, and for booleans that select code paths (`if CAUSAL:` is resolved at compile time).

`@triton.jit(do_not_specialize=[...], do_not_specialize_on_alignment=[...])` take parameter names or indices and disable the respective specialisation for those arguments (section 14).

### 3.6 `tl.assume`

```python
tl.assume(pid_m >= 0)
tl.assume(stride_am > 0)
```

Lets the compiler assume `cond` is true (integer range analysis for address arithmetic). An assumption that is false at run time is undefined behaviour.

---

## 4. Pointer and Offset Arithmetic

### 4.1 Tensor-of-Pointers Model

A `torch.Tensor` kernel argument arrives as a base pointer. Adding an integer tensor of offsets produces a tensor of pointers of the same shape, which `tl.load`/`tl.store` consume.

```python
# 1D block of BLOCK contiguous elements
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
```

```python
# 2D block addressed through strides
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)             # [BLOCK_M]
offs_k = tl.arange(0, BLOCK_K)                                # [BLOCK_K]
ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
a = tl.load(ptrs, mask=mask, other=0.0)
```

Offsets are `int32` unless an operand is `int64`; products of large sizes and strides must be computed in `int64` (cast an operand with `.to(tl.int64)`) when they can exceed 2^31.

### 4.2 Broadcasting

`x[:, None]` and `x[None, :]` insert size-1 axes (`tl.expand_dims`). Binary operations broadcast NumPy-style. `tl.broadcast_to(x, shape)` and `tl.broadcast(x, y)` are the explicit forms.

### 4.3 Masking

`mask` is an `int1` tensor broadcast to the pointer shape. Masked-off loads return `other` (undefined when `other` is omitted); masked-off stores are skipped.

Identity values for `other`: `0.0` for sums, `-float('inf')` for max, `float('inf')` for min, `1.0` for products.

### 4.4 Block Pointers — `tl.make_block_ptr`

```python
A = tl.make_block_ptr(base=a_ptr,
                      shape=(M, K),                      # parent shape (runtime ints)
                      strides=(stride_am, stride_ak),
                      offsets=(pid_m * BLOCK_M, 0),      # element offsets of this block
                      block_shape=(BLOCK_M, BLOCK_K),    # constexpr
                      order=(1, 0))                      # memory order of the dims, most contiguous first
a = tl.load(A, boundary_check=(0, 1), padding_option="zero")
A = tl.advance(A, (0, BLOCK_K))                          # returns a NEW block pointer
```

- `order` lists dimensions by memory contiguity, most contiguous first: `(1, 0)` for a row-major 2D block (axis 1 is contiguous).
- `boundary_check` names the axes to bounds-check against `shape`; `padding_option` is `""` (undefined), `"zero"` or `"nan"`.
- `tl.advance` does not modify its argument; the result must be re-assigned (the function is marked `must_use_result`, so a discarded result produces a warning).
- With a block pointer, `mask=`/`other=` must be `None`; with a tensor of pointers, `boundary_check`/`padding_option` must be empty (section 5.1).

### 4.5 Tensor Descriptors (TMA)

A tensor descriptor describes an N-D array and a constexpr block shape; `desc.load(offsets)` / `desc.store(offsets, value)` move whole blocks. On NVIDIA GPUs with TMA hardware (capability 9.0 and later) these lower to TMA; the API is accepted on other targets but is not hardware TMA there.

In-kernel construction:

```python
desc = tl.make_tensor_descriptor(base=x_ptr,
                                 shape=[M, N],                 # runtime ints
                                 strides=[N, 1],               # last stride must be 1
                                 block_shape=[BLOCK_M, BLOCK_N],   # constexpr
                                 padding_option="zero")        # "zero" | "nan"
tile = desc.load([m_off, n_off])                              # [BLOCK_M, BLOCK_N]
desc.store([m_off, n_off], tile)
```

In-kernel descriptors need a global-memory workspace; register an allocator on the host once:

```python
# host
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)
```

Host-side construction (no allocator needed; pass the object as a kernel argument and call `.load`/`.store` on it in the kernel):

```python
# host
from triton.tools.tensor_descriptor import TensorDescriptor
desc = TensorDescriptor(base=x, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N], padding="zero")
desc = TensorDescriptor.from_tensor(x, block_shape=[BLOCK_M, BLOCK_N])   # strides taken from x
```

Constraints checked by the compiler (`language/semantic.py`):
- base pointer 16-byte aligned; leading strides multiples of 16 bytes; last stride exactly 1;
- 1 to 5 dimensions (the docstring states 2 to 5 as the supported range);
- `block_shape[-1] * element_size >= 16` bytes;
- `padding_option="nan"` is rejected for integer element types;
- descriptor methods: `load`, `store`, `atomic_add` (element dtype in {int32, uint32, uint64, float32, float16, bfloat16}), `atomic_and/or/xor/max/min` (int32, uint32, int64, uint64 only), and `gather(x_offsets, y_offset)` / `scatter(value, x_offsets, y_offset)` which require a 2D descriptor with `block_shape[0] == 1`, a 1D `x_offsets` of at least 8 rows, element dtype in {int32, uint32, int64, uint64, float16, bfloat16}, and native TMA for the 16-bit float types.

---

## 5. Memory Operations

### 5.1 `tl.load`

```python
tl.load(pointer, mask=None, other=None, boundary_check=(), padding_option="",
        cache_modifier="", eviction_policy="", volatile=False)
```

| `pointer` is | `mask` / `other` | `boundary_check` / `padding_option` |
|--------------|------------------|-------------------------------------|
| scalar pointer | scalars | must be empty |
| tensor of pointers | broadcast to `pointer.shape` | must be empty |
| block pointer | must be `None` | control out-of-bounds behaviour |

- `cache_modifier` (NVIDIA): `""`, `".ca"` (cache all levels), `".cg"` (cache at L2, bypass L1), `".cv"` (do not cache, re-fetch).
- `eviction_policy`: `""`, `"evict_first"`, `"evict_last"`.
- `padding_option`: `""` (undefined out of bounds), `"zero"`, `"nan"`.
- `volatile=True` forbids the compiler from caching the loaded value across loads.

### 5.2 `tl.store`

```python
tl.store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="")
```

- `value` is broadcast to `pointer.shape` and cast to `pointer.dtype.element_ty`.
- `cache_modifier` (NVIDIA): `""`, `".wb"` (write-back), `".cg"`, `".cs"` (streaming), `".wt"` (write-through).
- Masked-off elements are not written.

### 5.3 `tl.gather` and `tl.histogram`

```python
out  = tl.gather(src, index, axis)          # out[..., i, ...] = src[..., index[..., i, ...], ...] along axis
hist = tl.histogram(input, num_bins, mask=None)   # bins of width 1 starting at 0; integer input
```

`gather` indexes one axis with an index tensor of the same rank as `src`; `num_bins` is a constexpr.

### 5.4 Atomics

All return the value stored at `pointer` *before* the operation.

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

- `sem`: `"acquire"`, `"release"`, `"acq_rel"` (default), `"relaxed"`.
- `scope`: `"gpu"` (default), `"cta"`, `"sys"`.
- Dtype rules: `atomic_cas` needs 16-, 32- or 64-bit elements; the RMW atomics reject `int16`/`uint16` and anything narrower than 16 bits; `float16` and `bfloat16` are accepted by `atomic_add` only (every other RMW rejects them); `atomic_max`/`atomic_min` on floating types are implemented by the compiler, on `int1` they are rejected. The pointer may not be `tl.const`.
- Floating-point atomic accumulation is order-dependent, so results are not bit-reproducible across runs.

Spin lock on an `int32` flag (generic API usage):

```python
while tl.atomic_cas(lock_ptr, 0, 1) == 1:
    pass
# ... critical section ...
tl.debug_barrier()            # order this program's writes before the release
tl.atomic_xchg(lock_ptr, 0)
```

---

## 6. Tile Creation

```python
tl.arange(start, end)          # int32 [start, end); both constexpr; end - start a power of two, <= 1048576
tl.zeros(shape, dtype)         # shape: tuple of constexpr
tl.zeros_like(x)
tl.full(shape, value, dtype)
```

```python
offs = tl.arange(0, BLOCK)
acc  = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
m_i  = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
```

---

## 7. Shape and View Operations

```python
tl.reshape(x, *shape, can_reorder=False)   # shape as tuple or separate ints; can_reorder permits element reordering
tl.view(x, *shape)                         # reshape that may reorder elements
tl.ravel(x, can_reorder=False)             # flatten to 1D
tl.expand_dims(x, axis)                    # insert size-1 axis (also x[:, None])
tl.broadcast_to(x, *shape)
tl.broadcast(x, y)                         # broadcast both to a common shape
tl.trans(x, *dims)                         # permute; x.T for 2D
tl.permute(x, *dims)
tl.cat(a, b, can_reorder=False)            # concatenate along axis 0
tl.join(a, b)                              # new innermost axis of size 2: (4, 8) x 2 -> (4, 8, 2)
tl.split(x)                                # inverse of join; last dim must be 2 -> (a, b)
tl.interleave(a, b)                        # join + reshape: zip along the last dim
tl.flip(x, dim=None)                       # reverse along dim
```

Method forms exist for most: `x.reshape(...)`, `x.trans(...)`, `x.permute(...)`, `x.broadcast_to(...)`, `x.expand_dims(...)`, `x.view(...)`, `x.ravel()`, `x.split()`, `x.flip(...)`.

Standard-library helpers (JIT functions in `triton.language.standard`, exported on `tl`): `tl.sort(x, dim=None, descending=False)`, `tl.topk(x, k, dim=None)` (`k` constexpr), `tl.softmax(x, dim=None, keep_dims=False, ieee_rounding=False)`, `tl.sigmoid(x)`, `tl.swizzle2d(i, j, size_i, size_j, size_g)` (row-major index to grouped column-major index), `tl.cdiv(x, div)` (in-kernel ceil division).

---

## 8. Arithmetic and Math

### 8.1 Operators

```python
a + b   a - b   a * b   a / b      # true division gives a float
a // b  a % b   a ** b
a & b   a | b   a ^ b   ~a   -a   a << b   a >> b
a > b   a >= b  a < b   a <= b  a == b  a != b      # int1 results
x.logical_and(y)   x.logical_or(y)                  # int1 operands
```

Python scalars promote: `x * 2`, `x + 0.5`. Integer division by zero and overflow are not checked unless `TRITON_DEBUG=1`/`debug=True` (section 14).

### 8.2 Explicit Element-wise Functions

```python
tl.maximum(x, y, propagate_nan=tl.PropagateNan.NONE)   # NONE: NaN-tolerant; ALL: NaN if either is NaN
tl.minimum(x, y, propagate_nan=tl.PropagateNan.NONE)
tl.clamp(x, min, max, propagate_nan=tl.PropagateNan.NONE)
tl.where(cond, x, y)                                     # element-wise select
tl.add(x, y)  tl.sub(x, y)  tl.mul(x, y)                 # functional forms of the operators
```

`tl.maximum`, `tl.minimum` and `tl.clamp` up-cast `bfloat16` operands to `float32` internally (the hardware has no bf16 min/max/compare); the result is `float32`.

### 8.3 Math Functions (`tl.math.*`, re-exported as `tl.*`)

```python
tl.exp(x)   tl.exp2(x)   tl.log(x)   tl.log2(x)   tl.cos(x)   tl.sin(x)
tl.sqrt(x)  tl.rsqrt(x)  tl.erf(x)   tl.floor(x)  tl.ceil(x)
tl.fma(x, y, z)                      # fused multiply-add
tl.fdiv(x, y, ieee_rounding=False)   # fast division
tl.umulhi(x, y)                      # high half of the full-width product; int32/int64/uint32/uint64 only
tl.sqrt_rn(x)                        # IEEE round-to-nearest sqrt;   float32 only
tl.div_rn(x, y)                      # IEEE round-to-nearest divide; float32 only
tl.abs(x)
```

Dtype rule (enforced by `_check_dtype` in `language/math.py`): `exp`, `exp2`, `log`, `log2`, `cos`, `sin`, `sqrt`, `rsqrt`, `erf`, `floor`, `ceil` accept **`float32` or `float64` only**. Cast `float16`/`bfloat16` inputs with `.to(tl.float32)` first. `sqrt`, `rsqrt`, `exp`, `log` are the fast (approximate) variants; `sqrt_rn`/`div_rn` are the correctly rounded ones.

`tl.exp(x)` equals `tl.exp2(x * 1.4426950408889634)` (log2 e); either form is valid, both require fp32.

Further CUDA math is available through `libdevice` (`from triton.language.extra import libdevice`; e.g. `libdevice.tanh`, `libdevice.pow`, `libdevice.atan2`); its functions follow libdevice dtype rules.

---

## 9. Matrix Multiply — `tl.dot`

```python
c = tl.dot(a, b, acc=None,
           input_precision=None,       # fp32 operands only: "tf32" (default) | "tf32x3" | "ieee" | "bf16x3" | "bf16x6"
           allow_tf32=None,            # deprecated alias of input_precision
           max_num_imprecise_acc=None, # fp8 -> fp32 on capability 9.0 only
           out_dtype=tl.float32)       # tl.float32 | tl.float16
```

Shape rules (`semantic.dot`):
- `a` is `[M, K]` or `[B, M, K]`, `b` is `[K, N]` or `[B, K, N]`; both 2D or both 3D (3D = batched); result `[M, N]` / `[B, M, N]`.
- Minimum sizes on NVIDIA: `M >= 1`, `N >= 1`, `K >= 16` for 16- and 32-bit operands, `K >= 32` for 8-bit operands (int8, fp8).
- If `acc` is given it must have exactly the result shape and dtype `out_dtype`; then `c = a @ b + acc`.

Dtype rules:
- Both operands must have the **same dtype**, except that any pair of fp8 formats is accepted.
- Accepted operand dtypes: `int8`, `float16`, `bfloat16`, `float32`, `float64`, and the fp8 types (section 2.1 for capability restrictions). `uint8` passes the first check but is then rejected ("only int8 supported").
- `int8 x int8` accumulates in `int32` (result `int32`).
- `float32` and `bfloat16` operands always produce `float32`.
- `float16` and fp8 operands produce `out_dtype` (`float32` default, `float16` allowed).
- `out_dtype=tl.bfloat16` is rejected; cast the fp32 result with `.to(tl.bfloat16)`.
- fp32 operands use TF32 tensor-core math unless `input_precision="ieee"`; `TRITON_F32_DEFAULT=ieee` changes the default.

Accumulator chaining across a reduction loop (generic illustration):

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in tl.range(0, num_k_blocks):
    a = ...                      # [BLOCK_M, BLOCK_K] block for step k (fp16/bf16/fp32/int8/fp8)
    b = ...                      # [BLOCK_K, BLOCK_N] block for step k, same dtype as a
    acc = tl.dot(a, b, acc)      # acc keeps the result dtype (fp32 here) across iterations
```

### 9.1 `tl.dot_scaled` — Microscaled Operands

```python
tl.dot_scaled(lhs, lhs_scale, lhs_format,    # formats: "e2m1" | "e4m3" | "e5m2" | "bf16" | "fp16"
              rhs, rhs_scale, rhs_format,
              acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True,
              out_dtype=tl.float32)
```

- `lhs` is `[M, K]`; `rhs` is `[K, N]`; `e2m1` (fp4) elements are packed two per `uint8` with the first element in the low bits; fp8 elements may be `uint8` or the fp8 dtype.
- Scales are `e8m0` stored as `uint8`, shape `[M, K // 32]` for `lhs` and `[N, K // 32]` for `rhs` (do not transpose `rhs_scale`); `None` means unscaled.
- Hardware without native microscaling support runs a software emulation (operands up-cast to `bf16`).

---

## 10. Reductions and Scans

```python
tl.sum(x, axis=None, keep_dims=False, dtype=None)
tl.max(x, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
tl.min(x, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
tl.argmax(x, axis, tie_break_left=True, keep_dims=False)
tl.argmin(x, axis, tie_break_left=True, keep_dims=False)
tl.xor_sum(x, axis=None, keep_dims=False)        # integer xor
tl.reduce_or(x, axis, keep_dims=False)           # integer or
```

- `axis=None` reduces every dimension to a scalar; `keep_dims=True` keeps a size-1 axis for broadcasting back.
- `tl.sum`: integers narrower than 32 bits are accumulated in `int32`/`uint32`; `dtype=` forces the accumulation dtype; **`bfloat16` is not promoted** by `sum` (pass `dtype=tl.float32` or cast first).
- `tl.max`, `tl.min`, `tl.cumsum`, `tl.cumprod` up-cast `bfloat16` input to `float32` before reducing; the result dtype is then `float32`.
- `return_indices=True` on `max`/`min` returns `(values, int32 indices)`.

### 10.1 Custom Reductions — `tl.reduce`

```python
@triton.jit
def _combine(a, b):
    return a + b

y = tl.reduce(x, axis, _combine, keep_dims=False)

# tuple input: combine_fn takes 2 x len(tuple) scalars and returns len(tuple) values
@triton.jit
def _argmin_combine(v1, i1, v2, i2):
    lt = v1 < v2
    return tl.where(lt, v1, v2), tl.where(lt, i1, i2)

min_val, min_idx = tl.reduce((vals, idxs), 0, _argmin_combine)
```

`combine_fn` must be a `@triton.jit` function and must be associative.

### 10.2 Scans

```python
tl.cumsum(x, axis=0, reverse=False, dtype=None)
tl.cumprod(x, axis=0, reverse=False)
tl.associative_scan(x, axis, combine_fn, reverse=False)   # custom; combine_fn as in tl.reduce
```

---

## 11. Control Flow

### 11.1 Compile-time Specialisation

`if`/`for` whose condition or bounds are `constexpr` are resolved at compile time:

```python
if CAUSAL:                         # CAUSAL: tl.constexpr -> dead branch removed
    ...
for i in tl.static_range(0, N):    # N: tl.constexpr -> fully unrolled; bounds must be constexpr
    ...
```

### 11.2 Runtime Loops — `tl.range`

```python
for k in tl.range(start, end, step,
                  num_stages=None,              # software-pipeline depth for loads in this loop body
                  loop_unroll_factor=None,      # IR-level unroll; values < 2 mean none
                  disallow_acc_multi_buffer=False,
                  flatten=False,                # flatten this loop nest into one loop
                  warp_specialize=False,        # Blackwell only; simple matmul loops only
                  disable_licm=False):          # do not hoist loop-invariant code
    ...
```

Plain Python `for k in range(...)` over runtime bounds is also compiled as a loop; `tl.range` is the form that accepts attributes. Loop-carried values must keep the same type and shape across iterations.

`num_stages` here pipelines most loads in the loop body; the kernel-level `num_stages` launch option (section 3.4) only pipelines loads that feed `tl.dot`.

Grid-stride loop (generic form for a program that handles several work items):

```python
for item in tl.range(tl.program_id(0), n_items, tl.num_programs(0)):
    ...
```

### 11.3 `while` Loops

```python
while tl.condition(cond, disable_licm=False):   # or plain `while cond:`
    ...
```

### 11.4 Function Calls

Other `@triton.jit` functions may be called from a kernel; they are inlined unless declared with `noinline=True`. Only Python built-ins `len`, `list`, `range`, `float`, `int`, `isinstance`, `getattr`, `hasattr`, `min`, `max`, `print` (= `tl.device_print`) are recognised inside a kernel, together with everything under `triton.language`, `tl.constexpr` globals, and module-level constants that are `constexpr`-valued.

---

## 12. Debug and Introspection

```python
tl.static_print(*values)                     # compile-time print of constexpr values / types
tl.static_assert(cond, msg="")               # compile-time assertion
tl.device_print(prefix, *args, hex=False)    # device printf; `print(...)` inside a kernel maps to this
tl.device_assert(cond, msg="", mask=None)    # runtime assert; active when debug=True / TRITON_DEBUG=1
tl.debug_barrier()                           # barrier across the program's threads (__syncthreads)
```

`tl.device_print` output from many programs interleaves; it changes code generation and timing.

---

## 13. Compiler Hints

```python
tl.assume(cond)                      # integer assumption (section 3.6)
tl.multiple_of(x, values)            # every element of x is a multiple of values
tl.max_contiguous(x, values)         # the first `values` elements of x are contiguous
tl.max_constancy(x, values)          # elements are equal in groups of `values`
```

Typical use on offset vectors so the backend can vectorise addressing:

```python
offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
```

A hint that is false at run time is undefined behaviour.

---

## 14. `@triton.jit`

```python
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    ...

@triton.jit(do_not_specialize=["stride"],            # names or indices: no divisibility specialisation
            do_not_specialize_on_alignment=["n"],
            debug=False,                             # True: keep device assertions
            noinline=False,                          # True: compile as a separate device function
            launch_metadata=None)                    # callable(grid, kernel, args) -> dict, for profilers
def kernel(...): ...
```

Compilation and caching: a new binary is produced for every distinct combination of `constexpr` values, argument dtypes, specialisation attributes (16-byte divisibility of pointers/integers) and compile options; binaries are cached on disk under `TRITON_CACHE_DIR`. Runtime integer values that are not `constexpr` do not trigger recompilation (except through the divisibility attribute).

`triton.constexpr_function` marks a plain Python function that may be called at compile time on `constexpr` values.

### 14.1 `@triton.heuristics` — Derived constexpr Values

```python
@triton.heuristics(values={"BLOCK": lambda args: triton.next_power_of_2(args["n_cols"])})
@triton.jit
def kernel(x_ptr, n_cols, BLOCK: tl.constexpr):
    ...
```

Each lambda receives the dict of launch arguments and its result is injected as that `constexpr` at every launch (no benchmarking is involved; the value becomes part of the compilation key).

---

## 15. Randomness (Philox)

```python
tl.rand(seed, offset, n_rounds=10)      # float32 uniform in [0, 1)
tl.randn(seed, offset, n_rounds=10)     # float32 standard normal
tl.randint(seed, offset, n_rounds=10)   # int32
tl.rand4x(seed, offsets, n_rounds=10)   # four independent uniform blocks per offset
tl.randn4x(...)   tl.randint4x(...)
```

`seed` is a scalar, `offset` an integer block; the stream is a deterministic function of `(seed, offset)`.

```python
noise = tl.randn(seed, offs)          # same shape as offs, float32
y = x.to(tl.float32) + sigma * noise
```

---

## 16. Host Helpers

```python
triton.cdiv(a, b)                 # ceil division of Python ints (host); tl.cdiv is the in-kernel form
triton.next_power_of_2(n)         # host only; no in-kernel equivalent — compute BLOCK on the host
triton.set_allocator(fn)          # workspace allocator for in-kernel tensor descriptors (section 4.5)
driver.active.get_device_capability()   # (major, minor) of the current device
driver.active.get_current_target()      # GPUTarget(backend, arch, warp_size)
```

---

## 17. Constraints and Pitfalls

### 17.1 Shapes are constexpr
`tl.arange` bounds, `tl.zeros`/`tl.full` shapes, `tl.dot` operand shapes, and every `block_shape` must be compile-time constants (section 3.5).

### 17.2 `tl.advance` returns a new pointer
`ptr = tl.advance(ptr, (0, BLOCK))` — the un-assigned form has no effect.

### 17.3 Tensors are immutable
`acc += v` and `ptrs += BLOCK` re-bind names (fine); `x[i] = v` is not supported.

### 17.4 Masks or boundary checks are not optional
Block extents are powers of two; the problem size usually is not. Mask tensor-of-pointer accesses, or `boundary_check` block-pointer accesses, whenever an extent can exceed the remaining size. Pick `BLOCK = triton.next_power_of_2(n)` on the host when one block must cover a whole row.

### 17.5 Precision of accumulation
Reductions and `tl.dot` accumulate in the dtype stated in sections 9 and 10. Loaded fp16/bf16 data is kept in that dtype by elementwise ops; `.to(tl.float32)` before `tl.exp`/`tl.log`/`tl.sqrt`/... is mandatory (section 8.3), and `tl.store` casts back to the destination element type.

### 17.6 Two load styles do not mix
Tensor-of-pointers: `tl.load(ptr + offs, mask=..., other=...)`. Block pointer: `tl.load(block_ptr, boundary_check=..., padding_option=...)`. Passing `mask`/`other` with a block pointer, or `boundary_check` with a tensor of pointers, is a compile error.

### 17.7 `num_stages` has two meanings
Kernel-level `num_stages` (launch option) pipelines only loads feeding `tl.dot`; `tl.range(..., num_stages=n)` pipelines the loads of that loop body (section 11.2).

### 17.8 fp8 operands of `tl.dot`
`float8e4nv` requires capability >= 8.9; `float8e4b15` is deprecated and up-cast on >= 9.0; fp8 x fp8 produces `float32` (or `float16` via `out_dtype`); `K >= 32` for 8-bit operands.

### 17.9 Common failure signatures

| Symptom | Cause |
|---------|-------|
| `CompileTimeAssertionFailure` | `tl.static_assert` failed: inspect constexpr values with `tl.static_print` |
| `triton.OutOfResources` | the configuration needs more registers or shared memory than the device offers: reduce block sizes, `num_stages`, or set `maxnreg` |
| `CUDA error: misaligned address` | descriptor/block pointer on a base or stride that violates the 16-byte rules (section 4.5) |
| all-zero output | missing or wrong `mask` on `tl.store`, or accumulator never updated |
| NaN output | `tl.exp` of an unbounded value, division by an empty reduction (zero), or fp16 input to a math function (compile error for fp16, overflow for values > 65504) |
| non-deterministic output | floating-point atomics (order-dependent) |
| `ValueError: arange's range must be a power of 2` | non-power-of-two `tl.arange` extent (section 6) |
| `num_ctas > 1 requires NVIDIA SM90+` | cluster launch on a pre-Hopper target |

---

## 18. Environment Variables and IR Inspection

| Variable | Effect |
|----------|--------|
| `TRITON_INTERPRET=1` | run kernels in the pure-Python interpreter (`print`, `pdb` work; no GPU code generated; very slow) |
| `TRITON_CACHE_DIR=/path` | location of the compiled-kernel cache (`.ttir`, `.ttgir`, `.llir`, `.ptx`, `.cubin` per compilation key) |
| `TRITON_ALWAYS_COMPILE=1` | ignore the cache and recompile |
| `TRITON_DEBUG=1` | enable device-side assertions (`tl.device_assert`, overflow sanitising) |
| `TRITON_F32_DEFAULT=ieee` | default `input_precision` for fp32 `tl.dot` (`tf32` otherwise) |
| `TRITON_KERNEL_DUMP=1`, `TRITON_DUMP_DIR=/path` | dump every compilation stage of every kernel to a directory |
| `MLIR_ENABLE_DUMP=1`, `LLVM_IR_ENABLE_DUMP=1` | print MLIR / LLVM IR after each pass to stderr (read by the C++ library) |
| `TRITON_OVERRIDE_ARCH=sm90` | compile for a different architecture than the active device |

# Device context: B200

# B200 Device Context (snapshot 2026-10-06)

Sourced hardware and environment facts only. Every number carries a unit, a scope and a
source tag `[S<n>]` (see Sources). `unknown` means no allowed source states the value.
Whether a DSL (Triton, cuTile, TileLang) exposes a mechanism listed here is NOT asserted in
this file; confirm it in the backend Reference Skill. No benchmark-derived guidance, no
measured throughput of this benchmark, no scoring targets.

## Identity

- Product name: "NVIDIA B200" (`torch.cuda.get_device_properties().name`) [S1].
- Architecture: NVIDIA Blackwell, compute capability 10.0 [S1]. Framework label: `tilebench.hardware.detect_arch()` maps capability (10, 0) to `"blackwell"` [S5].
- Capture host: dgx003, 2026-10-05, runtime capture (torch + nvidia-smi) [S1]. `verified_on_device: true`.
- Die: "Two reticle-limited dies connected by a 10 terabytes per second (TB/s) chip-to-chip interconnect", "208 billion transistors" [S6-6]. Whether and how the L2 / SM set is partitioned across the two dies: unknown.

## Execution model (native terms)

| Item | Value | Scope | Source |
|---|---|---|---|
| Streaming Multiprocessors (SMs) | 148 | whole device | [S1] |
| Warp size | 32 threads | per warp | [S1][S6-1] |
| Max resident threads | 2048 | per SM | [S1][S6-1] |
| Max resident warps | 64 | per SM | [S6-1][S6-2] |
| Max resident thread blocks | 32 | per SM | [S6-1][S6-2] |
| Max threads per thread block | 1024 | per block | [S1][S6-1] |
| Max block dimensions (x, y, z) | 1024, 1024, 64 | per block | [S6-1] |
| Max grid dimensions (x; y or z) | 2^31-1; 65535 | per grid | [S6-1] |
| Thread block cluster, portable max | 8 blocks | per cluster | [S6-2] |
| Max SM clock | 1965 MHz | whole device | [S1] |
| Memory clock | 3996 MHz | whole device | [S1] |
| Memory bus width | 7680 bit | whole device | [S1] |

## Compute capabilities by dtype (capability only; DSL support to be confirmed per Reference Skill)

- Matrix unit: 5th-generation Tensor Core, PTX `tcgen05` family, "sm_100a / sm_100f" [S6-3]. `tcgen05.mma` kinds: `kind::f16`, `kind::tf32`, `kind::f8f6f4`, `kind::i8`, `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4`, `kind::ti16` [S6-3]. The `.f8f6f4type` operand set is `{.e4m3, .e5m2, .e3m2, .e2m3, .e2m1}`; block-scaled kinds use `.e2m1` operands with `.ue8m0` scales (`mxf4`) or `.ue8m0`/`.ue4m3` scales (`mxf4nvf4`) [S6-3]. The verbatim operand table of `kind::f16` was not captured in the fetch; FP16/BF16 Tensor Core throughput is listed by [S6-4].
- Tensor-Core dtypes with vendor throughput figures: FP4, FP8, FP6, INT8, FP16, BF16, TF32, FP64 [S6-4]. "Second-generation Transformer Engine", FP4 and "community-defined microscaling formats" [S6-6].
- Vendor peak throughput, HGX B200 platform page, 8-GPU totals, footnote "1. Sparse specification. 2. Dense specification." [S6-4]. The page shows FP4 as sparse | dense explicitly; the FP8/FP6, INT8, FP16/BF16 and TF32 rows carry the sparse footnote, so their dense value is one-half of the displayed figure. Per-GPU values are the 8-GPU dense totals divided by 8 (derived, not stated by the page):

| dtype | HGX B200 displayed (8 GPUs) | 8-GPU dense | per-GPU dense (derived) |
|---|---|---|---|
| FP4 Tensor Core | 144 \| 72 PFLOPS (sparse \| dense) | 72 PFLOPS | 9 PFLOP/s |
| FP8 / FP6 Tensor Core | 72 PFLOPS (sparse) | 36 PFLOPS | 4.5 PFLOP/s |
| INT8 Tensor Core | 72 POPS (sparse) | 36 POPS | 4.5 POP/s |
| FP16 / BF16 Tensor Core | 36 PFLOPS (sparse) | 18 PFLOPS | 2.25 PFLOP/s |
| TF32 Tensor Core | 18 PFLOPS (sparse) | 9 PFLOPS | 1.125 PFLOP/s |
| FP32 | 600 TFLOPS | 600 TFLOPS | 75 TFLOP/s |
| FP64 / FP64 Tensor Core | 296 TFLOPS | 296 TFLOPS | 37 TFLOP/s |

  These are vendor specifications of the hardware (dense, no 2:4 sparsity), not measured or achievable rates of any kernel, and not the scoring targets of this benchmark.

## Memory hierarchy

| Level | Value | Scope | Source |
|---|---|---|---|
| 32-bit registers | 65,536 (64 K; 256 KiB) | per SM | [S1][S6-1][S6-2] |
| Max 32-bit registers | 255 | per thread | [S6-1][S6-2] |
| Max 32-bit registers | 64 K | per thread block | [S6-1] |
| Shared memory, default limit | 49,152 B (48 KiB) | per block | [S1] |
| Shared memory, opt-in max | 232,448 B (227 KiB); "227 KB" | per block | [S1][S6-1][S6-2] |
| Shared memory | 233,472 B (228 KiB); "228 KB" | per SM | [S1][S6-1][S6-2] |
| Shared memory banks | 32 | per SM | [S6-1] |
| Unified L1 / texture / shared cache | 256 KB | per SM | [S6-2] |
| Tensor Memory (TMEM) | 512 columns x 128 lanes, 32-bit cells = 256 KiB (derived) | per CTA-visible TMEM of one SM ("per CTA" in PTX wording) | [S6-3] |
| Local memory max | 512 KB | per thread | [S6-1] |
| Constant memory | 64 KB; cache working set 8 KB per SM | whole device / per SM | [S6-1] |
| L2 cache | 132,644,864 B (126.5 MiB) | whole device (runtime-reported) | [S1] |
| HBM3e capacity | 191,495,471,104 B (torch total_memory, 178.3 GiB); 183,359 MiB (nvidia-smi); 1,440 GB total / 8 GPUs = 180 GB (derived) | whole device | [S1][S6-5] |
| HBM3e bandwidth, vendor specification | DGX B200 "64 TB/s HBM3e bandwidth" total / 8 GPUs = 8 TB/s per GPU (derived) | whole device | [S6-5] |

Framework facts [S5]: `last_level_cache_bytes()` returns the runtime `L2_cache_size` on Blackwell (no `_LLC_BYTES` override), i.e. 132,644,864 B; the timer's eviction buffer is "2x" that size (rule stated in the `hardware.py` comment).

## Data movement

- Tensor Memory Accelerator (TMA): PTX `cp.async.bulk.tensor` "Requires sm_90 or higher"; the `.multicast::cluster` qualifier is "advised to be used with .target sm_90a or sm_100f or sm_100a ..." [S6-3]. Framework: `supports_tma()` is True for `blackwell` [S5].
- Tensor Memory load/store and allocation: TMEM "must be allocated by a single warp in a CTA"; "The unit of allocation is 32 columns and the number of columns being allocated must be a power of 2. When a column is allocated, all 128 lanes of the column are allocated."; all allocated TMEM "must be explicitly deallocated before the kernel exits" [S6-3]. Framework: `supports_tmem()` is True only for `blackwell` [S5].
- Thread block clusters (portable size 8) and distributed shared memory (a block may access the shared memory of other blocks in its cluster) [S6-2].
- Hardware decompression engine ("LZ4, Snappy, and Deflate") [S6-6] — not a kernel-level mechanism.
- Asynchronous non-tensor bulk copies (`cp.async`, `cp.async.bulk`): not fetched from vendor docs for this snapshot; capability: unknown.

## Limits

- 1024 threads per block; 2048 threads, 64 warps, 32 blocks per SM; 255 registers per thread; 64 K registers per block and per SM; 227 KB opt-in shared memory per block; 228 KB shared memory per SM [S1][S6-1][S6-2].
- Max SM clock 1965 MHz; memory clock 3996 MHz [S1]. Base/boost clock schedule under load: unknown.
- TMEM: 512 columns per SM, allocation unit 32 columns, power-of-two column counts [S6-3].

## Software snapshot (host dgx003, captured 2026-10-05) [S1]

| Component | Version |
|---|---|
| NVIDIA driver | 595.58.03 |
| CUDA runtime (torch.version.cuda) | 13.0 |
| torch | 2.10.0+cu130 |
| triton | 3.6.0 |
| cuda-tile | 1.5.0 |
| tilelang | 0.1.11 |
| Python | unknown |
| apache-tvm-ffi | unknown |
| tileiras (CUDA Tile compiler used by cuda-tile) | 13.2 [S7] |
| nvcc / nvvm | unknown |
| OS / kernel | unknown |
| Proton backend | unknown |

APIs whose documented minimum tileiras version is 13.3 or later are unavailable in this B200 environment.

## Sources

1. **[S1]** Runtime capture on dgx003, 2026-10-05: `torch.cuda.get_device_properties(0)` + `nvidia-smi` (name, compute capability, multi_processor_count, warp_size, max_threads_per_block, max_threads_per_multi_processor, regs_per_multiprocessor, shared_memory_per_block, shared_memory_per_block_optin, shared_memory_per_multiprocessor, L2_cache_size, total_memory, clock_rate, memory_clock_rate, memory_bus_width, driver, torch/CUDA/triton/cuda-tile/tilelang versions).
2. **[S5]** `tilebench/hardware.py`, worktree `/projects/kzhou6/bcui2/research/tilebench/llm_wt` at HEAD ea04fb36 (`_NVIDIA_ARCH`, `supports_tma`, `supports_tmem`, `last_level_cache_bytes`).
3. **[S6-1]** NVIDIA CUDA C Programming Guide v12.9.0 (archived single page), "Technical Specifications per Compute Capability", column 10.x — https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#features-and-technical-specifications (accessed 2026-10-05). The current v13.4.2 split page `05-appendices/compute-capabilities.html` returned HTTP 404 on 2026-10-05.
4. **[S6-2]** NVIDIA Blackwell Tuning Guide — https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html (accessed 2026-10-05).
5. **[S6-3]** NVIDIA PTX ISA, sections 9.7.18 (TensorCore 5th Generation: Tensor Memory, `tcgen05.mma` kinds, allocation) and `cp.async.bulk.tensor` Target ISA Notes — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (accessed 2026-10-05).
6. **[S6-4]** NVIDIA HGX platform page, HGX B200 column and footnotes ("1. Sparse specification. 2. Dense specification.") — https://www.nvidia.com/en-us/data-center/hgx/ (accessed 2026-10-05).
7. **[S6-5]** NVIDIA DGX B200 page ("1,440 GB total, 64 TB/s HBM3e bandwidth", 8 GPUs) — https://www.nvidia.com/en-us/data-center/dgx-b200/ (accessed 2026-10-05).
8. **[S6-6]** NVIDIA Blackwell architecture page — https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ (accessed 2026-10-05).
9. **[S7]** `tileiras --version` on dgx003, 2026-10-07: "Cuda compilation tools, release 13.2, V13.2.78" (the compiler cuda-tile uses on this host).



# Canonical algorithm contract: vector_add

# vector_add: canonical algorithm contract

## Functional semantics

Elementwise addition of two 1-D tensors of `n` elements with the same
dtype:

    out[i] = x[i] + y[i]        for 0 <= i < n

This is the reference `x + y`. For int8 the sum follows PyTorch's integer
semantics (wraparound on overflow); the benchmark generator keeps int8
inputs in a range where no overflow occurs, so the result must equal the
exact integer sum. There is no reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(x, y)`: `x`, `y` positional; no keyword
arguments are passed.

- `x`: `(n,)`, contiguous, dtype one of fp16, bf16, fp32, int8. Read-only.
- `y`: `(n,)`, contiguous, same dtype and length as `x` (not validated by
  the benchmark). Read-only.
- Returned: one new tensor of `x`'s shape and dtype, allocated inside
  `run()` on every call. It must not alias either input.

## Required logical stages

1. **Allocate** the output.
2. **Elementwise add**: load each element of `x` and of `y`, add, and
   store, with edge handling where the last tile is partial.

Stage 2 is a single logical stage with no inter-element dependency: each
element of `x` and `y` is read once and each output element written once,
and no intermediate tensor is materialised in global memory. No second pass
over the data is permitted, because it would change the algorithm rather
than its mapping. These are logical traversal counts, not a guarantee about
physical DRAM transactions, which caches and the compiler may change. How
the elements are split across programs or launches is a mapping choice.

## Algorithm family and structure

Memory-bound 1-D elementwise binary map: two logical reads and one logical
write per element. No reduction, scan or sort.

## Precision and accumulation

- The add is performed in the input dtype (fp16 in fp16, bf16 in bf16,
  fp32 in fp32, int8 in int8), exactly as the reference does. Computing in a
  wider type and casting back once is accepted for the floating dtypes
  (within the verifier's per-dtype defaults) and for int8 provided the
  cast reproduces the exact int8 sum.
- The verifier compares integer dtypes exactly; floating dtypes use the
  per-dtype default tolerances (no `verify` override).
- No accumulation.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

The inputs are consumed as given; no copy, cast, cached state or
precomputation outside `run()`.

## Permitted implementation mappings

- Elements per program, the assignment of elements to programs (contiguous
  blocks or any other partition), grid shape, pipelining, vector width and
  other launch parameters.
- The number of launches is free: the element range may be covered by one
  launch or partitioned across several, provided each element is still
  processed once and no intermediate is written to global memory.
- Explicit masks or the DSL's bounds-padded loads and bounds-clipped stores
  for a last partial block. Edge handling is required wherever a case's `n` is not a multiple of the chosen tile (no element at index `>= n`
  may be read or written); supporting shapes outside the task's configured cases is not required.
- Whether `n` is a compile-time constant or a runtime argument.

## Forbidden substitutions

- Returning `x + y`, `torch.add`, `x.add(y)`, `out.copy_(x + y)` or any
  other PyTorch arithmetic on the tensors inside `run()`.
- In-place accumulation into an input (`x.add_(y)`, `x += y`) or
  returning a view of an input.
- A dtype conversion that changes the rounding of the result (for example
  adding fp16 inputs in fp16 after an intermediate bf16 cast), or
  producing an output dtype different from `x`'s.
- Any additional logical pass over the data or global scratch buffer,
  because either changes the algorithm's data movement rather than its
  mapping.

## Permitted PyTorch operations

- `torch.empty_like(x)` or `torch.empty(...)` for the output only.
- Reading `.shape`, `.dtype`, `.device`, `Tensor.numel`, `Tensor.stride`;
  `Tensor.contiguous` / `Tensor.view` as no-op guards on the flat inputs.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.

# Task (unchanged)

- Operator: `vector_add`; DSL `triton` 3.6.0; output file `impl_triton.py`
- Datatype: `fp16` (torch `torch.float16`)
- Evaluated inputs: 20 configured cases of this dtype, the same set in every round; the file must be correct and is timed on each of them. Parameter domain:
  - `n`: an integer from `1048576` to `20971520`, always a multiple of `1048576`
- Numerical acceptance: `atol=0.001, rtol=0.001` (source: config.verify+dtype_default)
- Interface: `def run(x, y):`; returns the output tensor(s) described in the contract.

## PyTorch functional reference

```python
def run(x, y):
    return x + y
```

# Optimization round 5 of 5

## Previous candidate (round 4)

```python title="impl_triton.py"
# slower candidate source
```

Configurations reported by `get_last_config()` across the cases (1 distinct): `{"BLOCK": 256}`

Outcome: compiled, passed numerical verification and was timed on all 20 cases: geometric-mean runtime 0.0532 ms


## Best valid candidate so far (round 3, geometric mean 0.0410 ms over the 20 cases)

```python title="impl_triton.py"
# best valid candidate source
```

Configurations reported by `get_last_config()` across the cases (2 distinct): `{"BLOCK": 1024}`, `{"BLOCK": 2048}`

## Runtime history of this task (valid candidates only: all 20 cases valid; geometric mean over the 20 cases of the per-case runtime, each the mean of 3 timed runs after 1 warmup)

| round | valid cases | geometric-mean ms |
|---|---|---|
| 3 | 20/20 | 0.0410 |
| 4 | 20/20 | 0.0532 |

Improve on the best valid geometric-mean runtime while keeping the contract and staying correct on every case. Return exactly one fenced block titled `impl_triton.py`.
