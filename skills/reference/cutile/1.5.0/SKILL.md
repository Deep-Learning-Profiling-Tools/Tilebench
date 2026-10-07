---
name: cutile-reference
dsl: cutile
version: "1.5.0"
kind: reference
derived_from: skills/cutile-guide/SKILL.md@7f25fdf9 + .claude copy @767f4677
revised: 2026-10-05
---

# cuTile (`cuda.tile`) 1.5.0 API Reference

Scope of this document:

- Pinned to `cuda-tile` 1.5.0. Every symbol, signature and default below was checked against the installed package by introspection. Treat any `ct.*` name that is not listed here as unavailable.
- Hardware facts (SM count, cache sizes, memory bandwidth, which `mma` tile shapes reach tensor cores, peak rates) are not in this document; they are in the Device Context document supplied with the task.
- Autotuning is not permitted in this task setting; this reference documents no tuning API.

---

## 1. Imports and execution spaces

```python
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]
```

cuTile distinguishes two execution spaces:

- Host code: ordinary Python. Allocates tensors, computes the grid, calls `ct.launch`.
- Tile code: the body of a `@ct.kernel` or `@ct.function`. Compiled by cuTile from a restricted Python subset. All values are `Tile`s (or scalars that promote to 0-d tiles).

`ct.cdiv(x, y)` (ceiling division) is usable in both spaces. Kernels cannot be called directly; they are queued with `ct.launch`.

---

## 2. Data types

| Integer | Float | Narrow / MMA-oriented |
|---------|-------|------------------------|
| `ct.bool_` (8-bit) | `ct.float16` | `ct.tfloat32` (19-bit in a 32-bit container) |
| `ct.int8` / `ct.uint8` | `ct.bfloat16` | `ct.float8_e4m3fn` |
| `ct.int16` / `ct.uint16` | `ct.float32` | `ct.float8_e5m2` |
| `ct.int32` / `ct.uint32` | `ct.float64` | `ct.float8_e8m0fnu` (scale type, no sign, 8 exponent bits) |
| `ct.int64` / `ct.uint64` | | `ct.float4_e2m1fn` |

- `DType` objects have `.name` and `.bitwidth`; they are immutable, usable in host and tile code, and may be kernel parameters.
- Wherever a `dtype=` argument is accepted, numpy scalar types and dtypes (`np.float32`, `np.dtype("int32")`, ml_dtypes `bfloat16`/fp8 dtypes) and torch dtypes (`torch.float16`, `torch.bfloat16`, ...) are accepted in addition to `ct.*` objects. `array.dtype` of a kernel argument yields the matching `ct` dtype.
- The docstrings classify `float16/bfloat16/float32/float64/bool_/int*/uint*` as *arithmetic* dtypes and `tfloat32`, the fp8 types and `float4_e2m1fn` as *numeric* dtypes (storage and MMA operands). Cast narrow types with `ct.astype` before elementwise math.

---

## 3. Programming model: kernels, grid, launch

### 3.1 `@ct.kernel`

```python
@ct.kernel
def my_kernel(x, y, out, alpha: float, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    ...

@ct.kernel(occupancy=2, num_ctas=4, opt_level=3, num_worker_warps=4)
def hinted_kernel(...):
    ...

@ct.kernel(num_ctas=ct.ByTarget(sm_100=8, sm_120=4, default=2))
def per_target_kernel(...):
    ...
```

A kernel is executed once by every block of the grid. Compiler options accepted by the decorator (each may also be a `ct.ByTarget(...)` object keyed by `"sm_<major><minor>"` strings with an optional `default=`):

| Option | Legal values | Default |
|--------|--------------|---------|
| `num_ctas` | power of two in [1, 16] (CTAs per cluster) | auto |
| `occupancy` | [1, 32] (expected active CTAs per SM) | auto |
| `opt_level` | [0, 3] | 3 |
| `num_worker_warps` | 4 or 8 (warp-specialized kernels; CTK 13.3+, otherwise ignored with a warning) | auto |

`kernel.replace_hints(**hints)` returns a new kernel object with the given options and its own JIT cache; the original is unchanged. In this task setting every hint value must be a fixed literal.

### 3.2 Kernel parameters

| Host argument | Annotation | Inside the kernel |
|---------------|------------|-------------------|
| CUDA tensor (any object exposing `__cuda_array_interface__` or `__dlpack__`, e.g. a torch CUDA tensor) | none | `ct.Array` |
| Python `int` | none | runtime `int32` scalar |
| Python `int` | `ct.ScalarInt64` | runtime `int64` scalar |
| Python `float` / `bool` | none | runtime scalar |
| any value | `ct.Constant[int]`, `ct.Constant[bool]`, `ct.Constant[float]`, bare `ct.Constant` | compile-time constant embedded in the binary; a new value triggers a new compilation |
| `ct.DType` | none | dtype constant |
| tensor | `Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64, static_shape_dims=(0, -1))]` | shape/stride values in int64; listed shape dims become compile-time constants |
| tensor | `ct.IndexedWithInt64` | shape and strides in int64 |
| list of tensors | `typing.Annotated` metadata `ct.ListAnnotation(element=...)`, where `element` is an `ArrayAnnotation` or an `Annotated` array type such as `ct.IndexedWithInt64` | list of arrays |

Compilation happens on first launch for each distinct combination of constant values and argument types, and is cached.

### 3.3 `ct.launch`

```python
ct.launch(stream, grid, kernel, kernel_args)   # all positional
```

- `stream`: CUDA stream, e.g. `torch.cuda.current_stream()`.
- `grid`: tuple of up to 3 block counts, `(x,)`, `(x, y)` or `(x, y, z)`.
- `kernel`: a `@ct.kernel` object (possibly from `replace_hints`).
- `kernel_args`: tuple in the kernel's parameter order.

### 3.4 Block and tile-space queries (tile code)

```python
ct.bid(axis)                      # int32 block index along axis 0, 1 or 2
ct.num_blocks(axis)               # int32 number of blocks along axis
ct.num_tiles(array, axis, shape, order="C")   # int32 tiles along axis when array is
                                              # partitioned into tiles of `shape`
```

### 3.5 `@ct.function`

```python
@ct.function                      # tile-only helper (default: host=False, tile=True)
def helper(x, y): return x + y

@ct.function(host=True)           # callable from host and tile code
def cdiv2(a, b): return (a + b - 1) // b
```

Unannotated Python functions called from tile code are inlined as tile functions automatically (recursively). Recursion limits raise `TileRecursionError` at compile time.

---

## 4. Core objects

### 4.1 `Array` (global memory view of a kernel argument)

```python
array.dtype      # DType, compile-time constant
array.ndim       # int, compile-time constant
array.shape      # tuple of runtime int32 scalars (int64 under IndexedWithInt64);
                 # a dim is a compile-time constant only if listed in static_shape_dims
array.strides    # tuple of runtime scalars, in elements
array.slice(axis, start, stop)             # view restricted to [start, stop) along axis;
                                           # axis is a constant (negative allowed),
                                           # start/stop are ints or 0-d tiles; no copy
array.tiled_view(tile_shape, *, padding_mode=ct.PaddingMode.UNDETERMINED,
                 traversal_steps=None)     # -> TiledView (see 6.3)
array.get_raw_memory()                     # -> RawArrayMemory (see 6.6)
```

Because `array.shape` is a runtime value, it cannot be used where a compile-time shape is required (tile shapes, `reshape` targets). Derive those from `ct.Constant` parameters or use `static_shape_dims`.

### 4.2 `Tile`

Immutable, block-local, N-dimensional value. Every dimension must be a power of two (0-d tiles are allowed). Properties `dtype`, `shape` (tuple of constants), `ndim`.

Methods: `astype(dtype)`, `reshape(shape)`, `permute(axes)`, `transpose(axis0=None, axis1=None)`, `extract(index, shape)`, `item()` (equivalent to `reshape(())`).

Operators: `+ - * / // % ** @`, bitwise `& | ^ ~`, unary `-`, comparisons `== != < <= > >=` (bool tiles), reflected forms for `scalar op tile`. `tile[:, None]` and `tile[None, :]` are syntax sugar for `expand_dims` only; integer indexing or slicing with `[]` is not supported (use `ct.extract`). A 0-d integer tile can be used as an index (`__index__`). Python builtins `max(x, y)` and `min(x, y)` map to `ct.maximum` / `ct.minimum`.

Results of every operation are new tiles; `t += 1` must be written `t = t + 1`.

### 4.3 `Scalar`

`int | float` Python scalars promote to 0-d tiles in binary operations with tiles.

---

## 5. Enumerations

| Enum | Members | Used by |
|------|---------|---------|
| `ct.PaddingMode` | `UNDETERMINED` (default), `ZERO`, `NEG_ZERO`, `NAN`, `POS_INF`, `NEG_INF` | `load`, `tiled_view`, `load_advanced_indexing` |
| `ct.RoundingMode` | `RN` (nearest even, default), `RZ`, `RM`, `RP`, `FULL`, `APPROX`, `RZI` | float arithmetic, `sqrt`, `sum`, `prod`, `cumsum`, `cumprod`; `exp`/`tanh` accept only `FULL`/`APPROX` (f32) |
| `ct.MemoryOrder` | `WEAK`, `RELAXED`, `ACQUIRE`, `RELEASE`, `ACQ_REL` | atomics (default `ACQ_REL`), `load` (`WEAK`/`RELAXED`/`ACQUIRE`), `store` (`WEAK`/`RELAXED`/`RELEASE`) |
| `ct.MemoryScope` | `NONE`, `BLOCK`, `CLUSTER`, `DEVICE`, `SYS` | atomics (default `DEVICE`), `load`/`store` (default `NONE`; meaningful only with a non-`WEAK` order) |

Padding choice rule: `ZERO` for additive reductions, `NEG_INF` when out-of-bounds elements must not affect `max`/`argmax`/softmax, `POS_INF` for `min`.

---

## 6. Memory operations

### 6.1 `ct.load`

```python
tile = ct.load(array, index, shape, *,
               order="C",
               padding_mode=ct.PaddingMode.UNDETERMINED,
               latency=None,            # const int 1..10, DRAM-traffic hint
               allow_tma=None,          # const bool; False disables TMA
               memory_order=ct.MemoryOrder.WEAK,
               memory_scope=ct.MemoryScope.NONE)
```

Semantics: the array is partitioned into a *tile space* of equally sized tiles of `shape`; `index` is one coordinate in that space (one tuple entry per array dimension). For a 2-D array and tile `(tm, tn)`:

```
t = ct.load(array, (i, j), (tm, tn))
t[x, y] = array[i*tm + x, j*tn + y]      for 0 <= x < tm, 0 <= y < tn
```

- `index` entries are scalars: Python ints, runtime int32 scalars (`ct.bid(0)`, loop variables, arithmetic on them) or 0-d integer tiles. One coordinate per call, uniform for the block.
- `shape` entries are compile-time constants and powers of two; `shape=()` loads a 0-d tile; an int is accepted for 1-D (`shape=4`).
- `order`: permutation applied to the array axes before the tile space is built. `"C"` = identity, `"F"` = reversed, or an explicit tuple. A transposed load of the tile above is `ct.load(array, (j, i), shape=(tn, tm), order=(1, 0))`.
- A tile that partially extends past the array is filled according to `padding_mode`; a tile that lies entirely outside the array is undefined behaviour.

```python
x_t = ct.load(x, (bid,), (TILE,))
row = ct.load(x2d, (r, j), (1, TILE), padding_mode=ct.PaddingMode.ZERO)
k_t = ct.load(K, (b, h, 0, j), (1, 1, D, N), order=(0, 1, 3, 2))   # transposed last two axes
s   = ct.load(x, (i,), ())                                          # 0-d tile
```

### 6.2 `ct.store`

```python
ct.store(array, index, tile, *,
         order="C", latency=None, allow_tma=None,
         memory_order=ct.MemoryOrder.WEAK, memory_scope=ct.MemoryScope.NONE)
```

- The tile shape defines the tile space; `index` is a tile-space coordinate with the same scalar rules as `load`. The tile rank must equal the array rank, unless `tile` is a scalar or 0-d tile, which is broadcast.
- Elements of a partially out-of-bounds tile are dropped; an entirely out-of-bounds tile is undefined behaviour.
- `store` writes one rectangular tile at one block-uniform coordinate. Per-element destination indices require `ct.scatter`.

### 6.3 `TiledView` (fixed-tile-shape view of an array)

```python
tv = array.tiled_view((tm, tn), padding_mode=ct.PaddingMode.ZERO, traversal_steps=None)
tv.tile_shape            # tuple of constants
tv.traversal_steps       # element step between tile origins per axis (defaults to tile_shape)
tv.num_tiles(axis)       # int32
tv.load(index, *, latency=None, allow_tma=None)      # padding from the view
tv.store(index, tile, *, latency=None, allow_tma=None)   # implicit cast to view dtype;
                                                         # tile broadcastable to tile_shape
tv.atomic_store_add(index, update)   # also atomic_store_and/_max/_min/_or/_xor; no return value
```

`traversal_steps[i] < tile_shape[i]` produces overlapping tiles (sliding windows), `>` leaves gaps (CTK 13.3+).

### 6.4 `ct.gather` / `ct.scatter` (per-element indexing)

```python
t = ct.gather(array, indices, *, mask=None, padding_value=0, check_bounds=True, latency=None)
ct.scatter(array, indices, value, *, mask=None, check_bounds=True, latency=None)
```

- `indices`: tuple of length `array.ndim`; each entry an integer tile or scalar; all entries are broadcast to a common shape, which is the result shape (gather) or the shape `value` must broadcast to (scatter). For a 1-D array a single tile may be passed instead of a 1-tuple.
- Example with 2-D array, `ind0` shape `(M, N, 1)`, `ind1` shape `(M, 1, K)`: `ct.gather(array, (ind0, ind1))` has shape `(M, N, K)` with `t[i, j, k] = array[ind0[i, j, 0], ind1[i, 0, k]]`.
- `mask`: boolean tile broadcastable to the index shape; `False` elements return `padding_value` (gather) or are not stored (scatter). With `check_bounds=True` the effective mask is `mask AND in-bounds`.
- Out-of-bounds indices (including negative ones, which are *not* Python-style) return `padding_value` / are skipped. `check_bounds=False` makes any out-of-bounds access undefined behaviour.

```python
offs = ct.arange(TILE, dtype=ct.int32) + bid * TILE
vals = ct.gather(x, offs, mask=offs < n, padding_value=0.0)
ct.scatter(out, (dest_row, offs), vals)
```

### 6.5 `ct.load_advanced_indexing` / `ct.store_advanced_indexing` (one sparse dim)

```python
t = ct.load_advanced_indexing(array, indices, *, padding_mode=ct.PaddingMode.UNDETERMINED,
                              latency=None, allow_tma=None)
ct.store_advanced_indexing(array, indices, tile, *, latency=None, allow_tma=None)
```

`indices` has length `array.ndim`; exactly one entry is a 1-D integer tile (the sparse dim, element-space indices) and every other entry is `ct.Slice(start, length)` with a runtime `start` and a compile-time power-of-two `length`. Result shape = index-tile length on the sparse dim, `Slice.length` on dense dims; `padding_mode` applies to both.

```python
rows = ct.arange(4, dtype=ct.int32) * 2
t = ct.load_advanced_indexing(x, (rows, ct.Slice(col0, 8)), padding_mode=ct.PaddingMode.ZERO)
```

### 6.6 `RawArrayMemory` (element-offset addressing)

```python
raw = array.get_raw_memory()
t = raw.load_offset(offset, *, mask=None, padding_value=0, latency=None)   # offset in elements
raw.store_offset(offset, value, *, mask=None, latency=None)
```

`offset` is an integer scalar or tile; result/value shape follows the offset shape.

---

## 7. Atomic operations

All return a tile of the *old* values; each element is updated atomically but the set of elements is not, and write order is unspecified. Index convention as in `gather`/`scatter` (tuple of length `array.ndim`, or a single tile for 1-D arrays); `update` is a scalar or tile broadcastable to the index shape.

```python
ct.atomic_add(array, indices, update, *, check_bounds=True,
              memory_order=ct.MemoryOrder.ACQ_REL, memory_scope=ct.MemoryScope.DEVICE)
ct.atomic_max / ct.atomic_min / ct.atomic_and / ct.atomic_or / ct.atomic_xor / ct.atomic_xchg
    # same signature as atomic_add
ct.atomic_cas(array, indices, expected, desired, *, check_bounds=True,
              memory_order=..., memory_scope=...)   # writes desired where old == expected
```

Out-of-bounds elements: no operation; `atomic_cas` returns the corresponding `expected`, the others an implementation-defined value. `check_bounds=False` makes out-of-bounds access undefined behaviour. When the old value is not needed, `TiledView.atomic_store_add` and friends (6.3) express a tile-shaped atomic update without a return value.

---

## 8. Tile creation

```python
ct.full(shape, fill_value, dtype)          # shape tuple of constants; fill int/float/bool
ct.zeros(shape, dtype)
ct.ones(shape, dtype)
ct.arange(size, *, dtype, start=0, step=1) # size: constant power of two; 1-D
ct.astile(value, *, dtype)                 # scalar -> 0-d tile; nested tuples of scalars
                                           # (power-of-two lengths, uniform per level) -> N-d tile
```

```python
acc  = ct.zeros((TM, TN), ct.float32)
m_i  = ct.full((TM, 1), float("-inf"), ct.float32)
offs_m = ct.arange(TM, dtype=ct.int32)[:, None]      # (TM, 1)
offs_n = ct.arange(TN, dtype=ct.int32)[None, :]      # (1, TN)
```

---

## 9. Shape and view operations

```python
ct.reshape(x, shape)              # one entry may be -1; element count preserved
ct.permute(x, axes)
ct.transpose(x, axis0=None, axis1=None)   # 2-D: axes optional; >2-D: both required
ct.expand_dims(x, axis)           # same as x[:, None] / x[None, :]
ct.broadcast_to(x, shape)         # NumPy broadcasting rules
ct.cat((a, b), axis)              # exactly two tiles of the SAME shape (power-of-two rule)
ct.extract(x, index, shape)       # sub-tile; `shape` must divide x.shape in every dim;
                                  # `index` is the sub-tile grid coordinate, not an element index
ct.bitcast(x, dtype)              # reinterpret bits, same bitwidth
ct.pack_to_bytes(x)               # flatten to a 1-D uint8 tile (total bits divisible by 8)
ct.unpack_from_bytes(x_u8, dtype) # inverse: 1-D uint8 tile -> 1-D tile of dtype
```

```python
q   = ct.load(Q, (b, h, i, 0), (1, 1, M, D)).reshape((M, D))
sub = ct.extract(t, (0, 1), shape=(32, 32))     # columns 32..63 of the top 32 rows of a (64,64) tile
```

---

## 10. Type conversion and promotion

```python
y = ct.astype(x, ct.float32)      # or x.astype(ct.float32)
y = x.astype(array.dtype)         # match an array's dtype
```

- Binary elementwise operations broadcast shapes and promote both operands to a common dtype; Python scalars adopt the tile's dtype class.
- `ct.mma` does not promote: `x` and `y` must have the same dtype (except an `int8`/`uint8` mix) and `acc` must be in the dtype table of section 15.
- Common pattern: load, cast to `float32`, compute, cast back to the output array's dtype before `store`.
- `float16` represents integers exactly only up to 2048; index-like values should be kept in integer dtypes or `float32`.

---

## 11. Arithmetic, comparison, bitwise

Operators (section 4.2) are the normal form. Explicit functions exist for the cases that take rounding or flush-to-zero control:

```python
ct.add(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.sub(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.mul(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.truediv(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.floordiv(x, y)                 # ints and floats (floor(x / y) as float)
ct.mod(x, y)
ct.pow(x, y)
ct.negative(x)
ct.abs(x)
ct.minimum(x, y, *, flush_to_zero=False)
ct.maximum(x, y, *, flush_to_zero=False)
ct.cdiv(x, y)                     # ceil(x / y); host and tile code

ct.equal, ct.not_equal, ct.less, ct.less_equal, ct.greater, ct.greater_equal   # (x, y) -> bool tile
ct.bitwise_and, ct.bitwise_or, ct.bitwise_xor, ct.bitwise_lshift, ct.bitwise_rshift   # (x, y)
ct.bitwise_not(x)
```

`rounding_mode` applies to float dtypes only (default `RN`); `flush_to_zero=True` flushes subnormal inputs and results to signed zero.

---

## 12. Math functions

```python
ct.exp(x, *, rounding_mode=None)     # rounding_mode: FULL or APPROX, f32 only (CTK 13.3+)
ct.exp2(x, *, flush_to_zero=False)
ct.log(x)          ct.log2(x)
ct.sqrt(x, *, rounding_mode=None, flush_to_zero=False)
ct.rsqrt(x, *, flush_to_zero=False)
ct.pow(x, y)
ct.sin(x)  ct.cos(x)  ct.tan(x)  ct.sinh(x)  ct.cosh(x)
ct.tanh(x, *, rounding_mode=None)    # FULL or APPROX, f32 only (CTK 13.2+)
ct.atan2(x1, x2)                     # radians in [-pi, pi]
ct.floor(x)  ct.ceil(x)  ct.abs(x)  ct.isnan(x)
```

Not available in 1.5.0: `log10`, `log1p`, `sigmoid`, `erf`. Express them with the functions above (for example `ct.log(x) / math.log(10.0)` with a host-side constant).

---

## 13. Reductions and scans

```python
ct.sum(x, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False)
ct.prod(x, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False)
ct.max(x, axis=None, *, keepdims=False, flush_to_zero=False)
ct.min(x, axis=None, *, keepdims=False, flush_to_zero=False)
ct.argmax(x, axis=None, *, keepdims=False)   # ties -> smallest index; axis tuple not supported
ct.argmin(x, axis=None, *, keepdims=False)
```

`axis=None` reduces everything to a 0-d tile; an int or (except for argmax/argmin) a tuple of ints selects axes; `keepdims=True` keeps size-1 axes for broadcasting.

Custom reduction and scan:

```python
ct.reduce(x, axis, func, identity, *, keepdims=False)
ct.scan(x, axis, func, identity, *, reverse=False)          # inclusive prefix
ct.cumsum(x, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False)
ct.cumprod(x, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False)
```

- `x` may be a single tile or a tuple of N tiles (broadcastable to a common shape). For a single tile `func(a, b)` combines two 0-d tiles; for a tuple `func(a1..aN, b1..bN)` returns an N-tuple. `identity` is a constant scalar or tuple of scalars.
- `func` bodies may not contain loops, and reductions/scans may not be nested inside `func`.

```python
# value/index pair reduction (minimum and its index along axis 0)
mn, mi = ct.reduce((vals, idxs), 0,
                   lambda v1, i1, v2, i2: (ct.where(v1 < v2, v1, v2), ct.where(v1 < v2, i1, i2)),
                   (float("inf"), 0))
```

---

## 14. Selection

```python
ct.where(cond, x, y)    # cond: bool tile; x, y: tiles or scalars, promoted to a common dtype
                        # and broadcast with cond to the result shape
```

```python
clipped = ct.where(x > 0, x, 0.0)
masked = ct.where(offs_m >= offs_n, s, float("-inf"))
```

---

## 15. Matrix multiply

### 15.1 `ct.mma` (fused multiply-accumulate; accumulator dtype preserved)

```python
acc = ct.mma(x, y, acc, *, use_fast_acc=False)
# x: [M, K] or [B, M, K];  y: [K, N] or [B, K, N];  acc: [M, N] or [B, M, N]
# returns (x @ y) + acc in acc's dtype; batch dims broadcast
```

| Input (`x`, `y`) | Accumulator / output |
|------------------|----------------------|
| `float16` | `float16` or `float32` |
| `bfloat16` | `float32` |
| `float32` | `float32` |
| `tfloat32` | `float32` |
| `float64` | `float64` |
| `float8_e4m3fn`, `float8_e5m2` | `float16` or `float32` |
| `int8` / `uint8` (may be mixed) | `int32` |

`x` and `y` must share a dtype (except the int8/uint8 mix); no promotion is performed. `use_fast_acc=True` requires fp8 inputs and only affects Hopper (silently ignored elsewhere). Whether a given `(M, K, N)` tile combination is lowered to tensor-core instructions, and the minimum tile dimensions for that, is decided by the compiler for the target device: the API only requires power-of-two tile dims and the dtype table above. See the Device Context for the thresholds that apply to the target GPU.

```python
acc = ct.zeros((TM, TN), ct.float32)
for k in range(num_k_tiles):                       # runtime loop, see section 16
    a = ct.load(A, (i, k), (TM, TK), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(B, (k, j), (TK, TN), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.mma(a, b, acc)
ct.store(C, (i, j), acc.astype(C.dtype))
```

### 15.2 `ct.matmul` / `@`

```python
r = ct.matmul(x, y)      # or x @ y; 1-D, 2-D or 3-D operands, batch broadcast
```

Supported inputs: f16, bf16, f32, f64, tf32, f8e4m3fn, f8e5m2, i8, u8. Operands are promoted to a common dtype, which is also the result dtype; no accumulator argument.

### 15.3 `ct.mma_scaled` (block-scaled MMA)

```python
acc = ct.mma_scaled(x, x_scale, y, y_scale, acc)
# x: [.., M, K], x_scale: [.., M, K_s];  y: [.., K, N], y_scale: [.., K_s, N];  acc: [.., M, N]
# block size B = K // K_s; result[i, j] = sum_k x[i,k]*x_scale[i,k//B]*y[k,j]*y_scale[k//B,j] + acc[i,j]
```

| Input (`x`, `y`) | Scale | Acc / out | Allowed `B` |
|------------------|-------|-----------|-------------|
| `float8_e4m3fn`, `float8_e5m2` | `float8_e8m0fnu` | `float32` | 32 |
| `float4_e2m1fn` | `float8_e8m0fnu` | `float32` | 16, 32 |
| `float4_e2m1fn` | `float8_e4m3fn` | `float32` | 16 |

Availability depends on the target architecture (the package's own tests gate it on Blackwell or newer).

---

## 16. Control flow

### 16.1 Runtime loops and branches

- `for i in range(start, stop, step)` always compiles to a runtime loop, whether the bounds are compile-time constants, `ct.Constant` parameters or runtime scalars such as `ct.num_tiles(...)`. The induction variable is a runtime int32 scalar and therefore cannot be used in tile shapes, `reshape` targets or `extract` shapes.
- `while cond:` compiles to a runtime loop. `break`, `continue` and early `return` are supported in runtime loops.
- Variables assigned inside a loop and read after it are loop-carried; their dtype and shape must be the same on every path (initial value, each iteration, each `break`/`continue`). Initialise accumulators before the loop with the final dtype (`ct.zeros((TM, TN), ct.float32)`).
- `if cond:` / `else:` on a runtime scalar (for example `ct.bid(0) == 0` or a comparison of loaded 0-d tiles) compiles to a runtime branch; both branches are compiled and variables defined in both must agree in type. `and`/`or`/`not` on scalars are allowed in conditions.
- `if` on a compile-time constant (`ct.Constant[bool]` parameter, constant expression) is resolved at compile time.

### 16.2 Compile-time constructs

```python
for G, j in ct.static_iter(STAGES):         # STAGES: Python iterable from constants/globals
    x3 = ct.reshape(x, (G, 2, j))            # loop variables are Python constants ->
    ...                                      # usable in shapes and extract/reshape arguments
k = ct.static_eval(TILE // 2 if TILE > 64 else TILE)   # Python-evaluated constant
ct.static_assert(TILE % 32 == 0, f"TILE={TILE} must be a multiple of 32")
```

- `ct.static_iter(iterable)`: the only compile-time unrolled loop. Allowed only directly as the iterable of a `for`; the iterable is evaluated with Python semantics from constants, globals and compile-time attributes (`x.shape`, `x.dtype`); at most 1000 iterations; the body is inlined per item; `break`/`continue`/`return` are not allowed inside.
- `ct.static_eval(expr)`: evaluates `expr` with full Python semantics at compile time. May reference local/global constants and proxy objects of dynamic values (to read `shape`/`dtype` or to select between dynamic values), but must not perform runtime operations (`ct.static_eval(x + 1)` on a tile is an error) or assign (`:=`). Raises `TileStaticEvalError` on violation.
- `ct.static_assert(condition, message=None)`: compile-time assertion; on failure raises `TileStaticAssertionError` with the evaluated message.

Both are treated like keywords (the expression bypasses tile-code translation) and must be called by their plain `ct.`-qualified names.

---

## 17. Debugging and errors

```python
ct.print(f"bid={bid} acc={acc}", sep=" ", end="\n")   # Python-style; f-string expressions must be tiles
ct.printf("bid=%d val=%f\n", bid_tile, val_tile)       # C-style; tile arguments only;
                                                        # specifiers [diuoxXeEfFgGaA]
ct.assert_(cond_tile, "message")                        # all elements must be True
with ct.compiler_timeout(10):                           # seconds; not thread-safe
    ct.launch(...)
```

All three device-side functions have significant overhead and are for debugging only; outputs from different blocks interleave (`opt_level=0` serialises).

Exceptions (all subclasses of `ct.TileError`): `TileSyntaxError` (unsupported Python construct), `TileTypeError` (bad type or dtype, e.g. an unsupported `mma` combination), `TileValueError`, `TileUnsupportedFeatureError` (feature not supported by the compiler or the GPU), `TileStaticAssertionError`, `TileStaticEvalError`, `TileRecursionError`, `TileCompilerExecutionError` (the `tileiras` compiler failed), `TileCompilerTimeoutError`, `TileInternalError`.

---

## 18. Compiler hints inside tile code

```python
n = ct.assume_divisible_by(n, 128)    # n: integer scalar; divisor: positive constant;
                                      # value unchanged; undefined behaviour if the claim is false
```

This is the only value-level hint in 1.5.0. Kernel-level options are the decorator arguments of section 3.1; memory-level hints are the `latency` and `allow_tma` arguments of the load/store family.

---

## 19. Legality constraints and semantics checklist

1. Every tile dimension is a power of two: `load`/`store` shapes, `arange` size, `cat` operands (which must also be equal in shape), `Slice.length`, `astile` tuple lengths. Round problem sizes up on the host (`1 << (n - 1).bit_length()`) and rely on `padding_mode` for reads; partial out-of-bounds stores drop the excess elements.
2. Tile shapes are compile-time constants: derive them from `ct.Constant` parameters, Python literals, `static_iter` variables or `static_eval`; never from `array.shape` (runtime unless `static_shape_dims`) or loop induction variables.
3. `load`/`store` take one block-uniform tile-space coordinate; data-dependent per-element addresses go through `gather`/`scatter`, `load_advanced_indexing`/`store_advanced_indexing` or `RawArrayMemory`.
4. A tile lying entirely outside its array is undefined behaviour for `load`, `store`, `TiledView` accesses and advanced indexing; partial overlap is padded (loads) or clipped (stores).
5. Tiles are immutable; rebinding is the only way to update (`acc = acc + x`).
6. `mma` requires matching input dtypes and an accumulator from the table in 15.1; keep the accumulator in `float32` (or `int32` for int8) for the whole K loop and cast once at the end.
7. `argmax`/`argmin` do not accept an axis tuple; ties resolve to the smallest index.
8. `extract` indices are sub-tile grid coordinates and `shape` must divide the source shape exactly.
9. `gather`/`scatter`/atomics treat negative indices as out of bounds.
10. `reduce`/`scan` combine functions may not contain loops or nested reductions/scans.
11. `static_iter` is the only unrolled loop; `break`/`continue`/`return` are not allowed inside it. Plain `for`/`while` are runtime loops with consistently typed loop-carried variables.
12. Grid is a tuple of up to three ints; `ct.bid`/`ct.num_blocks` accept axis 0, 1 or 2.
13. Integer kernel scalars are int32 unless annotated `ct.ScalarInt64`; shapes/strides are int32 unless `IndexedWithInt64` / `ArrayAnnotation(index_dtype=ct.int64)`. Use the int64 forms when element counts or strides exceed 2^31 - 1.
14. `exp`/`tanh` accept only `RoundingMode.FULL`/`APPROX` on f32; `exp2` and `rsqrt` take `flush_to_zero` only.

---

## 20. Minimal generic examples (API illustration only)

### 20.1 One tile per block, elementwise with runtime scalars

```python
@ct.kernel
def scale_shift(x, out, alpha: float, beta: float, TILE: ConstInt):
    i = ct.bid(0)
    t = ct.load(x, (i,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    r = ct.astype(t, ct.float32) * alpha + beta
    ct.store(out, (i,), ct.astype(r, out.dtype))     # out-of-range tail dropped

# host
n = x.numel()
TILE = 1024
ct.launch(torch.cuda.current_stream(), (ct.cdiv(n, TILE),), scale_shift,
          (x, out, 2.0, 0.5, TILE))
```

### 20.2 Loop over column tiles with a loop-carried accumulator, per-row scatter

```python
@ct.kernel
def row_abs_max(x, out, N: ConstInt, TR: ConstInt, TC: ConstInt):
    r = ct.bid(0)
    best = ct.full((TR, 1), 0.0, ct.float32)                     # carried: fixed dtype/shape
    for j in range(ct.cdiv(N, TC)):                              # runtime loop
        t = ct.load(x, (r, j), (TR, TC), padding_mode=ct.PaddingMode.ZERO)
        best = ct.maximum(best, ct.max(ct.abs(t.astype(ct.float32)), axis=1, keepdims=True))
    rows = ct.arange(TR, dtype=ct.int32) + r * TR
    ct.scatter(out, rows, best.reshape((TR,)), mask=rows < out.shape[0])
```

### 20.3 Tiled view with padding and `num_tiles`

```python
@ct.kernel
def copy_cast(x, y, TM: ConstInt, TN: ConstInt):
    xv = x.tiled_view((TM, TN), padding_mode=ct.PaddingMode.ZERO)
    yv = y.tiled_view((TM, TN))
    i, j = ct.bid(0), ct.bid(1)
    yv.store((i, j), xv.load((i, j)))        # implicit cast to y's dtype
# grid = (ct.cdiv(M, TM), ct.cdiv(N, TN)) computed on the host
```

---

## 21. Names that do not exist in `cuda.tile` 1.5.0

Do not use: `ct.log10`, `ct.log1p`, `ct.sigmoid`, `ct.erf`, `ct.assume_div_by`, `ct.assume_same_elements`, `ct.assume_bounded`, `ct.optimization_barrier`, `ct.autotune`, `ct.Config`, the module `cuda_tile.dialects.cuda_tile_ops` (`make_token`, `make_tensor_view`, `make_partition_view`, `load_view_tko`, ...), and `ct.num_ctas`/`ct.occupancy` as functions (they are decorator arguments). `ct.tune` and the separate `cuda.tile_experimental` package exist in the environment but are tuning APIs and are not permitted in this task setting.
