# Triton DSL Minimal Reference

This is a concise cookbook covering the Triton patterns you need for TileBench kernels.

---

## 1. Program IDs and grid

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(ptr_x, ptr_out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)          # 1-D grid
    # 2-D grid: pid_m = tl.program_id(0); pid_n = tl.program_id(1)
```

Launch grid:
```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
my_kernel[grid](x, out, n, BLOCK=1024)
```

---

## 2. Vectorised load / store with masking

```python
@triton.jit
def elementwise_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # [BLOCK]
    mask = offs < n
    x    = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)
```

---

## 3. Row-wise reduction (e.g. softmax, rmsnorm)

```python
@triton.jit
def row_reduce_kernel(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))

    row_max = tl.max(x, axis=0)
    x_shift = x - row_max
    x_exp   = tl.exp(x_shift)
    x_sum   = tl.sum(x_exp, axis=0)
    out     = x_exp / x_sum

    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)
```

---

## 4. Tiled matrix multiplication

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, triton.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr + rm[:, None] * stride_am + (k * BLOCK_K + rk[None, :]) * stride_ak,
                    mask=(rm[:, None] < M) & ((k * BLOCK_K + rk[None, :]) < K), other=0.0)
        b = tl.load(b_ptr + (k * BLOCK_K + rk[:, None]) * stride_bk + rn[None, :] * stride_bn,
                    mask=((k * BLOCK_K + rk[:, None]) < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(tl.float16), mask=mask_c)
```

---

## 5. Autotune pattern

```python
_last_config: dict | None = None

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}),
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["n"],
)
@triton.jit
def _autotuned_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    ...

def run(x, block_size: int = 1024, autotune: bool = False, **kwargs):
    global _last_config
    n = x.numel()
    out = torch.empty_like(x)
    if autotune:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        _autotuned_kernel[grid](x, out, n)
        _last_config = {k: v for k, v in _autotuned_kernel.best_config.kwargs.items()}
    else:
        grid = (triton.cdiv(n, block_size),)
        _fixed_kernel[grid](x, out, n, BLOCK=block_size)
        _last_config = None
    return out

def get_last_config() -> dict | None:
    return _last_config
```

---

## 6. Useful intrinsics

| API | Description |
|-----|-------------|
| `tl.arange(0, N)` | Integer range `[0, N)`, N must be a power of 2 |
| `tl.load(ptr, mask, other)` | Masked load with fill value |
| `tl.store(ptr, val, mask)` | Masked store |
| `tl.sum(x, axis)` | Reduction sum along axis |
| `tl.max(x, axis)` | Reduction max along axis |
| `tl.exp(x)` | Element-wise exp |
| `tl.sqrt(x)` | Element-wise sqrt |
| `tl.dot(a, b)` | Matrix multiply accumulate |
| `tl.zeros(shape, dtype)` | Zero-initialised tensor |
| `tl.cast(x, dtype)` | Type cast |
| `tl.constexpr` | Compile-time constant |
| `triton.cdiv(a, b)` | Ceiling integer division |

---

## 7. Boundary handling checklist

- Always compute `mask = offsets < n` before `tl.load`/`tl.store`.
- Use `other=0.0` for neutral fills in sum reductions.
- Use `other=-float("inf")` for neutral fills in max reductions.
- Block sizes do **not** need to be powers of two for `tl.arange` when
  combined with masking — but the constexpr dimension of arrays must be
  a power of two.
