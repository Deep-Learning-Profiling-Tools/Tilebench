# Train Example: Triton Tiled Matrix Multiplication (matrix_transpose)

**Operator:** `matrix_transpose`
**Backend:** Triton
**Pattern:** 2-D tiled grid, shared-memory coalesced access

This example shows a 2-D block grid launch and proper strided access.

```python
import torch
import triton
import triton.language as tl

_last_config: dict | None = None


@triton.jit
def _transpose_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # row indices in X
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # col indices in X

    mask = (rm[:, None] < m) & (rn[None, :] < n)

    # Load tile from X[rm, rn]
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    tile   = tl.load(x_ptrs, mask=mask, other=0.0)

    # Store to Out[rn, rm]  (transposed)
    out_ptrs = out_ptr + rn[None, :] * stride_om + rm[:, None] * stride_on
    tl.store(out_ptrs, tile.T, mask=mask.T)


def run(x: torch.Tensor,
        block_size: int = 32, autotune: bool = False) -> torch.Tensor:
    global _last_config
    m, n  = x.shape
    out   = torch.empty((n, m), dtype=x.dtype, device=x.device)
    BLOCK = block_size if block_size > 0 else 32

    grid = (triton.cdiv(m, BLOCK), triton.cdiv(n, BLOCK))
    _transpose_kernel[grid](
        x, out, m, n,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
    )
    _last_config = None
    return out


def get_last_config() -> dict | None:
    return _last_config
```
