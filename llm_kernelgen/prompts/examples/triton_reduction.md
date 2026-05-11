# Train Example: Triton Row Reduction (mean_reduction)

**Operator:** `mean_reduction`
**Backend:** Triton
**Pattern:** Row-wise reduction with online accumulator

This example shows the row-reduction pattern: one program per row,
scanning across columns with a tile-per-iteration loop.

```python
import torch
import triton
import triton.language as tl

_last_config: dict | None = None


@triton.jit
def _mean_reduction_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    acc   = tl.zeros((BLOCK_N,), dtype=tl.float32)
    total = tl.zeros((1,),       dtype=tl.float32)

    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x    = tl.load(x_ptr + row * n_cols + cols,
                       mask=mask, other=0.0).to(tl.float32)
        acc  = tl.where(mask, x, tl.zeros_like(x))
        total += tl.sum(acc, axis=0)

    mean = total / n_cols
    tl.store(out_ptr + row, mean)


def run(x: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_config
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    _mean_reduction_kernel[grid](
        x, out, n_rows, n_cols, BLOCK_N=block_size
    )
    _last_config = None
    return out


def get_last_config() -> dict | None:
    return _last_config
```
