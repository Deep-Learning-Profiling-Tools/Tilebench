# Train Example: cuTile Row Reduction (mean_reduction)

**Operator:** `mean_reduction`
**Backend:** cuTile
**Pattern:** Row-wise sum reduction using `ct.sum`, one block per row

```python
import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

_last_config: dict | None = None


@ct.kernel
def _mean_kernel(X, Out, N_COLS: ConstInt, BLOCK_N: ConstInt):
    bid = ct.bid(0)   # one block per row

    # Load the full row; zero-pad for boundary rows.
    row = ct.load(X, index=(bid, 0), shape=(1, BLOCK_N),
                  padding_mode=ct.PaddingMode.ZERO)

    # Sum along column dimension (dim=1) then divide by true width.
    row_sum  = ct.sum(row, dim=1)          # shape: (1,)
    row_mean = row_sum / N_COLS

    ct.store(Out, index=(bid,), tile=row_mean)


def run(x: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_config
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    out    = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream().cuda_stream

    # BLOCK_N must cover all columns (round up to next power of 2 recommended).
    BLOCK_N = 1
    while BLOCK_N < n_cols:
        BLOCK_N *= 2

    grid = (n_rows,)
    ct.launch(stream, grid, _mean_kernel, (x, out, n_cols, BLOCK_N))
    _last_config = None
    return out


def get_last_config() -> dict | None:
    return _last_config
```
