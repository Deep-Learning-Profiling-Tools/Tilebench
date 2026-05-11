# Train Example: cuTile Matrix Transpose (matrix_transpose)

**Operator:** `matrix_transpose`
**Backend:** cuTile
**Pattern:** 2-D grid with tile-space 2-D indexing

```python
import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

_last_config: dict | None = None


@ct.kernel
def _transpose_kernel(X, Out, M: ConstInt, N: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)   # tile row in X
    bid_n = ct.bid(1)   # tile col in X

    # Load tile of shape (TILE_M, TILE_N) from X at (bid_m, bid_n)
    tile = ct.load(X, index=(bid_m, bid_n), shape=(TILE_M, TILE_N),
                   padding_mode=ct.PaddingMode.ZERO)

    # Store transposed tile to Out at (bid_n, bid_m)
    # Out has shape (N, M) so row index = bid_n, col index = bid_m
    ct.store(Out, index=(bid_n, bid_m), tile=ct.transpose(tile))


def run(x: torch.Tensor,
        block_size: int = 32, autotune: bool = False) -> torch.Tensor:
    global _last_config
    m, n    = x.shape
    out     = torch.empty((n, m), dtype=x.dtype, device=x.device)
    TILE    = max(1, block_size)
    stream  = torch.cuda.current_stream().cuda_stream

    grid = (math.ceil(m / TILE), math.ceil(n / TILE))
    ct.launch(stream, grid, _transpose_kernel, (x, out, m, n, TILE, TILE))
    _last_config = None
    return out


def get_last_config() -> dict | None:
    return _last_config
```
