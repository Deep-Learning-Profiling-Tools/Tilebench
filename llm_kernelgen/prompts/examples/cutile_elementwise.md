# Train Example: cuTile Elementwise (vector_add)

**Operator:** `vector_add`
**Backend:** cuTile
**Pattern:** 1-D block grid, tile load/store with zero-padding

```python
import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

_last_config: dict | None = None


@ct.kernel
def _vector_add_kernel(X, Y, Out, N: ConstInt, TILE: ConstInt):
    bid  = ct.bid(0)
    x    = ct.load(X,   index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    y    = ct.load(Y,   index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    out  = x + y
    ct.store(Out, index=(bid,), tile=out)


def run(x: torch.Tensor, y: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_config
    n      = x.numel()
    out    = torch.empty_like(x)
    TILE   = block_size
    grid   = (math.ceil(n / TILE),)
    stream = torch.cuda.current_stream().cuda_stream

    ct.launch(stream, grid, _vector_add_kernel, (x, y, out, n, TILE))
    _last_config = None
    return out


def get_last_config() -> dict | None:
    return _last_config
```
