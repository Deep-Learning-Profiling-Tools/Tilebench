Looking at the trajectory:
- fp32 hits 96% (memory-bound), but fp16/bf16 only hit 37%
- This suggests fp16/bf16 cases are compute-bound on the `exp` function (memory traffic is halved, so compute matters more)
- The best verify-clean cfg was tile=4096, occupancy=4 at 65.3%

I'll revert to that config but switch `exp(-x)` to the faster `exp2(-x * log2(e))` with `flush_to_zero=True`. `ct.exp2` is hardware-accelerated, which should significantly speed up the fp16/bf16 compute-bound cases.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

# log2(e) — multiplier to convert exp(x) to exp2(x * LOG2E)
_LOG2E = 1.4426950408889634


@ct.kernel(occupancy=4)
def _sigmoid_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x_tile, np.float32)
    # sigmoid(x) = 1 / (1 + exp(-x))
    #            = 1 / (1 + exp2(-x * log2(e)))
    # ct.exp2 maps to hardware ex2.approx — much faster than exp on B200.
    neg_scaled = xf * (-_LOG2E)
    e = ct.exp2(neg_scaled, flush_to_zero=True)
    y = 1.0 / (1.0 + e)
    y_out = ct.astype(y, x.dtype)
    ct.store(output, index=(bid,), tile=y_out)


def run(X: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(X)
    n_elements = X.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _sigmoid_kernel, (X, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy,
                      "math": "exp2_approx"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: revert to the best verify-clean config (tile=4096, occupancy=4) and replace the slow `ct.exp(-x)` with `ct.exp2(-x * log2(e), flush_to_zero=True)` to use Blackwell's hardware-accelerated `ex2.approx` instruction — this directly attacks the compute-bound bottleneck in fp16/bf16 where memory traffic is halved but exp cost is constant.
