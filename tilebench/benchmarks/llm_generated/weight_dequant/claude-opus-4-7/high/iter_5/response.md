Looking at the trajectory: occ=2→54.6%, occ=4→61.1%, occ=8→78.7%. Clear monotonic improvement from occupancy. The fp16/bf16 cases at 74% suggest the kernel is latency-limited (per-block work is small: one scalar load + one 128×128 tile load + one 128×128 store). Pushing occupancy higher to keep more CTAs in flight should continue to help hide HBM latency.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

# Iter trajectory on (BLOCK_M=BLOCK_N=128, TILE_SIZE=128):
#   occ=2  -> 54.6%
#   occ=4  -> 61.1%
#   occ=8  -> 78.7%
# Per-block work is tiny: 1 scalar load + 1 TMA-tile load + 1 TMA-tile store.
# Clearly latency-bound (more CTAs in flight => more bandwidth). Push occ=16.
_OCCUPANCY = 16
_BLOCK_M = 128
_BLOCK_N = 128


@ct.kernel(occupancy=_OCCUPANCY)
def _wdq_kernel(X, S, Y, BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # BLOCK_M == BLOCK_N == TILE_SIZE so exactly one scale per block.
    s = ct.load(S, index=(bid_m, bid_n), shape=())

    x = ct.load(X, index=(bid_m, bid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO)

    y = ct.astype(ct.astype(x, np.float32) * ct.astype(s, np.float32), X.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=y)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    assert TILE_SIZE == _BLOCK_M, f"This impl expects TILE_SIZE={_BLOCK_M}, got {TILE_SIZE}"

    grid = (ct.cdiv(M, _BLOCK_M), ct.cdiv(N, _BLOCK_N), 1)
    ct.launch(stream, grid, _wdq_kernel, (X, S, Y, _BLOCK_M, _BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": _BLOCK_M,
        "BLOCK_N": _BLOCK_N,
        "occupancy": _OCCUPANCY,
    })
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Single change from iter 4: bump `occupancy` from 8 → 16, keeping `BLOCK_M=BLOCK_N=128` so we still match `TILE_SIZE` (one scale per block). The monotone improvement with occupancy says we're latency-limited on tiny per-block work — more in-flight CTAs per SM should continue closing the bandwidth gap on fp16/bf16.
