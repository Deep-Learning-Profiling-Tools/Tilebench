import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _wdq_kernel(X, S, Y, M, N, TILE_SIZE,
                BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x = ct.load(X, index=(bid_m, bid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO)

    offs_m = bid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = bid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    s_row = (offs_m // TILE_SIZE)[:, None]
    s_col = (offs_n // TILE_SIZE)[None, :]

    s = ct.gather(S, (s_row, s_col), padding_value=0)

    y = ct.astype(ct.astype(x, np.float32) * ct.astype(s, np.float32), X.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=y)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    BLOCK_M = 128
    BLOCK_N = 128
    occupancy = 4

    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)
    kernel = _wdq_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (X, S, Y, M, N, TILE_SIZE, BLOCK_M, BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "occupancy": occupancy,
    })
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
