import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

_OCCUPANCY = 2
_BLOCK_M = 128
_BLOCK_N = 256
_TILE = 128  # matches config.yaml TILE_SIZE


@ct.kernel(occupancy=_OCCUPANCY)
def _wdq_kernel(X, S, Y,
                BLOCK_M: ConstInt, BLOCK_N: ConstInt, TILE: ConstInt,
                SM: ConstInt, SN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x = ct.load(X, index=(bid_m, bid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO)

    # scale tile of shape (SM, SN) — one scalar per TILE x TILE sub-block
    s = ct.load(S, index=(bid_m, bid_n),
                shape=(SM, SN),
                padding_mode=ct.PaddingMode.ZERO)

    # broadcast (SM, SN) -> (BLOCK_M, BLOCK_N) via (SM, 1, SN, 1) -> (SM, TILE, SN, TILE)
    s4 = ct.reshape(s, (SM, 1, SN, 1))
    s4b = ct.broadcast_to(s4, (SM, TILE, SN, TILE))
    s_full = ct.reshape(s4b, (BLOCK_M, BLOCK_N))

    y = ct.astype(ct.astype(x, np.float32) * ct.astype(s_full, np.float32), X.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=y)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    assert TILE_SIZE == _TILE, f"This impl expects TILE_SIZE={_TILE}, got {TILE_SIZE}"
    SM = _BLOCK_M // _TILE  # 1
    SN = _BLOCK_N // _TILE  # 2

    grid = (ct.cdiv(M, _BLOCK_M), ct.cdiv(N, _BLOCK_N), 1)
    ct.launch(stream, grid, _wdq_kernel,
              (X, S, Y, _BLOCK_M, _BLOCK_N, _TILE, SM, SN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": _BLOCK_M, "BLOCK_N": _BLOCK_N,
        "occupancy": _OCCUPANCY,
    })
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
