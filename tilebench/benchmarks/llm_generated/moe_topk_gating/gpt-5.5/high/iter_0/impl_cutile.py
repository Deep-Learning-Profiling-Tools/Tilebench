import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _argmax_first_combine(a_val, a_idx, b_val, b_idx):
    take_a = (a_val > b_val) | ((a_val == b_val) & (a_idx <= b_idx))
    return ct.where(take_a, a_val, b_val), ct.where(take_a, a_idx, b_idx)


@ct.kernel
def _moe_topk2_kernel(logits, weights, indices, ROWS: ConstInt, TILE_E: ConstInt):
    bid = ct.bid(0)

    vals = ct.astype(
        ct.load(
            logits,
            index=(bid, 0),
            shape=(ROWS, TILE_E),
            padding_mode=ct.PaddingMode.NEG_INF,
            allow_tma=False,
            latency=1,
        ),
        np.float32,
    )

    offs = ct.arange(TILE_E, dtype=np.int32)[None, :]
    idxs = ct.broadcast_to(offs, (ROWS, TILE_E))

    # First max, first-occurrence tie break. Stored in output column 1.
    val1, idx1 = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_first_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    vals2 = ct.where(idxs == idx1, -np.inf, vals)

    # Second max. Stored in output column 0.
    val0, idx0 = ct.reduce(
        (vals2, idxs),
        axis=1,
        func=_argmax_first_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    e = ct.exp(val0 - val1)
    den = e + 1.0
    w0 = e / den
    w1 = 1.0 / den

    ct.store(weights, index=(bid, 0), tile=ct.astype(w0, logits.dtype), allow_tma=False)
    ct.store(weights, index=(bid, 1), tile=ct.astype(w1, logits.dtype), allow_tma=False)

    ct.store(indices, index=(bid, 0), tile=ct.astype(idx0, np.int32), allow_tma=False)
    ct.store(indices, index=(bid, 1), tile=ct.astype(idx1, np.int32), allow_tma=False)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    ROWS = 2
    TILE_E = 256
    occupancy = 8

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, ROWS), 1, 1)
    kernel = _moe_topk2_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (logits, topk_weights, topk_indices, ROWS, TILE_E))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ROWS": ROWS,
            "TILE_E": TILE_E,
            "occupancy": occupancy,
            "allow_tma": False,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
