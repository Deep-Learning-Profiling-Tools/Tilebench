import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _top2_combine(a_v1, a_i1, a_v2, a_i2, b_v1, b_i1, b_v2, b_i2):
    take_a1 = (a_v1 > b_v1) | ((a_v1 == b_v1) & (a_i1 <= b_i1))

    out_v1 = ct.where(take_a1, a_v1, b_v1)
    out_i1 = ct.where(take_a1, a_i1, b_i1)

    cand_a_v = ct.where(take_a1, a_v2, a_v1)
    cand_a_i = ct.where(take_a1, a_i2, a_i1)
    cand_b_v = ct.where(take_a1, b_v1, b_v2)
    cand_b_i = ct.where(take_a1, b_i1, b_i2)

    take_a2 = (cand_a_v > cand_b_v) | ((cand_a_v == cand_b_v) & (cand_a_i <= cand_b_i))
    out_v2 = ct.where(take_a2, cand_a_v, cand_b_v)
    out_i2 = ct.where(take_a2, cand_a_i, cand_b_i)

    return out_v1, out_i1, out_v2, out_i2


@ct.kernel
def _moe_topk2_kernel(logits, weights, indices, ROWS: ConstInt, TILE_E: ConstInt):
    bid = ct.bid(0)

    vals = ct.load(
        logits,
        index=(bid, 0),
        shape=(ROWS, TILE_E),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    offs = ct.arange(TILE_E, dtype=np.int32)[None, :]
    idxs = ct.broadcast_to(offs, (ROWS, TILE_E))
    neg_vals = ct.full((ROWS, TILE_E), -np.inf, dtype=logits.dtype)
    big_idxs = ct.full((ROWS, TILE_E), 2147483647, dtype=np.int32)

    val1, idx1, val0, idx0 = ct.reduce(
        (vals, idxs, neg_vals, big_idxs),
        axis=1,
        func=_top2_combine,
        identity=(-np.inf, 2147483647, -np.inf, 2147483647),
        keepdims=True,
    )

    v0 = ct.astype(val0, np.float32)
    v1 = ct.astype(val1, np.float32)
    e = ct.exp(v0 - v1)
    den = e + 1.0
    w0 = e / den
    w1 = 1.0 / den

    cols = ct.arange(2, dtype=np.int32)[None, :]
    w_out = ct.where(cols == 0, w0, w1)
    i_out = ct.where(cols == 0, idx0, idx1)

    ct.store(weights, index=(bid, 0), tile=ct.astype(w_out, logits.dtype), allow_tma=False)
    ct.store(indices, index=(bid, 0), tile=ct.astype(i_out, np.int32), allow_tma=False)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    ROWS = 8
    TILE_E = 128
    occupancy = 4

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
            "one_pass_top2": True,
            "combined_store": True,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
