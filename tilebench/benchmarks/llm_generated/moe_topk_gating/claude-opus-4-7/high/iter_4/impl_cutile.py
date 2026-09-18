import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _argmax_combine(v1, i1, v2, i2):
    win1 = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return (ct.where(win1, v1, v2), ct.where(win1, i1, i2))


@ct.kernel(occupancy=4)
def _moe_topk_kernel(logits, weights, indices,
                     M: ConstInt, E: ConstInt, K: ConstInt,
                     BLOCK_M: ConstInt, BLOCK_E: ConstInt):
    pid = ct.bid(0)
    x2d = ct.load(logits, index=(pid, 0), shape=(BLOCK_M, BLOCK_E),
                  padding_mode=ct.PaddingMode.NEG_INF)
    x = ct.astype(x2d, np.float32)

    k_arange = ct.arange(K, dtype=np.int32)[None, :]
    col_offs = ct.arange(BLOCK_E, dtype=np.int32)[None, :]
    top_vals = ct.full((BLOCK_M, K), -float('inf'), dtype=np.float32)
    top_idxs = ct.full((BLOCK_M, K), 0, dtype=np.int32)

    for i in range(K):
        v, idx = ct.reduce(
            (x, ct.broadcast_to(ct.arange(BLOCK_E, dtype=np.int32)[None, :], (BLOCK_M, BLOCK_E))),
            axis=1,
            func=_argmax_combine,
            identity=(-float('inf'), BLOCK_E + 1),
            keepdims=False,
        )
        pos = K - 1 - i
        sel = k_arange == pos
        top_vals = ct.where(sel, v[:, None], top_vals)
        top_idxs = ct.where(sel, idx[:, None], top_idxs)
        x = ct.where(col_offs == idx[:, None], -float('inf'), x)

    m_val = ct.max(top_vals, axis=1, keepdims=True)
    ex = ct.exp(top_vals - m_val)
    sm = ct.sum(ex, axis=1, keepdims=True)
    w = ex / sm

    w_out = ct.astype(w, logits.dtype)
    ct.store(weights, index=(pid, 0), tile=w_out)
    ct.store(indices, index=(pid, 0), tile=top_idxs)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = 1
    while BLOCK_E < E:
        BLOCK_E *= 2
    BLOCK_M = 16
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = ((M + BLOCK_M - 1) // BLOCK_M, 1, 1)
    ct.launch(stream, grid, _moe_topk_kernel,
              (logits, weights, indices, M, E, k, BLOCK_M, BLOCK_E))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_E": BLOCK_E,
        "K": k,
        "occupancy": occupancy,
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
