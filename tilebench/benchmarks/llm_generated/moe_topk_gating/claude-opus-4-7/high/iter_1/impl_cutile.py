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
                     E: ConstInt, K: ConstInt, BLOCK_E: ConstInt):
    row = ct.bid(0)
    x2d = ct.load(logits, index=(row, 0), shape=(1, BLOCK_E),
                  padding_mode=ct.PaddingMode.NEG_INF)
    x = ct.astype(x2d, np.float32)
    x = ct.reshape(x, (BLOCK_E,))

    k_arange = ct.arange(K, dtype=np.int32)
    offs = ct.arange(BLOCK_E, dtype=np.int32)
    top_vals = ct.full((K,), -float('inf'), dtype=np.float32)
    top_idxs = ct.full((K,), 0, dtype=np.int32)

    for i in range(K):
        v, idx = ct.reduce(
            (x, offs), axis=0,
            func=_argmax_combine,
            identity=(-float('inf'), BLOCK_E + 1),
        )
        pos = K - 1 - i
        top_vals = ct.where(k_arange == pos, v, top_vals)
        top_idxs = ct.where(k_arange == pos, idx, top_idxs)
        x = ct.where(offs == idx, -float('inf'), x)

    m_val = ct.max(top_vals)
    ex = ct.exp(top_vals - m_val)
    sm = ct.sum(ex)
    w = ex / sm

    w_out = ct.astype(w, logits.dtype)
    ct.store(weights, index=(row, 0), tile=ct.reshape(w_out, (1, K)))
    ct.store(indices, index=(row, 0), tile=ct.reshape(top_idxs, (1, K)))


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = 1
    while BLOCK_E < E:
        BLOCK_E *= 2

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    ct.launch(stream, grid, _moe_topk_kernel,
              (logits, weights, indices, E, k, BLOCK_E))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_E": BLOCK_E,
        "K": k,
        "occupancy": 4,
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
