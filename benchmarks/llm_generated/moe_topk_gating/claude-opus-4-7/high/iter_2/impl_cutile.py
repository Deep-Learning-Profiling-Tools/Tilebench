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
                     E: ConstInt, K: ConstInt,
                     BLOCK_M: ConstInt, BLOCK_E: ConstInt):
    pid = ct.bid(0)
    x2d = ct.load(logits, index=(pid, 0), shape=(BLOCK_M, BLOCK_E),
                  padding_mode=ct.PaddingMode.NEG_INF)
    x = ct.astype(x2d, np.float32)

    offs_1d = ct.arange(BLOCK_E, dtype=np.int32)
    offs_row = ct.reshape(offs_1d, (1, BLOCK_E))
    offs_2d = ct.broadcast_to(offs_row, (BLOCK_M, BLOCK_E))

    k_arange = ct.arange(K, dtype=np.int32)
    k_row = ct.reshape(k_arange, (1, K))

    top_vals = ct.full((BLOCK_M, K), -float('inf'), dtype=np.float32)
    top_idxs = ct.full((BLOCK_M, K), 0, dtype=np.int32)

    for i in range(K):
        v, idx = ct.reduce(
            (x, offs_2d), axis=1,
            func=_argmax_combine,
            identity=(-float('inf'), BLOCK_E + 1),
        )
        # v, idx: (BLOCK_M,) -> (BLOCK_M, 1)
        v_col = ct.reshape(v, (BLOCK_M, 1))
        idx_col = ct.reshape(idx, (BLOCK_M, 1))

        pos = K - 1 - i
        sel_k = (k_row == pos)  # (1, K)
        sel_k_full = ct.broadcast_to(sel_k, (BLOCK_M, K))
        v_full = ct.broadcast_to(v_col, (BLOCK_M, K))
        idx_full_k = ct.broadcast_to(idx_col, (BLOCK_M, K))
        top_vals = ct.where(sel_k_full, v_full, top_vals)
        top_idxs = ct.where(sel_k_full, idx_full_k, top_idxs)

        # Mask out chosen index in x
        idx_full_e = ct.broadcast_to(idx_col, (BLOCK_M, BLOCK_E))
        chosen = (offs_2d == idx_full_e)
        x = ct.where(chosen, -float('inf'), x)

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
    # Ensure grid covers M (OOB rows handled via padding/silent-drop)
    num_blocks = (M + BLOCK_M - 1) // BLOCK_M

    stream = torch.cuda.current_stream()
    grid = (num_blocks, 1, 1)
    ct.launch(stream, grid, _moe_topk_kernel,
              (logits, weights, indices, E, k, BLOCK_M, BLOCK_E))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_E": BLOCK_E,
        "K": k,
        "occupancy": 4,
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
