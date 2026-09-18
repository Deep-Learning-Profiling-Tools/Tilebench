import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _split_kernel(
    mid_o, mid_o_lse, b_seqlen,
    partial_o, partial_lse,
    block_seq: int,
    HEAD_DIM: ConstInt, BLOCK_K: ConstInt,
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)

    seqlen = ct.load(b_seqlen, index=(b,), shape=())
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    offs_k = ct.arange(BLOCK_K, dtype=np.int32) + s * BLOCK_K
    mask_k = offs_k < valid_blocks

    lse = ct.load(mid_o_lse, index=(b, h, s), shape=(1, 1, BLOCK_K),
                  padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_K,))
    lse = ct.astype(lse, np.float32)
    lse = ct.where(mask_k, lse, ct.full((BLOCK_K,), -np.inf, dtype=np.float32))

    m = ct.max(lse, axis=0, keepdims=True)  # (1,)
    # Clamp -inf to -1e30 so exp(lse - safe_m) is well-defined.
    neg_clamp = ct.full((1,), -1.0e30, dtype=np.float32)
    safe_m = ct.maximum(m, neg_clamp)

    w = ct.exp(lse - safe_m)
    w = ct.where(mask_k, w, ct.full((BLOCK_K,), 0.0, dtype=np.float32))
    l = ct.sum(w, axis=0, keepdims=True)  # (1,)

    vals = ct.load(mid_o, index=(b, h, s, 0),
                   shape=(1, 1, BLOCK_K, HEAD_DIM),
                   padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_K, HEAD_DIM))
    vals = ct.astype(vals, np.float32)

    acc = ct.sum(vals * w[:, None], axis=0)  # (HEAD_DIM,)

    is_valid = l > ct.full((1,), 0.0, dtype=np.float32)
    one = ct.full((1,), 1.0, dtype=np.float32)
    zero = ct.full((1,), 0.0, dtype=np.float32)
    # avoid div-by-zero
    l_safe = ct.where(is_valid, l, one)
    inv_l = ct.where(is_valid, one / l_safe, zero)
    partial_out = acc * inv_l  # broadcast (1,) -> (HEAD_DIM,)

    neg_inf_t = ct.full((1,), -np.inf, dtype=np.float32)
    local_lse = ct.where(is_valid, ct.log(l_safe) + m, neg_inf_t)

    partial_out_tile = partial_out.reshape((1, 1, 1, HEAD_DIM))
    ct.store(partial_o, index=(b, h, s, 0), tile=partial_out_tile)

    local_lse_tile = local_lse.reshape((1, 1, 1))
    ct.store(partial_lse, index=(b, h, s), tile=local_lse_tile)


@ct.kernel(occupancy=4)
def _merge_kernel(
    partial_o, partial_lse, output,
    HEAD_DIM: ConstInt, BLOCK_S: ConstInt,
):
    b = ct.bid(0)
    h = ct.bid(1)

    lses = ct.load(partial_lse, index=(b, h, 0), shape=(1, 1, BLOCK_S),
                   padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_S,))
    lses = ct.astype(lses, np.float32)

    global_m = ct.max(lses, axis=0, keepdims=True)
    neg_clamp = ct.full((1,), -1.0e30, dtype=np.float32)
    safe_m = ct.maximum(global_m, neg_clamp)

    w = ct.exp(lses - safe_m)
    l = ct.sum(w, axis=0, keepdims=True)

    accs = ct.load(partial_o, index=(b, h, 0, 0),
                   shape=(1, 1, BLOCK_S, HEAD_DIM),
                   padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_S, HEAD_DIM))
    accs = ct.astype(accs, np.float32)

    numer = ct.sum(accs * w[:, None], axis=0)  # (HEAD_DIM,)
    out = numer / l  # broadcast

    out_tile = ct.astype(out, output.dtype).reshape((1, 1, HEAD_DIM))
    ct.store(output, index=(b, h, 0), tile=out_tile)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    if isinstance(block_seq, torch.Tensor):
        block_seq = int(block_seq.item())
    else:
        block_seq = int(block_seq)

    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim),
                         device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    HEAD_DIM_C = _next_pow2(head_dim)
    K_SPLITS = (num_blocks + BLOCK_K - 1) // BLOCK_K
    BLOCK_S = _next_pow2(K_SPLITS) if K_SPLITS > 1 else 2
    if BLOCK_S < 2:
        BLOCK_S = 2

    partial_o = torch.zeros((batch, heads, BLOCK_S, head_dim),
                            device=mid_o.device, dtype=torch.float32)
    partial_lse = torch.full((batch, heads, BLOCK_S), -float('inf'),
                             device=mid_o.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    grid1 = (batch, heads, K_SPLITS)
    ct.launch(stream, grid1, _split_kernel,
              (mid_o, mid_o_lse, b_seqlen, partial_o, partial_lse,
               block_seq, HEAD_DIM_C, BLOCK_K))

    grid2 = (batch, heads, 1)
    ct.launch(stream, grid2, _merge_kernel,
              (partial_o, partial_lse, output, HEAD_DIM_C, BLOCK_S))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_S": BLOCK_S, "K_SPLITS": K_SPLITS,
        "HEAD_DIM": HEAD_DIM_C, "occupancy": 4,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
