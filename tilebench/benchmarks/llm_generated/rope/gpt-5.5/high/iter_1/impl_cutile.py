import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _rope_kernel(q, cos, sin, output,
                 HEAD_DIM: ConstInt, BLOCK_H: ConstInt, BLOCK_D: ConstInt):
    s = ct.bid(0)
    hb = ct.bid(1)
    b = ct.bid(2)

    half = HEAD_DIM // 2

    q_first = q.slice(3, 0, half)
    q_second = q.slice(3, half, HEAD_DIM)
    out_first = output.slice(3, 0, half)
    out_second = output.slice(3, half, HEAD_DIM)

    q1 = ct.load(
        q_first,
        index=(b, s, hb, 0),
        shape=(1, 1, BLOCK_H, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    q2 = ct.load(
        q_second,
        index=(b, s, hb, 0),
        shape=(1, 1, BLOCK_H, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    cos_tile = ct.load(
        cos,
        index=(s, 0),
        shape=(1, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    sin_tile = ct.load(
        sin,
        index=(s, 0),
        shape=(1, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    cos4 = ct.reshape(cos_tile, (1, 1, 1, BLOCK_D))
    sin4 = ct.reshape(sin_tile, (1, 1, 1, BLOCK_D))

    q1f = ct.astype(q1, np.float32)
    q2f = ct.astype(q2, np.float32)
    cosf = ct.astype(cos4, np.float32)
    sinf = ct.astype(sin4, np.float32)

    p1 = ct.astype(q1f * cosf, output.dtype)
    p2 = ct.astype(q2f * sinf, output.dtype)
    out1 = ct.astype(p1, np.float32) - ct.astype(p2, np.float32)
    ct.store(
        out_first,
        index=(b, s, hb, 0),
        tile=ct.astype(out1, output.dtype),
        latency=1,
        allow_tma=False,
    )

    p3 = ct.astype(q2f * cosf, output.dtype)
    p4 = ct.astype(q1f * sinf, output.dtype)
    out2 = ct.astype(p3, np.float32) + ct.astype(p4, np.float32)
    ct.store(
        out_second,
        index=(b, s, hb, 0),
        tile=ct.astype(out2, output.dtype),
        latency=1,
        allow_tma=False,
    )


def run(q, cos, sin):
    output = torch.empty_like(q)

    batch_size = q.shape[0]
    seq_len = q.shape[1]
    n_heads = q.shape[2]
    head_dim = q.shape[3]
    half = head_dim // 2

    BLOCK_H = 32
    BLOCK_D = 1 << (half - 1).bit_length()
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (seq_len, ct.cdiv(n_heads, BLOCK_H), batch_size)
    ct.launch(stream, grid, _rope_kernel, (q, cos, sin, output, head_dim, BLOCK_H, BLOCK_D))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_H": BLOCK_H,
        "BLOCK_D": BLOCK_D,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
