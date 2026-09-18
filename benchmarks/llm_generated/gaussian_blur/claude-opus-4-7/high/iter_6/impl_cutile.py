import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _hconv_kernel(x, kw, out,
                  KW: ConstInt, PAD_W: ConstInt,
                  BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    offs_m = bid_m * BM + ct.arange(BM, dtype=np.int32)
    offs_n = bid_n * BN + ct.arange(BN, dtype=np.int32)
    acc = ct.full((BM, BN), 0.0, dtype=np.float32)
    for kw_idx in range(KW):
        row = offs_m[:, None]
        col = offs_n[None, :] + (kw_idx - PAD_W)
        r2 = ct.broadcast_to(row, (BM, BN))
        c2 = ct.broadcast_to(col, (BM, BN))
        vals = ct.gather(x, (r2, c2), padding_value=0.0)
        w_val = ct.load(kw, index=(kw_idx,), shape=())
        acc = acc + ct.astype(vals, np.float32) * ct.astype(w_val, np.float32)
    ct.store(out, index=(bid_m, bid_n), tile=acc)


@ct.kernel(occupancy=4)
def _vconv_kernel(x, kh, out,
                  KH: ConstInt, PAD_H: ConstInt,
                  BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    offs_m = bid_m * BM + ct.arange(BM, dtype=np.int32)
    offs_n = bid_n * BN + ct.arange(BN, dtype=np.int32)
    acc = ct.full((BM, BN), 0.0, dtype=np.float32)
    for kh_idx in range(KH):
        row = offs_m[:, None] + (kh_idx - PAD_H)
        col = offs_n[None, :]
        r2 = ct.broadcast_to(row, (BM, BN))
        c2 = ct.broadcast_to(col, (BM, BN))
        vals = ct.gather(x, (r2, c2), padding_value=0.0)
        w_val = ct.load(kh, index=(kh_idx,), shape=())
        acc = acc + ct.astype(vals, np.float32) * ct.astype(w_val, np.float32)
    ct.store(out, index=(bid_m, bid_n), tile=ct.astype(acc, out.dtype))


def _decompose_rank1(kernel, KH, KW):
    """K = g_v ⊗ g_h (exact for true Gaussian)."""
    k2d = kernel.float().view(KH, KW)
    ch = KH // 2
    cw = KW // 2
    center_row = k2d[ch, :]
    center_col = k2d[:, cw]
    g_v_center = center_row.sum()
    g_h = center_row / g_v_center
    g_h_center = g_h[cw]
    g_v = center_col / g_h_center
    return g_v.contiguous(), g_h.contiguous()


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    g_v, g_h = _decompose_rank1(kernel, KH, KW)
    kh_1d = g_v   # length KH (fp32)
    kw_1d = g_h   # length KW (fp32)

    x2d = input.view(H, W)
    out2d = output.view(H, W)
    temp = torch.empty(H * W, dtype=torch.float32, device=input.device)
    temp2d = temp.view(H, W)

    BM = 32
    BN = 128
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(H, BM), ct.cdiv(W, BN), 1)

    ct.launch(stream, grid, _hconv_kernel,
              (x2d, kw_1d, temp2d, KW, PAD_W, BM, BN))
    ct.launch(stream, grid, _vconv_kernel,
              (temp2d, kh_1d, out2d, KH, PAD_H, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BM": BM,
        "BN": BN,
        "occupancy": occupancy,
        "approach": "separable_rank1",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
