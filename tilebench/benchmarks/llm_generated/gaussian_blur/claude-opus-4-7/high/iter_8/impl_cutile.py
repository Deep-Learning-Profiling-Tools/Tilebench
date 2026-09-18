import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _hconv_kernel(x, k, out,
                  KW: ConstInt, PAD_W: ConstInt,
                  BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    offs_m = bid_m * BM + ct.arange(BM, dtype=np.int32)
    offs_n = bid_n * BN + ct.arange(BN, dtype=np.int32)

    acc = ct.full((BM, BN), 0.0, dtype=np.float32)

    for kw in range(KW):
        row = offs_m[:, None]
        col = (offs_n + (kw - PAD_W))[None, :]
        row2 = ct.broadcast_to(row, (BM, BN))
        col2 = ct.broadcast_to(col, (BM, BN))
        vals = ct.gather(x, (row2, col2), padding_value=0.0)
        w_val = ct.load(k, index=(kw,), shape=())
        acc = acc + ct.astype(vals, np.float32) * ct.astype(w_val, np.float32)

    out_tile = ct.astype(acc, x.dtype)
    ct.store(out, index=(bid_m, bid_n), tile=out_tile)


@ct.kernel(occupancy=4)
def _vconv_kernel(x, k, out,
                  KH: ConstInt, PAD_H: ConstInt,
                  BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    offs_m = bid_m * BM + ct.arange(BM, dtype=np.int32)
    offs_n = bid_n * BN + ct.arange(BN, dtype=np.int32)

    acc = ct.full((BM, BN), 0.0, dtype=np.float32)

    for kh in range(KH):
        row = (offs_m + (kh - PAD_H))[:, None]
        col = offs_n[None, :]
        row2 = ct.broadcast_to(row, (BM, BN))
        col2 = ct.broadcast_to(col, (BM, BN))
        vals = ct.gather(x, (row2, col2), padding_value=0.0)
        w_val = ct.load(k, index=(kh,), shape=())
        acc = acc + ct.astype(vals, np.float32) * ct.astype(w_val, np.float32)

    out_tile = ct.astype(acc, x.dtype)
    ct.store(out, index=(bid_m, bid_n), tile=out_tile)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    # Separable Gaussian: K[i,j] = col_k[i] * row_k[j]
    k2d = kernel.view(KH, KW)
    center_f32 = k2d[KH // 2, KW // 2].float()
    row_k = k2d[KH // 2, :].contiguous()
    col_k = (k2d[:, KW // 2].float() / center_f32).to(kernel.dtype).contiguous()

    intermediate = torch.empty_like(input)
    output = torch.empty_like(input)

    x2d = input.view(H, W)
    i2d = intermediate.view(H, W)
    o2d = output.view(H, W)

    BM = 32
    BN = 128
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(H, BM), ct.cdiv(W, BN), 1)

    ct.launch(stream, grid, _hconv_kernel,
              (x2d, row_k, i2d, KW, PAD_W, BM, BN))
    ct.launch(stream, grid, _vconv_kernel,
              (i2d, col_k, o2d, KH, PAD_H, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BM": BM,
        "BN": BN,
        "occupancy": occupancy,
        "approach": "separable",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
