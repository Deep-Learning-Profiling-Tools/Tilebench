import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv2d_kernel(x, k, out,
                   KH: ConstInt, KW: ConstInt,
                   PAD_H: ConstInt, PAD_W: ConstInt,
                   BM: ConstInt, BN: ConstInt,
                   HM: ConstInt, HN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    base_m = bid_m * BM - PAD_H
    base_n = bid_n * BN - PAD_W

    rows = ct.arange(HM, dtype=np.int32) + base_m       # [HM]
    cols = ct.arange(HN, dtype=np.int32) + base_n       # [HN]

    row2d = ct.broadcast_to(rows[:, None], (HM, HN))
    col2d = ct.broadcast_to(cols[None, :], (HM, HN))

    # Load halo'd input region once
    halo = ct.gather(x, (row2d, col2d), padding_value=0.0)
    halo_f32 = ct.astype(halo, np.float32)

    acc = ct.full((BM, BN), 0.0, dtype=np.float32)

    # Reuse halo for all kernel taps via static sub-tile extraction
    for kh in range(KH):
        for kw in range(KW):
            sub = ct.extract(halo_f32, (kh, kw), shape=(BM, BN))
            w_val = ct.load(k, index=(kh, kw), shape=())
            acc = acc + sub * ct.astype(w_val, np.float32)

    ct.store(out, index=(bid_m, bid_n), tile=ct.astype(acc, x.dtype))


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    x2d = input.view(H, W)
    k2d = kernel.view(KH, KW)
    out2d = output.view(H, W)

    # Output tile (must be pow2)
    BM = 32
    BN = 64
    # Halo tile (must be pow2, >= BM+KH-1 and BN+KW-1).
    # For KH=KW=7: BM+6=38 -> HM=64; BN+6=70 -> HN=128.
    HM = 64
    HN = 128
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(H, BM), ct.cdiv(W, BN), 1)
    ct.launch(stream, grid, _conv2d_kernel,
              (x2d, k2d, out2d, KH, KW, PAD_H, PAD_W, BM, BN, HM, HN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BM": BM,
        "BN": BN,
        "HM": HM,
        "HN": HN,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
