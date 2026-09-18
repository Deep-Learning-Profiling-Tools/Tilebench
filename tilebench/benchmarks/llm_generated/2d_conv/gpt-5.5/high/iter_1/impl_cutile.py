import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
def _conv2d_implicit_gemm_kernel(
    inp,
    w_flat,
    out3,
    KH: ConstInt,
    KW: ConstInt,
    STRIDE: ConstInt,
    PADDING: ConstInt,
    GROUPS: ConstInt,
    TO: ConstInt,
    TP: ConstInt,
    TK: ConstInt,
):
    pid = ct.bid(0)

    CIN = inp.shape[1]
    H = inp.shape[2]
    W = inp.shape[3]
    COUT = out3.shape[1]
    SPATIAL = out3.shape[2]

    OW = (W + 2 * PADDING - KW) // STRIDE + 1
    CIN_G = CIN // GROUPS
    COUT_G = COUT // GROUPS
    K_TOTAL = CIN_G * KH * KW
    KHW = KH * KW

    num_pos_tiles = ct.cdiv(SPATIAL, TP)
    num_oc_tiles = ct.cdiv(COUT_G, TO)
    tiles_per_bg = num_pos_tiles * num_oc_tiles

    bg = pid // tiles_per_bg
    rem = pid - bg * tiles_per_bg

    batch = bg // GROUPS
    group = bg - batch * GROUPS

    pid_oc = rem // num_pos_tiles
    pid_pos = rem - pid_oc * num_pos_tiles

    acc = ct.full((TO, TP), 0.0, dtype=np.float32)

    offs_o_col = ct.arange(TO, dtype=np.int32)[:, None]
    offs_p_row = ct.arange(TP, dtype=np.int32)[None, :]
    offs_k_col = ct.arange(TK, dtype=np.int32)[:, None]
    offs_k_row = ct.arange(TK, dtype=np.int32)[None, :]

    pos = pid_pos * TP + offs_p_row
    ow = pos - (pos // OW) * OW
    oh = pos // OW

    for kt in range(0, ct.cdiv(K_TOTAL, TK)):
        if (COUT_G % TO) == 0:
            row_tile = group * (COUT_G // TO) + pid_oc
            w_tile = ct.load(
                w_flat,
                index=(row_tile, kt),
                shape=(TO, TK),
                padding_mode=ct.PaddingMode.ZERO,
                latency=2,
            )
        else:
            oc_w = group * COUT_G + pid_oc * TO + offs_o_col
            kk_w = kt * TK + offs_k_row
            w_tile = ct.gather(
                w_flat,
                (oc_w, kk_w),
                padding_value=0.0,
                check_bounds=True,
                latency=2,
            )
            w_valid = (oc_w < (group + 1) * COUT_G) & (kk_w < K_TOTAL)
            w_tile = ct.where(w_valid, w_tile, 0.0)

        kk = kt * TK + offs_k_col
        ic_g = kk // KHW
        rem_k = kk - ic_g * KHW
        kh = rem_k // KW
        kw = rem_k - kh * KW

        ih = oh * STRIDE + kh - PADDING
        iw = ow * STRIDE + kw - PADDING
        ic = group * CIN_G + ic_g

        x_tile = ct.gather(
            inp,
            (batch, ic, ih, iw),
            padding_value=0.0,
            check_bounds=True,
            latency=1,
        )
        x_valid = (kk < K_TOTAL) & (pos < SPATIAL)
        x_tile = ct.where(x_valid, x_tile, 0.0)

        acc = ct.mma(w_tile, x_tile, acc)

    out_tile = ct.astype(acc, inp.dtype)

    if (COUT_G % TO) == 0:
        row_tile = group * (COUT_G // TO) + pid_oc
        ct.store(
            out3,
            index=(batch, row_tile, pid_pos),
            tile=out_tile.reshape((1, TO, TP)),
            latency=1,
        )
    else:
        oc = group * COUT_G + pid_oc * TO + offs_o_col
        ct.scatter(
            out3,
            (batch, oc, pos),
            out_tile,
            check_bounds=True,
            latency=1,
        )


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    **kwargs
):
    batch = input.shape[0]
    in_channels = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]

    out_channels = weight.shape[0]
    KH = weight.shape[2]
    KW = weight.shape[3]

    out_H = (H + 2 * padding - KH) // stride + 1
    out_W = (W + 2 * padding - KW) // stride + 1
    spatial = out_H * out_W

    output = torch.empty(
        (batch, out_channels, out_H, out_W),
        device=input.device,
        dtype=input.dtype,
    )

    weight_flat = weight.reshape(out_channels, -1)
    output_flat = output.reshape(batch, out_channels, spatial)

    TO = 128
    TP = 128
    TK = 64
    occupancy = 1

    cout_g = out_channels // groups
    grid = (
        batch
        * groups
        * ct.cdiv(cout_g, TO)
        * ct.cdiv(spatial, TP),
        1,
        1,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _conv2d_implicit_gemm_kernel,
        (
            input,
            weight_flat,
            output_flat,
            KH,
            KW,
            stride,
            padding,
            groups,
            TO,
            TP,
            TK,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TO": TO,
            "TP": TP,
            "TK": TK,
            "occupancy": occupancy,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
