import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _max_pool2d_k3s2_bh8_split320_kernel(input2d, output,
                                          H: ConstInt, W: ConstInt,
                                          H_OUT: ConstInt, W_OUT: ConstInt,
                                          PADDING: ConstInt,
                                          TILE_A: ConstInt,
                                          TILE_B: ConstInt,
                                          IN_A: ConstInt,
                                          IN_B: ConstInt):
    h_block = ct.bid(0)
    nc = ct.bid(1)

    cols_a = ct.arange(TILE_A, dtype=np.int32)
    cols_b = TILE_A + ct.arange(TILE_B, dtype=np.int32)

    oh0 = h_block * 8
    ih_base0 = oh0 * 2 - PADDING
    base_row = nc * H

    acca0 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca1 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca2 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca3 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca4 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca5 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca6 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca7 = ct.full((TILE_A,), -np.inf, dtype=np.float32)

    accb0 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb1 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb2 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb3 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb4 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb5 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb6 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb7 = ct.full((TILE_B,), -np.inf, dtype=np.float32)

    for rh in range(0, 17):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)
        safe_row = ct.where(valid_h, base_row + ih, -1)

        rowa2d = ct.load(
            input2d,
            index=(safe_row, 0),
            shape=(1, IN_A),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )
        rowa = ct.reshape(rowa2d, (IN_A,))
        pairsa = ct.reshape(rowa, (TILE_A, 2))
        evena = ct.reshape(ct.extract(pairsa, (0, 0), shape=(TILE_A, 1)), (TILE_A,))
        odda = ct.reshape(ct.extract(pairsa, (0, 1), shape=(TILE_A, 1)), (TILE_A,))

        evena_f = ct.astype(evena, np.float32)
        odda_f = ct.astype(odda, np.float32)

        left_a_col = cols_a * 2 - PADDING
        valid_left_a = (cols_a < W_OUT) & valid_h & (left_a_col >= 0) & (left_a_col < W)
        safe_left_a = ct.where(valid_left_a, left_a_col, -1)
        lefta_f = ct.astype(
            ct.gather(input2d, (safe_row, safe_left_a),
                      padding_value=-np.inf, latency=1),
            np.float32,
        )
        hmaxa = ct.maximum(ct.maximum(lefta_f, evena_f), odda_f)

        rowb2d = ct.load(
            input2d,
            index=(safe_row, 4),
            shape=(1, IN_B),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )
        rowb = ct.reshape(rowb2d, (IN_B,))
        pairsb = ct.reshape(rowb, (TILE_B, 2))
        evenb = ct.reshape(ct.extract(pairsb, (0, 0), shape=(TILE_B, 1)), (TILE_B,))
        oddb = ct.reshape(ct.extract(pairsb, (0, 1), shape=(TILE_B, 1)), (TILE_B,))

        evenb_f = ct.astype(evenb, np.float32)
        oddb_f = ct.astype(oddb, np.float32)

        left_b_col = cols_b * 2 - PADDING
        valid_left_b = (cols_b < W_OUT) & valid_h & (left_b_col >= 0) & (left_b_col < W)
        safe_left_b = ct.where(valid_left_b, left_b_col, -1)
        leftb_f = ct.astype(
            ct.gather(input2d, (safe_row, safe_left_b),
                      padding_value=-np.inf, latency=1),
            np.float32,
        )
        hmaxb = ct.maximum(ct.maximum(leftb_f, evenb_f), oddb_f)

        if rh < 3:
            acca0 = ct.maximum(acca0, hmaxa)
            accb0 = ct.maximum(accb0, hmaxb)
        if (rh >= 2) and (rh < 5):
            acca1 = ct.maximum(acca1, hmaxa)
            accb1 = ct.maximum(accb1, hmaxb)
        if (rh >= 4) and (rh < 7):
            acca2 = ct.maximum(acca2, hmaxa)
            accb2 = ct.maximum(accb2, hmaxb)
        if (rh >= 6) and (rh < 9):
            acca3 = ct.maximum(acca3, hmaxa)
            accb3 = ct.maximum(accb3, hmaxb)
        if (rh >= 8) and (rh < 11):
            acca4 = ct.maximum(acca4, hmaxa)
            accb4 = ct.maximum(accb4, hmaxb)
        if (rh >= 10) and (rh < 13):
            acca5 = ct.maximum(acca5, hmaxa)
            accb5 = ct.maximum(accb5, hmaxb)
        if (rh >= 12) and (rh < 15):
            acca6 = ct.maximum(acca6, hmaxa)
            accb6 = ct.maximum(accb6, hmaxb)
        if rh >= 14:
            acca7 = ct.maximum(acca7, hmaxa)
            accb7 = ct.maximum(accb7, hmaxb)

    out_row = nc * H_OUT + oh0

    outa0 = ct.reshape(ct.astype(acca0, input2d.dtype), (1, TILE_A))
    outa1 = ct.reshape(ct.astype(acca1, input2d.dtype), (1, TILE_A))
    outa2 = ct.reshape(ct.astype(acca2, input2d.dtype), (1, TILE_A))
    outa3 = ct.reshape(ct.astype(acca3, input2d.dtype), (1, TILE_A))
    outa4 = ct.reshape(ct.astype(acca4, input2d.dtype), (1, TILE_A))
    outa5 = ct.reshape(ct.astype(acca5, input2d.dtype), (1, TILE_A))
    outa6 = ct.reshape(ct.astype(acca6, input2d.dtype), (1, TILE_A))
    outa7 = ct.reshape(ct.astype(acca7, input2d.dtype), (1, TILE_A))

    outb0 = ct.reshape(ct.astype(accb0, input2d.dtype), (1, TILE_B))
    outb1 = ct.reshape(ct.astype(accb1, input2d.dtype), (1, TILE_B))
    outb2 = ct.reshape(ct.astype(accb2, input2d.dtype), (1, TILE_B))
    outb3 = ct.reshape(ct.astype(accb3, input2d.dtype), (1, TILE_B))
    outb4 = ct.reshape(ct.astype(accb4, input2d.dtype), (1, TILE_B))
    outb5 = ct.reshape(ct.astype(accb5, input2d.dtype), (1, TILE_B))
    outb6 = ct.reshape(ct.astype(accb6, input2d.dtype), (1, TILE_B))
    outb7 = ct.reshape(ct.astype(accb7, input2d.dtype), (1, TILE_B))

    ct.store(output, index=(out_row, 0), tile=outa0, allow_tma=False)
    ct.store(output, index=(out_row, 4), tile=outb0, allow_tma=False)
    ct.store(output, index=(out_row + 1, 0), tile=outa1, allow_tma=False)
    ct.store(output, index=(out_row + 1, 4), tile=outb1, allow_tma=False)
    ct.store(output, index=(out_row + 2, 0), tile=outa2, allow_tma=False)
    ct.store(output, index=(out_row + 2, 4), tile=outb2, allow_tma=False)
    ct.store(output, index=(out_row + 3, 0), tile=outa3, allow_tma=False)
    ct.store(output, index=(out_row + 3, 4), tile=outb3, allow_tma=False)
    ct.store(output, index=(out_row + 4, 0), tile=outa4, allow_tma=False)
    ct.store(output, index=(out_row + 4, 4), tile=outb4, allow_tma=False)
    ct.store(output, index=(out_row + 5, 0), tile=outa5, allow_tma=False)
    ct.store(output, index=(out_row + 5, 4), tile=outb5, allow_tma=False)
    ct.store(output, index=(out_row + 6, 0), tile=outa6, allow_tma=False)
    ct.store(output, index=(out_row + 6, 4), tile=outb6, allow_tma=False)
    ct.store(output, index=(out_row + 7, 0), tile=outa7, allow_tma=False)
    ct.store(output, index=(out_row + 7, 4), tile=outb7, allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    input2d = input.reshape(N * C * H, W)
    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE_A = 256
    TILE_B = 64
    IN_A = 512
    IN_B = 128
    BLOCK_H = 8
    occupancy = 4

    grid = (ct.cdiv(H_out, BLOCK_H), N * C, 1)
    ct.launch(
        stream,
        grid,
        _max_pool2d_k3s2_bh8_split320_kernel,
        (input2d, output, H, W, H_out, W_out, padding, TILE_A, TILE_B, IN_A, IN_B),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh8_split320_fixed",
        "TILE_A": TILE_A,
        "TILE_B": TILE_B,
        "IN_A": IN_A,
        "IN_B": IN_B,
        "BLOCK_H": BLOCK_H,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
