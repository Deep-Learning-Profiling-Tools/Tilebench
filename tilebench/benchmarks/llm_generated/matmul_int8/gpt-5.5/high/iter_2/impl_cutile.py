import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _matmul_int8_packed_pair_kernel(
    a,
    b,
    output,
    KB: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TKB: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.full((TM, TN), 0, dtype=np.int32)
    num_k_tiles = ct.cdiv(KB, TKB)

    a0_view = a.slice(1, 0, KB)
    a1_view = a.slice(1, KB, 2 * KB)
    a2_view = a.slice(1, 2 * KB, 3 * KB)
    a3_view = a.slice(1, 3 * KB, 4 * KB)

    for j in range(0, num_k_tiles):
        b_tile = ct.load(
            b,
            index=(j, bid_n),
            shape=(TKB, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_i32 = ct.astype(b_tile, np.int32)

        b0_bits = ct.bitwise_and(b_i32, 3)
        b1_bits = ct.bitwise_and(ct.bitwise_rshift(b_i32, 2), 3)
        b0 = ct.astype(b0_bits - 1, np.int8)
        b1 = ct.astype(b1_bits - 1, np.int8)
        b01 = ct.cat((b0, b1), axis=0)

        a0 = ct.load(
            a0_view,
            index=(bid_m, j),
            shape=(TM, TKB),
            padding_mode=ct.PaddingMode.ZERO,
        )
        a1 = ct.load(
            a1_view,
            index=(bid_m, j),
            shape=(TM, TKB),
            padding_mode=ct.PaddingMode.ZERO,
        )
        a01 = ct.cat((a0, a1), axis=1)

        acc = ct.mma(a01, b01, acc)

        b2_bits = ct.bitwise_and(ct.bitwise_rshift(b_i32, 4), 3)
        b3_bits = ct.bitwise_and(ct.bitwise_rshift(b_i32, 6), 3)
        b2 = ct.astype(b2_bits - 1, np.int8)
        b3 = ct.astype(b3_bits - 1, np.int8)
        b23 = ct.cat((b2, b3), axis=0)

        a2 = ct.load(
            a2_view,
            index=(bid_m, j),
            shape=(TM, TKB),
            padding_mode=ct.PaddingMode.ZERO,
        )
        a3 = ct.load(
            a3_view,
            index=(bid_m, j),
            shape=(TM, TKB),
            padding_mode=ct.PaddingMode.ZERO,
        )
        a23 = ct.cat((a2, a3), axis=1)

        acc = ct.mma(a23, b23, acc)

    ct.store(output, index=(bid_m, bid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    KB = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    TM = 128
    TN = 256
    TKB = 64
    occupancy = 1

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    kernel = _matmul_int8_packed_pair_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (a, b, output, KB, TM, TN, TKB))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TM": TM,
            "TN": TN,
            "TKB": TKB,
            "DOT_K": 2 * TKB,
            "PAIR_FIELDS": 2,
            "occupancy": occupancy,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
