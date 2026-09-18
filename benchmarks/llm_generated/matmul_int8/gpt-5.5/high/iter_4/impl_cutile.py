import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _matmul_int8_packed_kernel(
    a,
    b,
    output,
    KB: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.full((TM, TN), 0, dtype=np.int32)
    num_k_tiles = ct.cdiv(KB, TK)

    a0_view = a.slice(1, 0, KB)
    a1_view = a.slice(1, KB, 2 * KB)
    a2_view = a.slice(1, 2 * KB, 3 * KB)
    a3_view = a.slice(1, 3 * KB, 4 * KB)

    for j in range(0, num_k_tiles):
        b_tile = ct.load(
            b,
            index=(j, bid_n),
            shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )

        b0_bits = ct.bitwise_and(b_tile, 3)
        b0 = ct.astype(ct.astype(b0_bits, np.int8) - 1, np.int8)
        a0 = ct.load(
            a0_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a0, b0, acc)

        b1_bits = ct.bitwise_and(ct.bitwise_rshift(b_tile, 2), 3)
        b1 = ct.astype(ct.astype(b1_bits, np.int8) - 1, np.int8)
        a1 = ct.load(
            a1_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a1, b1, acc)

        b2_bits = ct.bitwise_and(ct.bitwise_rshift(b_tile, 4), 3)
        b2 = ct.astype(ct.astype(b2_bits, np.int8) - 1, np.int8)
        a2 = ct.load(
            a2_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a2, b2, acc)

        b3_bits = ct.bitwise_and(ct.bitwise_rshift(b_tile, 6), 3)
        b3 = ct.astype(ct.astype(b3_bits, np.int8) - 1, np.int8)
        a3 = ct.load(
            a3_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a3, b3, acc)

    ct.store(output, index=(bid_m, bid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    KB = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    TM = 128
    TN = 128
    TK = 128
    occupancy = 2

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, _matmul_int8_packed_kernel, (a, b, output, KB, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TM": TM,
            "TN": TN,
            "TK": TK,
            "occupancy": occupancy,
            "U8_SHIFT_DECODE": 1,
            "UNROLL_FIELDS": 4,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
