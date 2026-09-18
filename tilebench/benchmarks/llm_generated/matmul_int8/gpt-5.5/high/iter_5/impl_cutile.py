import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _zero_output_kernel(
    output,
    TM: ConstInt,
    TN: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    z = ct.full((TM, TN), 0, dtype=np.int32)
    ct.store(output, index=(bid_m, bid_n), tile=z)


@ct.kernel(occupancy=2)
def _matmul_int8_packed_splitk_kernel(
    a,
    b,
    output,
    KB: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    SPLIT_K: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    split_id = ct.bid(2)

    acc = ct.full((TM, TN), 0, dtype=np.int32)

    k_tiles_per_split = KB // (SPLIT_K * TK)
    k_tile_base = split_id * k_tiles_per_split

    for j in range(0, k_tiles_per_split):
        kj = k_tile_base + j

        b_tile = ct.load(
            b,
            index=(kj, bid_n),
            shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )

        for field in range(0, 4):
            a_field = a.slice(1, field * KB, (field + 1) * KB)
            a_tile = ct.load(
                a_field,
                index=(bid_m, kj),
                shape=(TM, TK),
                padding_mode=ct.PaddingMode.ZERO,
            )

            shifted = ct.bitwise_rshift(ct.astype(b_tile, np.int32), 2 * field)
            bits = ct.bitwise_and(shifted, 3)
            b_vals = ct.astype(bits - 1, np.int8)

            acc = ct.mma(a_tile, b_vals, acc)

    offs_m = bid_m * TM + ct.arange(TM, dtype=np.int32)[:, None]
    offs_n = bid_n * TN + ct.arange(TN, dtype=np.int32)[None, :]

    ct.atomic_add(
        output,
        (offs_m, offs_n),
        acc,
        memory_order=ct.MemoryOrder.RELAXED,
        memory_scope=ct.MemoryScope.DEVICE,
    )


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    KB = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    TM = 128
    TN = 128
    TK = 128
    SPLIT_K = 2
    occupancy = 2
    zero_occupancy = 4

    tile_grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, tile_grid, _zero_output_kernel, (output, TM, TN))

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), SPLIT_K)
    ct.launch(
        stream,
        grid,
        _matmul_int8_packed_splitk_kernel,
        (a, b, output, KB, TM, TN, TK, SPLIT_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TM": TM,
            "TN": TN,
            "TK": TK,
            "SPLIT_K": SPLIT_K,
            "occupancy": occupancy,
            "zero_occupancy": zero_occupancy,
            "ATOMIC_SPLITK": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
