import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
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
    num_k_tiles = KB // TK

    for j in range(0, num_k_tiles):
        b_tile = ct.load(
            b,
            index=(j, bid_n),
            shape=(TK, TN),
        )

        for field in range(0, 4):
            a_field = a.slice(1, field * KB, (field + 1) * KB)
            a_tile = ct.load(
                a_field,
                index=(bid_m, j),
                shape=(TM, TK),
            )

            shifted = ct.bitwise_rshift(b_tile, 2 * field)
            bits = ct.bitwise_and(shifted, 3)
            b_vals = ct.astype(ct.astype(bits, np.int8) - 1, np.int8)

            acc = ct.mma(a_tile, b_vals, acc)

    ct.store(output, index=(bid_m, bid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    KB = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    TM = 64
    TN = 256
    TK = 128
    occupancy = 2

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    kernel = _matmul_int8_packed_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (a, b, output, KB, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TM": TM,
            "TN": TN,
            "TK": TK,
            "occupancy": occupancy,
            "U8_SHIFT_DECODE": 1,
            "LOAD_PADDING": 0,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
