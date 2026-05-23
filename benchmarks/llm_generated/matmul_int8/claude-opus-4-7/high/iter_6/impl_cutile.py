import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _matmul_int8_kernel(
    a, b, out,
    K_b_tiles: ConstInt,
    TM: ConstInt, TN: ConstInt, TKB: ConstInt,
):
    bm = ct.bid(0)
    bn = ct.bid(1)

    acc = ct.full((TM, TN), 0, dtype=np.int32)

    for kb_block in range(0, K_b_tiles):
        b_packed = ct.load(
            b, index=(kb_block, bn), shape=(TKB, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_packed_i32 = ct.astype(b_packed, np.int32)

        # Unpack into 4 (TKB, TN) int8 tiles
        b0 = ct.astype(((b_packed_i32 >> 0) & 3) - 1, np.int8)
        b1 = ct.astype(((b_packed_i32 >> 2) & 3) - 1, np.int8)
        b2 = ct.astype(((b_packed_i32 >> 4) & 3) - 1, np.int8)
        b3 = ct.astype(((b_packed_i32 >> 6) & 3) - 1, np.int8)
        # Concat along axis 0 → (4*TKB, TN)
        b_big = ct.cat(
            (ct.cat((b0, b1), axis=0), ct.cat((b2, b3), axis=0)),
            axis=0,
        )

        # Load 4 (TM, TKB) chunks of A and concat along axis 1 → (TM, 4*TKB)
        a0 = ct.load(a, index=(bm, 0 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a1 = ct.load(a, index=(bm, 1 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a2 = ct.load(a, index=(bm, 2 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a3 = ct.load(a, index=(bm, 3 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a_big = ct.cat(
            (ct.cat((a0, a1), axis=1), ct.cat((a2, a3), axis=1)),
            axis=1,
        )

        acc = ct.mma(a_big, b_big, acc)

    ct.store(out, index=(bm, bn), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    TM = 128
    TN = 128
    TKB = 64
    occupancy = 2

    K_b_tiles = (K_b + TKB - 1) // TKB

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, _matmul_int8_kernel,
              (a, b, output, K_b_tiles, TM, TN, TKB))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM, "TN": TN, "TKB": TKB,
        "TK_DOT": 4 * TKB,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
