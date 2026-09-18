import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
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
        for i in range(4):
            b_masked = (b_packed >> (2 * i)) & 3
            b_int8 = ct.astype(b_masked, ct.int8) - ct.astype(ct.full((1,), 1, dtype=np.int8), ct.int8).item() if False else ct.astype(b_masked, ct.int8) - 1
            a_tile = ct.load(
                a, index=(bm, i * K_b_tiles + kb_block),
                shape=(TM, TKB),
                padding_mode=ct.PaddingMode.ZERO,
            )
            acc = ct.mma(a_tile, b_int8, acc)

    ct.store(out, index=(bm, bn), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    TM = 128
    TN = 128
    TKB = 64
    occupancy = 2

    K_b_tiles = K_b // TKB

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    kernel = _matmul_int8_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (a, b, output, K_b_tiles, TM, TN, TKB))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM, "TN": TN, "TKB": TKB,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
