import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


def _tile_dim_from_block_size(block_size: int) -> int:
    root = int(math.sqrt(block_size))
    tile = 16
    while tile * 2 <= root:
        tile *= 2
    return max(16, tile)


@ct.kernel
def streamk_matmul_kernel(a_ptr, b_ptr, c_ptr, K_TILES: ConstInt, TILE: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    acc = ct.zeros((TILE, TILE), dtype=ct.float32)
    for bid_k in range(K_TILES):
        a_tile = ct.astype(ct.load(a_ptr, index=(bid_m, bid_k), shape=(TILE, TILE)), ct.float32)
        b_tile = ct.astype(ct.load(b_ptr, index=(bid_k, bid_n), shape=(TILE, TILE)), ct.float32)
        acc = acc + ct.matmul(a_tile, b_tile)
    ct.store(c_ptr, index=(bid_m, bid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024, **kwargs):
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("streamk_scheduling expects 2D inputs.")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matmul.")

    a = a.contiguous()
    b = b.contiguous()
    m, k = a.shape
    _, n = b.shape

    tile = _tile_dim_from_block_size(block_size)
    m_pad = math.ceil(m / tile) * tile
    k_pad = math.ceil(k / tile) * tile
    n_pad = math.ceil(n / tile) * tile

    a_pad = torch.zeros((m_pad, k_pad), device=a.device, dtype=a.dtype)
    b_pad = torch.zeros((k_pad, n_pad), device=b.device, dtype=b.dtype)
    a_pad[:m, :k] = a
    b_pad[:k, :n] = b

    c_pad = torch.empty((m_pad, n_pad), device=a.device, dtype=torch.float32)
    grid = (m_pad // tile, n_pad // tile, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        streamk_matmul_kernel,
        (a_pad, b_pad, c_pad, k_pad // tile, tile),
    )
    return c_pad[:m, :n]
