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
def quantized_gemm_kernel(a_ptr, b_ptr, c_ptr, out_scale, K_TILES: ConstInt, TILE: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    acc = ct.zeros((TILE, TILE), dtype=ct.float32)
    for bid_k in range(K_TILES):
        a_tile = ct.astype(ct.load(a_ptr, index=(bid_m, bid_k), shape=(TILE, TILE)), ct.float32)
        b_tile = ct.astype(ct.load(b_ptr, index=(bid_k, bid_n), shape=(TILE, TILE)), ct.float32)
        acc = acc + ct.matmul(a_tile, b_tile)
    ct.store(c_ptr, index=(bid_m, bid_n), tile=acc * out_scale)


def run(a_q: torch.Tensor, b_q: torch.Tensor, scale: float, block_size: int = 1024, autotune: bool = False, **kwargs):
    if a_q.dim() != 2 or b_q.dim() != 2:
        raise ValueError("quantized_gemm expects 2D inputs.")
    if a_q.shape[1] != b_q.shape[0]:
        raise ValueError("Inner dimensions must match for GEMM.")

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    m, k = a_q.shape
    _, n = b_q.shape

    tile = _tile_dim_from_block_size(block_size)
    m_pad = (m + tile - 1) // tile * tile
    k_pad = (k + tile - 1) // tile * tile
    n_pad = (n + tile - 1) // tile * tile

    a_pad = torch.zeros((m_pad, k_pad), device=a_q.device, dtype=a_q.dtype)
    b_pad = torch.zeros((k_pad, n_pad), device=b_q.device, dtype=b_q.dtype)
    a_pad[:m, :k] = a_q
    b_pad[:k, :n] = b_q

    c_pad = torch.empty((m_pad, n_pad), device=a_q.device, dtype=torch.float32)
    grid = (m_pad // tile, n_pad // tile, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        quantized_gemm_kernel,
        (a_pad, b_pad, c_pad, scale * scale, k_pad // tile, tile),
    )
    return c_pad[:m, :n]


def get_last_config() -> dict | None:
    return None
