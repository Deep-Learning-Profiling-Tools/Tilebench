import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


def _tile_dim(block_size: int) -> int:
    root = int(math.sqrt(block_size))
    tile = 16
    while tile * 2 <= root:
        tile *= 2
    return max(16, tile)


@ct.kernel
def _gemm_kernel(a_ptr, b_ptr, c_ptr, K_TILES: ConstInt, TILE: ConstInt):
    """GEMM kernel: C = A @ B, tile-based."""
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    acc = ct.zeros((TILE, TILE), dtype=ct.float32)
    for bid_k in range(K_TILES):
        a_tile = ct.load(a_ptr, index=(bid_m, bid_k), shape=(TILE, TILE))
        b_tile = ct.load(b_ptr, index=(bid_k, bid_n), shape=(TILE, TILE))
        acc = acc + ct.matmul(a_tile, b_tile)
    ct.store(c_ptr, index=(bid_m, bid_n), tile=acc)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 1D convolution via im2col (unfold) + GEMM.
    input:  flat 1D tensor of size input_size
    kernel: flat 1D tensor of size kernel_size
    output: flat 1D tensor of size output_size = input_size - kernel_size + 1
    """
    output_size = input_size - kernel_size + 1

    tile = _tile_dim(block_size)
    stream = torch.cuda.current_stream()

    # im2col: unfold input to (output_size, kernel_size) matrix
    col = input.float().unfold(0, kernel_size, 1).contiguous()  # (output_size, kernel_size)

    # Pad to tile multiples for GEMM
    M, K = output_size, kernel_size
    M_pad = (M + tile - 1) // tile * tile
    K_pad = (K + tile - 1) // tile * tile
    N_pad = tile  # Single output channel, padded from 1 to tile

    a_pad = torch.zeros((M_pad, K_pad), device=input.device, dtype=torch.float32)
    b_pad = torch.zeros((K_pad, N_pad), device=input.device, dtype=torch.float32)
    a_pad[:M, :K] = col
    b_pad[:K, 0] = kernel.float()

    c_pad = torch.empty((M_pad, N_pad), device=input.device, dtype=torch.float32)
    grid = (M_pad // tile, N_pad // tile, 1)
    ct.launch(stream, grid, _gemm_kernel, (a_pad, b_pad, c_pad, K_pad // tile, tile))

    return c_pad[:M, 0].to(input.dtype)


def get_last_config() -> dict | None:
    return None
