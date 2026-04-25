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


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 3D convolution via vol2col + GEMM.
    input:  flat 1D tensor
    kernel: flat 1D tensor
    output: flat 1D tensor
    """
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    total_out = output_depth * output_rows * output_cols
    kernel_vol = kernel_depth * kernel_rows * kernel_cols

    tile = _tile_dim(block_size)
    stream = torch.cuda.current_stream()

    # Reshape flat input to 3D for vol2col
    inp_3d = input.float().view(input_depth, input_rows, input_cols)

    # Vol2col: extract all kD x kH x kW patches via unfold
    patches = inp_3d.unfold(0, kernel_depth, 1).unfold(1, kernel_rows, 1).unfold(2, kernel_cols, 1)
    # patches: (oD, oH, oW, kD, kH, kW)
    col = patches.contiguous().reshape(total_out, kernel_vol)  # (M, K)

    # Pad to tile multiples for GEMM
    M, K = total_out, kernel_vol
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
