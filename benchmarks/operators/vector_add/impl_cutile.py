import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def vec_add_kernel_1d(a, b, c, TILE: ConstInt):
    """
    cuTile kernel for 1D element-wise vector addition using direct tiled loads/stores.
    """
    # Get the global ID of the current block along the first dimension.
    bid = ct.bid(0)

    # Load TILE-sized chunks from input vectors 'a' and 'b'.
    # index=(bid,) specifies which tile to load based on the block ID.
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # Perform the element-wise addition on the loaded tiles.
    sum_tile = a_tile + b_tile

    # Store the resulting TILE-sized chunk back to the output vector 'c'.
    ct.store(c, index=(bid,), tile=sum_tile)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024):
    """
    Wrapper for cuTile vector addition.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")

    c = torch.empty_like(a)
    N = a.shape[0]

    # Use a fixed tile size for benchmarking consistency, or heuristic
    TILE = block_size
    grid = (math.ceil(N / TILE), 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, vec_add_kernel_1d, (a, b, c, TILE))

    return c
