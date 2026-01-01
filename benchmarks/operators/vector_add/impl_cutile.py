import torch
import cuda.tile as ct
import numpy as np
import math

# Define the cuTile kernel
@ct.kernel
def vector_add_kernel(X, Y, Out, N: ct.Constant[int], TILE_N: ct.Constant[int]):
    # Get block ID
    bid = ct.bid(0)
    
    # Calculate offsets and mask
    offs = bid * TILE_N + ct.arange(TILE_N, dtype=np.int32)
    mask = offs < N
    
    # Load tiles from X and Y
    # Note: cuTile load/store expects index as a tuple and shape as a tuple
    x_tile = ct.load(X, index=(bid * TILE_N,), shape=(TILE_N,))
    y_tile = ct.load(Y, index=(bid * TILE_N,), shape=(TILE_N,))
    
    # Perform addition
    out_tile = x_tile + y_tile
    
    # Store the result tile back to Out
    ct.store(Out, index=(bid * TILE_N,), tile=out_tile)

def run(x: torch.Tensor, y: torch.Tensor):
    n = x.numel()
    out = torch.empty_like(x)
    
    tile_n = 1024
    grid = (math.ceil(n / tile_n), 1, 1)
    
    ct.launch(torch.cuda.current_stream(), grid, vector_add_kernel, (
        x, y, out, n, tile_n
    ))
    
    return out
