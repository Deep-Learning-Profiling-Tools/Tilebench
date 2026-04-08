import torch
import cuda.tile as ct
import math


ConstInt = ct.Constant[int]

@ct.kernel
def softmax_kernel(
    input_tensor,  
    output_tensor,   
    N_COLS: ConstInt,  
    TILE_SIZE: ConstInt 
):
    """
    input_tensor: (Rows, Cols)
    output_tensor: (Rows, Cols)
    """
    row_idx = ct.bid(0)
    
    tile = ct.load(input_tensor, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.NEG_INF)

    max_val = ct.max(tile)

    exp_tile = ct.exp(tile - max_val)

    sum_val = ct.sum(exp_tile)

    output_tile = exp_tile / sum_val
    ct.store(output_tensor, index=(row_idx, 0), tile=output_tile)


def run(x: torch.Tensor, block_size: int, autotune: bool = False):
    n_rows, n_cols = x.shape
    

    if block_size < n_cols:
        raise RuntimeError(f"Block size ({block_size}) must be >= n_cols ({n_cols}) for cuTile Softmax.")
    
    output = torch.empty_like(x)
    

    grid = (n_rows, 1, 1)
    

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax_kernel,
        (x, output, n_cols, block_size)
    )
    
    return output

def get_last_config() -> dict | None:
    return None