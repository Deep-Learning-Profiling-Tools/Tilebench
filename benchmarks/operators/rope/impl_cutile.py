import torch
import cuda.tile as ct
import math


ConstInt = ct.Constant[int]

@ct.kernel
def rope_kernel(
    Q,                  # Rank 4: [TotalTokens, Heads, 2, HalfDim]
    Cos,                # Rank 2: [SeqLen, HalfDim]
    Sin,                # Rank 2: [SeqLen, HalfDim]
    SeqLen: ConstInt,   
    TILE_DIM: ConstInt  # HalfDim
):

    row_id = ct.bid(0)   # Batch*Seq
    head_id = ct.bid(1)  # Heads

    seq_idx = row_id % SeqLen

    cos_tile = ct.load(Cos, index=(seq_idx, 0), shape=(1, TILE_DIM))
    sin_tile = ct.load(Sin, index=(seq_idx, 0), shape=(1, TILE_DIM))

    q1 = ct.load(Q, index=(row_id, head_id, 0, 0), shape=(1, 1, 1, TILE_DIM))
    
    q2 = ct.load(Q, index=(row_id, head_id, 1, 0), shape=(1, 1, 1, TILE_DIM))

    out1 = q1 * cos_tile - q2 * sin_tile
    out2 = q2 * cos_tile + q1 * sin_tile

    ct.store(Q, index=(row_id, head_id, 0, 0), tile=out1)
    ct.store(Q, index=(row_id, head_id, 1, 0), tile=out2)


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_size: int = None, autotune: bool = False):
    output = q.clone().contiguous()
    
    batch, seq_len, n_heads, head_dim = output.shape
    half_dim = head_dim // 2

    output_view = output.view(batch * seq_len, n_heads, 2, half_dim)

    cos_view = cos.view(seq_len, half_dim)
    sin_view = sin.view(seq_len, half_dim)

    grid = (batch * seq_len, n_heads, 1)

    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        rope_kernel, 
        (output_view, cos_view, sin_view, seq_len, half_dim)
    )
    
    return output

def get_last_config() -> dict | None:
    return None