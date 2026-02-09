import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]

@ct.kernel
def flash_decode_stage2_kernel(
    Mid_O,              # [Batch, Head, NumBlocks, HeadDim]
    Mid_O_LSE,          # [Batch, Head, NumBlocks]
    B_Seqlen,           # [Batch]
    Out,                # [Batch, Head, HeadDim]
    HEAD_DIM: ConstInt,
    BLOCK_SEQ: ConstInt,
    TOTAL_BLOCKS: ConstInt 
):
    # 1. IDs
    bid_b = ct.bid(0)
    bid_h = ct.bid(1)
    

    seq_len = ct.load(B_Seqlen, index=(bid_b,), shape=(1,))

    real_num_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    

    acc = ct.full((1, 1, 1, HEAD_DIM), 0.0, dtype=ct.float32) 

    max_logic = ct.full((1, 1, 1), -float('inf'), dtype=ct.float32)
    sum_exp = ct.full((1, 1, 1), 0.0, dtype=ct.float32)
    
    for k in range(TOTAL_BLOCKS):

        is_valid = k < real_num_blocks 

        curr_lse = ct.load(Mid_O_LSE, index=(bid_b, bid_h, k), shape=(1, 1, 1))
        
        curr_o = ct.load(Mid_O, index=(bid_b, bid_h, k, 0), shape=(1, 1, 1, HEAD_DIM)) 
        
        curr_lse = ct.where(is_valid, curr_lse, -float('inf'))
        curr_o = ct.where(is_valid, curr_o, 0.0)
        
        new_max = ct.maximum(max_logic, curr_lse)
        
        scale = ct.exp(max_logic - new_max)
        curr_weight = ct.exp(curr_lse - new_max)
        
        acc = acc * scale
        
        # reshape curr_weight (1,1,1) -> (1,1,1,1)
        acc = acc + curr_o * curr_weight
        
        sum_exp = sum_exp * scale + curr_weight
        max_logic = new_max
            
    # 5. Store
    final_out = acc / sum_exp

    ct.store(Out, index=(bid_b, bid_h, 0, 0), tile=final_out)

def run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor, block_size: int = None):

    if isinstance(block_seq_tensor, torch.Tensor):
        block_seq = block_seq_tensor.item()
    else:
        block_seq = block_seq_tensor

    batch, head_num, num_blocks, head_dim = mid_o.shape
    out = torch.empty((batch, head_num, head_dim), dtype=mid_o.dtype, device=mid_o.device)
    

    out_view = out.view(batch, head_num, 1, head_dim)
    grid = (batch, head_num, 1)

    HEAD_DIM = head_dim
    TOTAL_BLOCKS = num_blocks
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        flash_decode_stage2_kernel,
        (mid_o, mid_o_lse, b_seqlen, out_view, HEAD_DIM, block_seq, TOTAL_BLOCKS)
    )
    
    return out