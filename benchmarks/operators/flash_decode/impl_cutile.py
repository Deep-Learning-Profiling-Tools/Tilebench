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

def run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor, block_size: int = None, autotune: bool = False):

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

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Profile Flash Decode Stage 2 cuTile Kernel")
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
    parser.add_argument("-h_q", "--heads", type=int, default=32, help="Number of Query Heads")
    parser.add_argument("-s", "--seq-len", type=int, default=4096, help="Sequence Length (Context Length)")
    parser.add_argument("-d", "--head-dim", type=int, default=128, help="Head Dimension")
    parser.add_argument("--block-seq", type=int, default=128, help="Stage 1 Block Size (Split size)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"], help="Data type")
    
    args = parser.parse_args()

    # 1. Setup Parameters
    device = "cuda"
    dtype = getattr(torch, args.dtype)
    
    batch = args.batch
    heads = args.heads
    seq_len = args.seq_len
    head_dim = args.head_dim
    block_seq = args.block_seq

    print(f"Profiling Config: Batch={batch}, Heads={heads}, SeqLen={seq_len}, Dim={head_dim}, BlockSeq={block_seq}, Dtype={dtype}")

    # 2. Data Generation (Simulation)
    num_blocks = (seq_len + block_seq - 1) // block_seq
    
    # B_Seqlen: Full context length
    b_seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    
    # Mid_O: Simulated output from Stage 1
    mid_o = torch.randn((batch, heads, num_blocks, head_dim), dtype=dtype, device=device)
    
    # Mid_O_LSE: Simulated LSE from Stage 1
    mid_o_lse = torch.randn((batch, heads, num_blocks), dtype=dtype, device=device)
    
    # Block Seq as Tensor (matches signature)
    block_seq_tensor = torch.tensor(block_seq, dtype=torch.int32, device='cpu')

    # 3. Warmup (Essential for JIT compilation)
    print("Warming up...")
    for _ in range(10):
        run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
    torch.cuda.synchronize()

    # 4. Profiling Run
    # Use nvtx to mark the range if viewing in Nsight Systems, 
    # but for ncu (Nsight Compute), just running it is enough.
    print("Starting Profile Run...")
    
    # Optional: Loop to ensure we capture enough samples if needed, 
    # but usually 1 run is enough for ncu --set full
    torch.cuda.nvtx.range_push("FlashDecodeStage2_cuTile")
    run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.synchronize()
    print("Done.")

def get_last_config() -> dict | None:
    return None