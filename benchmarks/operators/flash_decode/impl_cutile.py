import torch
import cuda.tile as ct
import math
import nvtx

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

TILE_D_CHUNK = 32

@ct.kernel
def flash_decode_stage2_kernel_split_d(
    Mid_O,              # [Batch, Head, NumBlocks, HeadDim]
    Mid_O_LSE,          # [Batch, Head, NumBlocks]
    B_Seqlen,           # [Batch]
    Out,                # [Batch, Head, HeadDim]
    HEAD_DIM: ConstInt,
    BLOCK_SEQ: ConstInt,
    TILE_D: ConstInt,   # 【内部参数】接收 TILE_D_CHUNK
    TOTAL_BLOCKS: int   # 【关键】运行时变量，防止编译器过度展开循环
):
    bid_b = ct.bid(0)
    bid_h = ct.bid(1)
    
    # 获取真实块数
    seq_len = ct.load(B_Seqlen, index=(bid_b,), shape=(1,))
    real_num_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    # =========================================================
    # Pass 1: 计算 Global Max LSE
    # =========================================================
    global_max = ct.full((1, 1, 1), -float('inf'), dtype=ct.float32)
    
    for k in range(TOTAL_BLOCKS):
        is_valid = k < real_num_blocks
        curr_lse = ct.load(Mid_O_LSE, index=(bid_b, bid_h, k), shape=(1, 1, 1))
        curr_lse = ct.where(is_valid, curr_lse, -float('inf'))
        global_max = ct.maximum(global_max, curr_lse)
        
    # =========================================================
    # Pass 2: 维度切分循环 (Split-D)
    # =========================================================
    
    # 计算切分数量。由于 HEAD_DIM 和 TILE_D 都是 ConstInt，cuTile 可以进行常量折叠
    num_chunks = HEAD_DIM // TILE_D

    # 使用 static_range 强制展开外层循环 (Split-D)
    # 这会生成 num_chunks 个独立的指令流，每个处理 32 个 float
    for i in range(num_chunks):
        d_offset = i * TILE_D
        
        # 初始化当前 Chunk 的 Accumulator (Shape 使用 TILE_D=32)
        # 32 个 float32 寄存器占用很小，极大缓解 Register Pressure
        acc_chunk = ct.full((1, 1, 1, TILE_D), 0.0, dtype=ct.float32)
        sum_exp = ct.full((1, 1, 1), 0.0, dtype=ct.float32)
        
        # 内层循环：遍历所有 Block (不展开，因为 TOTAL_BLOCKS 是 int)
        for k in range(TOTAL_BLOCKS):
            is_valid = k < real_num_blocks
            
            # Load LSE
            curr_lse = ct.load(Mid_O_LSE, index=(bid_b, bid_h, k), shape=(1, 1, 1))
            
            # Load Mid_O Chunk
            # index=(..., d_offset) 指向当前切片的起始位置
            curr_o_chunk = ct.load(Mid_O, index=(bid_b, bid_h, k, d_offset), shape=(1, 1, 1, TILE_D))
            
            # Masking
            curr_o_chunk = ct.where(is_valid, curr_o_chunk, 0.0)
            curr_lse = ct.where(is_valid, curr_lse, -float('inf'))
            
            # Compute Weight
            weight = ct.exp(curr_lse - global_max)
            
            # Accumulate
            acc_chunk = acc_chunk + curr_o_chunk * weight
            sum_exp = sum_exp + weight
        
        # Finalize Chunk
        final_out_chunk = acc_chunk / sum_exp
        
        # Cast & Store Chunk (显式类型转换)
        final_out_chunk = ct.astype(final_out_chunk, Out.dtype)
        ct.store(Out, index=(bid_b, bid_h, 0, d_offset), tile=final_out_chunk)


# 保持 run 函数签名与 Engine 兼容
def run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor, block_size: int = None):
    # 处理 block_seq 参数
    if isinstance(block_seq_tensor, torch.Tensor):
        block_seq = block_seq_tensor.item()
    else:
        block_seq = block_seq_tensor

    batch, head_num, num_blocks, head_dim = mid_o.shape
    
    # 简单的断言，确保 Head Dim 可以被 32 整除 (目前绝大多数 LLM 都是 64 或 128)
    if head_dim % TILE_D_CHUNK != 0:
        raise ValueError(f"Head Dim {head_dim} must be divisible by {TILE_D_CHUNK}")

    out = torch.empty((batch, head_num, head_dim), dtype=mid_o.dtype, device=mid_o.device)
    out_view = out.view(batch, head_num, 1, head_dim)
    
    grid = (batch, head_num, 1)

    # 准备 Kernel 参数
    HEAD_DIM = head_dim
    TOTAL_BLOCKS = int(num_blocks) # 强制转换为 int
    
    # 启动 Kernel
    # 在这里我们将全局定义的 TILE_D_CHUNK 传进去
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        flash_decode_stage2_kernel_split_d,
        (mid_o, mid_o_lse, b_seqlen, out_view, HEAD_DIM, block_seq, TILE_D_CHUNK, TOTAL_BLOCKS)
    )
    
    return out
# def run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor, block_size: int = None):

#     if isinstance(block_seq_tensor, torch.Tensor):
#         block_seq = block_seq_tensor.item()
#     else:
#         block_seq = block_seq_tensor

#     batch, head_num, num_blocks, head_dim = mid_o.shape
#     out = torch.empty((batch, head_num, head_dim), dtype=mid_o.dtype, device=mid_o.device)
    

#     out_view = out.view(batch, head_num, 1, head_dim)
#     grid = (batch, head_num, 1)

#     HEAD_DIM = head_dim
#     TOTAL_BLOCKS = num_blocks
    
#     ct.launch(
#         torch.cuda.current_stream(),
#         grid,
#         flash_decode_stage2_kernel,
#         (mid_o, mid_o_lse, b_seqlen, out_view, HEAD_DIM, block_seq, TOTAL_BLOCKS)
#     )
    
#     return out

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
    nvtx.push_range("FlashDecodeStage2_cuTile")
    run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
    torch.cuda.synchronize()
    nvtx.pop_range()
    print("Done.")