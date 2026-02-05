import torch
import cuda.tile as ct
import math

# 定义常量类型
ConstInt = ct.Constant[int]

@ct.kernel
def rope_kernel(
    Q,                  # Rank 4: [TotalTokens, Heads, 2, HalfDim]
    Cos,                # Rank 2: [SeqLen, HalfDim]
    Sin,                # Rank 2: [SeqLen, HalfDim]
    SeqLen: ConstInt,   # 序列长度
    TILE_DIM: ConstInt  # HalfDim
):
    # 1. 获取坐标
    row_id = ct.bid(0)   # Batch*Seq
    head_id = ct.bid(1)  # Heads
    
    # 2. 计算 Cos/Sin 索引
    seq_idx = row_id % SeqLen

    # 3. 加载 Cos/Sin
    # Cos/Sin 是 2D Tensor: (Seq, Dim)
    # 我们加载第 seq_idx 行，第 0 个 Tile。
    # Tile Shape 设为 (1, TILE_DIM)，显式匹配 2D Rank。
    # 结果 Tile 形状: (1, TILE_DIM)
    cos_tile = ct.load(Cos, index=(seq_idx, 0), shape=(1, TILE_DIM))
    sin_tile = ct.load(Sin, index=(seq_idx, 0), shape=(1, TILE_DIM))

    # 4. 加载 Q 的分量
    # Q 是 4D Tensor: (Total, Heads, 2, Dim)
    # Tile Shape 设为 (1, 1, 1, TILE_DIM)，显式匹配 4D Rank。
    # 结果 Tile 形状: (1, 1, 1, TILE_DIM)
    
    # 加载 x1 (第3维 index=0)
    q1 = ct.load(Q, index=(row_id, head_id, 0, 0), shape=(1, 1, 1, TILE_DIM))
    
    # 加载 x2 (第3维 index=1)
    q2 = ct.load(Q, index=(row_id, head_id, 1, 0), shape=(1, 1, 1, TILE_DIM))

    # 5. 计算
    out1 = q1 * cos_tile - q2 * sin_tile
    out2 = q2 * cos_tile + q1 * sin_tile

    # 6. 写回
    ct.store(Q, index=(row_id, head_id, 0, 0), tile=out1)
    ct.store(Q, index=(row_id, head_id, 1, 0), tile=out2)


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_size: int = None):
    # 1. 准备数据
    output = q.clone().contiguous()
    
    batch, seq_len, n_heads, head_dim = output.shape
    half_dim = head_dim // 2
    
    # 2. 创建 View
    
    # Q: [Batch*Seq, Heads, 2, HalfDim] (Rank 4)
    # 保持维度完整，方便 Kernel 里用 (r, h, 0, 0) 访问
    output_view = output.view(batch * seq_len, n_heads, 2, half_dim)
    
    # Cos/Sin: [Seq, HalfDim] (Rank 2)
    # 去掉中间的 1 维度，避免 rank 3 和 index 2 不匹配的问题
    cos_view = cos.view(seq_len, half_dim)
    sin_view = sin.view(seq_len, half_dim)

    # 3. 设置 Grid
    # Grid 覆盖 (Rows, Heads)
    grid = (batch * seq_len, n_heads, 1)
    
    # 4. 启动 Kernel
    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        rope_kernel, 
        (output_view, cos_view, sin_view, seq_len, half_dim)
    )
    
    return output