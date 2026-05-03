import torch

def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    """
    PyTorch reference implementation for Flash Decode Stage 2.
    Performs a weighted reduction of partial attention outputs based on LogSumExp.
    
    Args:
        mid_o: Partial outputs from Stage 1. Shape: [Batch, Heads, Num_Blocks, HeadDim]
        mid_o_lse: Partial LogSumExp from Stage 1. Shape: [Batch, Heads, Num_Blocks]
        b_seqlen: Actual sequence lengths. Shape: [Batch]
        block_seq: Block size used in Stage 1 partitioning (scalar or 0-d Tensor).
        block_size: Ignored (placeholder for benchmark framework compatibility).
        
    Returns:
        Final output tensor of shape [Batch, Heads, HeadDim]
    """
    
    # Handle block_seq input (could be passed as a Tensor by the generator)
    if isinstance(block_seq, torch.Tensor):
        block_seq = block_seq.item()
        
    # Get dimensions
    batch_size, num_heads, num_blocks, head_dim = mid_o.shape
    
    # 1. Create Validity Mask
    # Calculate how many blocks are actually valid for each batch index
    # Formula: ceil(seq_len / block_seq)
    valid_blocks_count = (b_seqlen + block_seq - 1) // block_seq
    
    # Create indices for blocks [0, 1, ..., NumBlocks-1]
    # Shape: [1, 1, NumBlocks]
    block_indices = torch.arange(num_blocks, device=mid_o.device).view(1, 1, -1)
    
    # Expand valid_blocks_count to allow broadcasting
    # Shape: [Batch, 1, 1]
    valid_blocks_expanded = valid_blocks_count.view(-1, 1, 1)
    
    # Generate mask: True if the block is valid, False otherwise
    # Shape: [Batch, 1, NumBlocks] -> Broadcasts to [Batch, Heads, NumBlocks]
    mask = block_indices < valid_blocks_expanded
    
    # 2. Mask LogSumExp (LSE)
    # Set LSE of invalid blocks to -inf so they don't impact the Global Max or Sum
    # Clone to avoid modifying the input tensor in-place
    masked_lse = mid_o_lse.clone()
    masked_lse = masked_lse.masked_fill(~mask, -float('inf'))
    
    # 3. Compute Global Max LSE (for numerical stability)
    # Find the max LSE across all blocks for each head
    # Shape: [Batch, Heads, 1]
    global_max_lse = torch.max(masked_lse, dim=2, keepdim=True)[0]
    
    # 4. Compute Weights
    # Calculate exp(LSE_i - GlobalMax)
    # Shape: [Batch, Heads, NumBlocks]
    weights = torch.exp(masked_lse - global_max_lse)
    
    # Explicitly zero out weights for invalid blocks (redundant if LSE is -inf, but safe)
    weights = weights.masked_fill(~mask, 0.0)
    
    # 5. Weighted Sum of Partial Outputs (Numerator)
    # Mid_O: [Batch, Heads, NumBlocks, HeadDim]
    # Weights: [Batch, Heads, NumBlocks] -> Unsqueeze to [Batch, Heads, NumBlocks, 1]
    # Result: Sum across the NumBlocks dimension (dim 2)
    weighted_values = mid_o * weights.unsqueeze(-1)
    numerator = torch.sum(weighted_values, dim=2)
    
    # 6. Sum of Weights (Denominator)
    # Shape: [Batch, Heads, 1]
    denominator = torch.sum(weights, dim=2, keepdim=True)
    
    # 7. Final Normalization
    # Output = Numerator / Denominator
    # Add a small epsilon to denominator to prevent division by zero in empty sequences (edge case)
    output = numerator / (denominator + 1e-10)
    
    return output