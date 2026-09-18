import torch

def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):


    if isinstance(block_seq, torch.Tensor):
        block_seq = block_seq.item()


    batch_size, num_heads, num_blocks, head_dim = mid_o.shape


    valid_blocks_count = (b_seqlen + block_seq - 1) // block_seq


    block_indices = torch.arange(num_blocks, device=mid_o.device).view(1, 1, -1)


    valid_blocks_expanded = valid_blocks_count.view(-1, 1, 1)


    mask = block_indices < valid_blocks_expanded


    masked_lse = mid_o_lse.clone()
    masked_lse = masked_lse.masked_fill(~mask, -float('inf'))


    global_max_lse = torch.max(masked_lse, dim=2, keepdim=True)[0]


    weights = torch.exp(masked_lse - global_max_lse)


    weights = weights.masked_fill(~mask, 0.0)


    weighted_values = mid_o * weights.unsqueeze(-1)
    numerator = torch.sum(weighted_values, dim=2)


    denominator = torch.sum(weights, dim=2, keepdim=True)


    output = numerator / (denominator + 1e-10)

    return output
