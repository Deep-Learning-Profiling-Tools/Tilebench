import torch
from torch.nn.functional import scaled_dot_product_attention

def run(q, k, v, causal=True, **kwargs):
    # PyTorch SDPA 支持 Flash Attention 后端
    # is_causal 参数对应 causal masking
    return scaled_dot_product_attention(q, k, v, is_causal=causal)