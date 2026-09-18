import torch
from torch.nn.functional import scaled_dot_product_attention

def run(q, k, v, causal=True, **kwargs):
    return scaled_dot_product_attention(q, k, v, is_causal=causal)
