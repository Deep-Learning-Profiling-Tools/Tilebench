"""Reference fused element-wise activation: out = silu(x * gate + bias).

silu(z) = z * sigmoid(z) -- modern LLM activation (Llama / Mistral).
"""
import torch
import torch.nn.functional as F


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    return F.silu(x * gate + bias)
