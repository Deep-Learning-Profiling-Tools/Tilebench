"""Reference dequantize_rowwise: out[r, c] = state_x[r] * x[r, c] / 127."""
import torch


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    # int8 x is read directly by the fused elementwise kernel (type promotion
    # to fp32 happens in-kernel) — the old x.to(torch.float32) materialised a
    # 4x-sized intermediate first. The final fp16 cast is the output contract.
    return (state_x.unsqueeze(1) * x * (1.0 / 127.0)).to(torch.float16)
