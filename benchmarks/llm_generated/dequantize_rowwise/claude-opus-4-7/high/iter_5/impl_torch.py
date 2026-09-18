"""Reference dequantize_rowwise: out[r, c] = state_x[r] * x[r, c] / 127."""
import torch


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    return (state_x.unsqueeze(1) * x.to(torch.float32) * (1.0 / 127.0)).to(torch.float16)


def get_last_config() -> dict | None:
    return None
