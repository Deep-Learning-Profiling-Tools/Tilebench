import torch


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    if x.device.type == "xla" and x.dtype == torch.int8:
        t = x.to(torch.int16).transpose(0, 1).contiguous()
        return torch.clamp(t, -128, 127).to(torch.int8)
    return x.transpose(0, 1).contiguous()
