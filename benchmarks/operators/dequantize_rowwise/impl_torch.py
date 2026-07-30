import torch


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):


    return (state_x.unsqueeze(1) * x * (1.0 / 127.0)).to(torch.float16)
