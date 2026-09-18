import torch

def run(x):
    return torch.softmax(x, dim=-1)
