import torch

def run(x):
    # dim=-1 表示在最后一维（列）上进行 Softmax
    return torch.softmax(x, dim=-1)