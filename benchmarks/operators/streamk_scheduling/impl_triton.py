import torch


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024, **kwargs):
    del kwargs
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("streamk_scheduling expects 2D inputs.")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matmul.")

    k = a.shape[1]
    chunk = max(1, min(k, block_size))
    out = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    for start in range(0, k, chunk):
        end = min(start + chunk, k)
        out += torch.matmul(a[:, start:end].to(torch.float32), b[start:end, :].to(torch.float32))
    return out
