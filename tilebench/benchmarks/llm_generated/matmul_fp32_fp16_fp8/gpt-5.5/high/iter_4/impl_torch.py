import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference matmul.

    For fp8 dtypes Torch has no native fp8 × fp8 matmul on every backend, so
    we cast to fp32, do the matmul, then cast back to the input dtype —
    matches what the Triton / cuTile kernels do (fp32 accumulator, cast
    output at the end).
    """
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        return out.to(a.dtype)
    return torch.matmul(a, b)
