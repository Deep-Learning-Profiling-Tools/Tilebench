import torch

def run(a, b):
    """
    Reference implementation for Matrix Multiplication.
    Handles standard types and provides a safe fallback for FP8.
    """
    orig_dtype = a.dtype
    
    # PyTorch's native `torch.matmul` may require explicit scaling functions 
    # (e.g., `_scaled_mm`) for FP8 tensors depending on the version.
    # To match the Triton kernel's behavior (which accumulates in FP32 and casts back),
    # we compute the reference in FP32 and cast the result back.
    if orig_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        return out.to(orig_dtype)
    
    # For float16, bfloat16, float32, etc.
    return torch.matmul(a, b)