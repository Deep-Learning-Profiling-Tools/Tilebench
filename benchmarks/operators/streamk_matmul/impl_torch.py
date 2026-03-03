import torch

def run(a, b, **kwargs):
    """
    Reference implementation for Matrix Multiplication (Stream-K target).
    """
    orig_dtype = a.dtype
    
    # Safe handling for FP8 if necessary
    if orig_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        return out.to(orig_dtype)
        
    return torch.matmul(a, b)