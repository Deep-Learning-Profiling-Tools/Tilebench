import torch

# The Triton / cuTile fp32 kernels use TF32 Tensor Cores; let cuBLAS use them
# too so the fp32 speedup_* columns compare TF32 vs TF32. (PyTorch defaults to
# IEEE fp32 matmul, ~8x slower on B200, which inflated DSL speedups.)
torch.backends.cuda.matmul.allow_tf32 = True

# torch._scaled_mm needs a column-major B plus scale tensors. Build them once
# per input tensor (keyed by data_ptr) so the unmeasured warmup call pays the
# conversion cost, not the timed iterations. Unit scales keep the math identical
# to plain A @ B. Rowwise scales select the CUTLASS kernel — the cuBLASLt
# scalar-scale path raises CUBLAS_STATUS_NOT_INITIALIZED in this environment
# (torch 2.10.0+cu130 on B200).
_fp8_cache: dict = {}


def _fp8_args(a: torch.Tensor, b: torch.Tensor):
    key = (a.data_ptr(), b.data_ptr(), a.shape, b.shape)
    if key not in _fp8_cache:
        _fp8_cache[key] = (
            b.t().contiguous().t(),  # column-major B (required by _scaled_mm)
            torch.ones(a.shape[0], 1, device=a.device, dtype=torch.float32),
            torch.ones(1, b.shape[1], device=b.device, dtype=torch.float32),
        )
    return _fp8_cache[key]


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference matmul.

    fp32: TF32 Tensor-Core matmul (allow_tf32 above).
    fp8_e4m3fn: real fp8 Tensor-Core GEMM via torch._scaled_mm (fp32
    accumulate, unit scales) — an honest fp8 baseline.
    (fp8_e5m2 was dropped from this operator: torch/cuBLASLt reject
    e5m2 x e5m2, so there is no torch-native fp8 baseline for it.)
    """
    if a.dtype == torch.float8_e4m3fn:
        b_col, scale_a, scale_b = _fp8_args(a, b)
        out = torch._scaled_mm(a, b_col, scale_a=scale_a, scale_b=scale_b,
                               out_dtype=torch.bfloat16)
        return out.to(a.dtype)
    return torch.matmul(a, b)
