import torch

# Per-dtype tolerances for correctness verification.
# Values match torch.testing.assert_close built-in defaults.
_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32:  (1e-5,  1.3e-6),
    torch.float16:  (1e-3,  1e-3),
    torch.bfloat16: (1e-2,  1.6e-2),
    torch.int8:     (0,     0),
    torch.int16:    (0,     0),
    torch.int32:    (0,     0),
    torch.int64:    (0,     0),
}

# Fallback for dtypes not listed above (e.g. future fp8 support).
_DEFAULT_ATOL = 1e-2
_DEFAULT_RTOL = 1e-2


def verify(output: torch.Tensor, reference: torch.Tensor) -> tuple[bool, str]:
    atol, rtol = _TOLERANCES.get(output.dtype, (_DEFAULT_ATOL, _DEFAULT_RTOL))
    try:
        torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
        return True, ""
    except Exception as e:
        return False, str(e)
