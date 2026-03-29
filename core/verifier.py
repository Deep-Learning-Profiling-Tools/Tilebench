import torch

# Per-dtype tolerances for correctness verification (atol, rtol).
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


def _verify_single(
    output: torch.Tensor,
    reference: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
) -> tuple[bool, str]:
    default_atol, default_rtol = _TOLERANCES.get(output.dtype, (_DEFAULT_ATOL, _DEFAULT_RTOL))
    atol = atol if atol is not None else default_atol
    rtol = rtol if rtol is not None else default_rtol
    try:
        torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
        return True, ""
    except Exception as e:
        return False, str(e)


def verify(
    output,
    reference,
    atol: float | None = None,
    rtol: float | None = None,
) -> tuple[bool, str]:
    if isinstance(output, (tuple, list)):
        for i, (o, r) in enumerate(zip(output, reference)):
            ok, err = _verify_single(o, r, atol=atol, rtol=rtol)
            if not ok:
                return False, f"output[{i}]: {err}"
        return True, ""
    return _verify_single(output, reference, atol=atol, rtol=rtol)
