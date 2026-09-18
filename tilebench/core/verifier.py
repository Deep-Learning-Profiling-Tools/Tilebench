import torch

# Per-dtype tolerances for correctness verification (atol, rtol).
_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32:        (1e-5,  1.3e-6),
    torch.float16:        (1e-3,  1e-3),
    torch.bfloat16:       (1e-2,  1.6e-2),
    torch.float8_e4m3fn:  (1.0,   0.1),
    torch.float8_e5m2:    (1.0,   0.1),
    torch.int8:           (0,     0),
    torch.int16:          (0,     0),
    torch.int32:          (0,     0),
    torch.int64:          (0,     0),
}

# Fallback for dtypes not listed above.
_DEFAULT_ATOL = 1e-2
_DEFAULT_RTOL = 1e-2

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _verify_single(
    output: torch.Tensor,
    reference: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
) -> tuple[bool, str]:
    default_atol, default_rtol = _TOLERANCES.get(output.dtype, (_DEFAULT_ATOL, _DEFAULT_RTOL))
    atol = atol if atol is not None else default_atol
    rtol = rtol if rtol is not None else default_rtol
    # torch.testing.assert_close refuses non-zero atol/rtol on fp8 dtypes
    # ("low dimensional floats"), so cast to fp32 for the comparison.
    if output.dtype in _FP8_DTYPES:
        output = output.to(torch.float32)
        reference = reference.to(torch.float32)
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
