from __future__ import annotations

import torch

# Bytes per element for each dtype string key
_DTYPE_SIZE_MAP: dict[str, int] = {
    "float32": 4, "fp32": 4,
    "float16": 2, "fp16": 2,
    "bfloat16": 2, "bf16": 2,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "fp8": 1, "fp8_e5m2": 1, "fp8_e4m3fn": 1,
    "float8_e5m2": 1, "float8_e4m3fn": 1,
}


def dtype_size(dtype_name: str) -> int:
    """Return element size in bytes for a dtype string."""
    return _DTYPE_SIZE_MAP.get(dtype_name.lower(), 4)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.lower()
    aliases = {
        "float32": "float32",
        "fp32": "float32",
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "fp8": "float8_e5m2",
        "fp8_e5m2": "float8_e5m2",
        "fp8_e4m3fn": "float8_e4m3fn",
    }
    candidate = aliases.get(key, key)
    if hasattr(torch, candidate):
        return getattr(torch, candidate)

    # FP4 naming varies across PyTorch versions; try common candidates.
    if key in {"fp4", "float4"}:
        for name in ("float4_e2m1fn_x2", "float4_e2m1fn"):
            if hasattr(torch, name):
                return getattr(torch, name)
    raise ValueError(f"Unsupported dtype '{dtype_name}' for this environment.")
