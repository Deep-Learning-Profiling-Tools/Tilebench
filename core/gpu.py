"""GPU model tag used to segregate results by hardware (results/<TAG>/...)."""

import os
import re

_GPU_MODEL_PATTERNS = [
    # Data center GPUs: B200, H100, H200, A100, A800, V100, P100, L40S.
    r"\b[BHAVLP]\d{2,4}[A-Z]*\b",
    # RTX GPUs: RTX 4090, RTX4090, RTX 6000 Ada.
    r"\bRTX\s*\d{4}(?:\s*Ada)?\b",
]


def gpu_label(gpu_name: str) -> str:
    """Short model label from a full device name.

    e.g. 'NVIDIA A100-SXM4-80GB' -> 'A100', 'NVIDIA B200' -> 'B200'.
    """
    for pattern in _GPU_MODEL_PATTERNS:
        model = re.search(pattern, gpu_name, flags=re.IGNORECASE)
        if model:
            return re.sub(r"\s+", "", model.group(0)).upper()

    label = re.sub(r"^NVIDIA\s+", "", gpu_name, flags=re.IGNORECASE)
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return label or "GPU"


def gpu_tag() -> str:
    """Label of the GPU results belong to, i.e. the results/<TAG>/ directory.

    Resolution order: TILEBENCH_GPU_TAG env var, then the current CUDA device.
    """
    tag = os.environ.get("TILEBENCH_GPU_TAG")
    if tag:
        return tag
    try:
        import torch
        name = torch.cuda.get_device_name(0)
    except Exception as e:
        raise RuntimeError(
            "Cannot detect GPU model for results/<GPU>/ paths; "
            "set TILEBENCH_GPU_TAG (e.g. TILEBENCH_GPU_TAG=B200) or pass an "
            "explicit GPU/output option."
        ) from e
    return gpu_label(name)
