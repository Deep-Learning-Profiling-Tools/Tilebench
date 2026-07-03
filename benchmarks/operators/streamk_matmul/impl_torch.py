"""Reference matmul (Stream-K target). Plain torch.matmul.

TF32 is enabled for the fp32 case so the baseline's precision path matches
the Triton kernel (`input_precision="tf32"`) and the cuTile kernel
(`ct.tfloat32` cast) — same policy as matmul_fp32_fp16_fp8's baseline.
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    return torch.matmul(a, b)


def get_last_config() -> dict | None:
    return None
