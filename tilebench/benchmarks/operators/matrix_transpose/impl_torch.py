import torch


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    if x.device.type == "xla" and x.dtype == torch.int8:
        # Neuron/XLA: the 1-byte transpose lowering is lossy (~50% of the values come
        # back off by up to 2). Transposing as int16 is exact, but a plain cast back
        # to int8 is hoisted in front of the transpose by the compiler and hits the
        # same path; the clamp (a no-op on int8-range values) keeps the cast after it.
        t = x.to(torch.int16).transpose(0, 1).contiguous()
        return torch.clamp(t, -128, 127).to(torch.int8)
    return x.transpose(0, 1).contiguous()
