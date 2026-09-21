import torch


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if x.device.type == "xla" and x.dtype != torch.float32:
        # Neuron/XLA: ``mean(dtype=float32)`` on a half-precision input accumulates in the
        # input precision (max error ~3e-5 vs the fp32 reference, failing verification);
        # widening first gives the fp32-accumulated result the non-XLA path computes.
        return x.to(torch.float32).mean(dim=dim)
    return x.mean(dim=dim, dtype=torch.float32)
