import torch
import torch.nn.functional as F


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    """
    Reference L2 normalisation (per last dimension).
    Computed in fp32 for accuracy, cast back to input dtype.
    """
    return F.normalize(x.float(), p=2, dim=-1, eps=eps).to(x.dtype)
