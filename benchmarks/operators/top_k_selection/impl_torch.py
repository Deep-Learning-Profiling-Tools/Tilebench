import torch


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    # Index [0] instead of .values: torch.topk returns a namedtuple on CPU but
    # a plain list under torch_xla on Neuron, which has no .values attribute.
    return torch.topk(input.contiguous(), k, largest=True, sorted=True)[0]
