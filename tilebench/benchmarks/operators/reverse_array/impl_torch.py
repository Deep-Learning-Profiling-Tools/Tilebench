import torch

# Neuron/XLA: measured device time of the two lowerings (fp16), neither is
# bandwidth-bound and neither is monotone in N:
#   N      1M     5M     10M    20M
#   flip   8.4    41.7   96     51   ms
#   gather 1.2    3.0    100    200  ms
_XLA_GATHER_MAX = 8_000_000


def run(input: torch.Tensor, N: int, **kwargs):
    if input.device.type == "xla" and N <= _XLA_GATHER_MAX:
        # An explicit index gather of the mirrored positions is 5-14x faster than
        # torch.flip's reverse lowering up to a few million elements (see table).
        idx = torch.arange(N - 1, -1, -1, device=input.device)
        return input.reshape(-1)[idx]
    return input.flip(0).contiguous()
