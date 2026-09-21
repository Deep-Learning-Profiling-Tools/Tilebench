import torch

# Neuron/XLA: the compiler's TopK lowering allocates an SBUF work area proportional to N * k
# and fails to compile beyond N * k = 2**28 (N=2**20, k=1024: "TopKImpl ... Allocated memory out
# of bound"). Larger problems are split into chunks whose top-k are merged by a second top-k,
# which yields exactly the same values.
_XLA_TOPK_MAX_NK = 1 << 28


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    if input.device.type == "xla":
        x = input.contiguous()
        if N * k > _XLA_TOPK_MAX_NK:
            chunk = max(k, _XLA_TOPK_MAX_NK // k)
            parts = [torch.topk(x[i:i + chunk], min(k, x[i:i + chunk].numel()), largest=True, sorted=True)[0]
                     for i in range(0, N, chunk)]
            x = torch.cat(parts)
        return torch.topk(x, k, largest=True, sorted=True)[0]
    return torch.topk(input.contiguous(), k, largest=True, sorted=True).values
