import torch


def run(input: torch.Tensor, N: int, **kwargs):
    if input.device.type != "xla":
        # CPU / CUDA reference and GPU baseline: unchanged native sort.
        return torch.sort(input).values
    # XLA/Neuron only: torch.sort lowers to XLA's `sort` op, which neuronx-cc
    # rejects ([NCC_EVRF029] Operation sort is not supported on trn2) -- so the
    # on-device baseline sorts via supported primitives only (compare, gather,
    # where), the same vectorised bitonic network as bitonic_sort's baseline.
    if N <= 1:
        return input.clone()

    M = 1 << ((N - 1).bit_length())
    pad_value = torch.iinfo(input.dtype).max
    work = torch.full((M,), pad_value, dtype=input.dtype, device=input.device)
    work[:N] = input

    offs = torch.arange(M, device=work.device, dtype=torch.int64)

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            ixj = offs ^ j
            is_lower = ixj > offs
            ascending = (offs & k) == 0

            partner = work[ixj]
            cmp = torch.where(ascending, work > partner, work < partner)
            # a pair (i, ixj[i]) must swap together; only the lower index's
            # comparison is meaningful, so gather it for the upper index too.
            decision = torch.where(is_lower, cmp, cmp[ixj])

            work = torch.where(decision, partner, work)

            j //= 2
        k *= 2

    return work[:N].contiguous()
