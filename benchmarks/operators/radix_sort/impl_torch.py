import torch


def run(input: torch.Tensor, N: int, **kwargs):
    # torch.sort/torch.topk both lower to XLA's `sort` op, which Neuron does not
    # support at all ([NCC_EVRF029] Operation sort is not supported on trn2) --
    # so the honest baseline has to sort via only supported primitives (compare,
    # gather, where), same approach as bitonic_sort's torch baseline.
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
