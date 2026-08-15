import torch


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    M = 1 << ((N - 1).bit_length())
    work = torch.full((M,), float("inf"), dtype=data.dtype, device=data.device)
    work[:N] = data

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
