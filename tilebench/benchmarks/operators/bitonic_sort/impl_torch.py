import torch


def _run_fixed_shape(data: torch.Tensor, N: int):
    M = 1 << ((N - 1).bit_length())
    work = torch.full((M,), float("inf"), dtype=data.dtype, device=data.device)
    work[:N] = data

    rows = {}
    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            R = M // (2 * j)
            if R not in rows:
                rows[R] = torch.arange(R, dtype=torch.int32, device=data.device).view(R, 1)
            pairs = work.view(R, 2, j)
            lo = torch.minimum(pairs[:, 0, :], pairs[:, 1, :])
            hi = torch.maximum(pairs[:, 0, :], pairs[:, 1, :])
            descending = torch.bitwise_and(rows[R], k // (2 * j)).bool()
            work = torch.stack((torch.where(descending, hi, lo),
                                torch.where(descending, lo, hi)), dim=1).reshape(M)
            j //= 2
        k *= 2

    return work[:N].contiguous()


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    if data.device.type == "neuron":
        return _run_fixed_shape(data, N)

    M = 1 << ((N - 1).bit_length())
    work = torch.full((M,), float("inf"), dtype=data.dtype, device=data.device)
    work[:N] = data

    offs = torch.arange(M, device=work.device, dtype=torch.int64)

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            ixj = offs ^ j
            active = ixj > offs
            ascending = (offs & k) == 0

            active_i = offs[active]
            active_j = ixj[active]
            a_vals = work[active_i]
            b_vals = work[active_j]
            asc = ascending[active]
            need_swap = torch.where(asc, a_vals > b_vals, a_vals < b_vals)

            swap_i = active_i[need_swap]
            swap_j = active_j[need_swap]

            a_keep = work[swap_i]
            b_keep = work[swap_j]
            work[swap_i] = b_keep
            work[swap_j] = a_keep

            j //= 2
        k *= 2

    return work[:N].contiguous()
