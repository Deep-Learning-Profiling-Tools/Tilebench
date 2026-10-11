import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _exchange_results(a, b, descending):
    swap = nl.ndarray(a.shape, dtype=nl.uint8, buffer=nl.sbuf)
    if descending:
        nisa.tensor_tensor(dst=swap, data1=a, data2=b, op=nl.less)
    else:
        nisa.tensor_tensor(dst=swap, data1=a, data2=b, op=nl.greater)

    result_a = nl.where(swap, b, a)
    result_b = nl.where(swap, a, b)
    return result_a, result_b


def _local_exchange(a, b, descending):
    result_a, result_b = _exchange_results(a, b, descending)
    nisa.tensor_copy(dst=a, src=result_a)
    nisa.tensor_copy(dst=b, src=result_b)


@nki.jit
def _initialize(data, work, N, B):
    M = work.shape[0]
    pid = nl.program_id(0)
    P = 128
    width = 2 * B
    tiles = M // (2 * P * width)
    output_view = work.reshape((2, tiles, P, width))

    for tile in range(tiles):
        base = pid * (M // 2) + tile * P * width
        values = nl.ndarray((P, width), dtype=data.dtype, buffer=nl.sbuf)

        with nl.no_reorder():
            nisa.memset(dst=values, value=float("inf"))

            valid = min(P * width, max(0, N - base))
            full_rows = valid // width
            remainder = valid - full_rows * width

            if full_rows > 0:
                source = data[
                    base:base + full_rows * width
                ].reshape((full_rows, width))
                nisa.dma_copy(dst=values[0:full_rows, :], src=source)

            if remainder > 0:
                start = base + full_rows * width
                source = data[start:start + remainder].reshape((1, remainder))
                nisa.dma_copy(
                    dst=values[full_rows:full_rows + 1, 0:remainder],
                    src=source,
                )

            k = 2
            while k <= B:
                j = k // 2
                while j > 0:
                    pattern = [
                        [width, P],
                        [2 * k, width // (2 * k)],
                        [2 * j, k // (2 * j)],
                        [1, j],
                    ]

                    a = values.ap(pattern=pattern, offset=0)
                    b = values.ap(pattern=pattern, offset=j)
                    _local_exchange(a, b, False)

                    a = values.ap(pattern=pattern, offset=k)
                    b = values.ap(pattern=pattern, offset=k + j)
                    _local_exchange(a, b, True)
                    j //= 2
                k *= 2

            nisa.dma_copy(dst=output_view[pid, tile, :, :], src=values)

    return work


def _global_tile(view, mode, pid, r, s, q, P, F, descending):
    if mode == 0:
        source_a = view[r, :, pid, s, 0, q, :]
        source_b = view[r, :, pid, s, 1, q, :]
    elif mode == 1:
        source_a = view[r, pid, s, :, 0, q, :]
        source_b = view[r, pid, s, :, 1, q, :]
    elif mode == 2:
        source_a = view[r, pid, s, 0, q, :, :]
        source_b = view[r, pid, s, 1, q, :, :]
    elif mode == 3:
        source_a = view[pid, r, :, 0, s, :]
        source_b = view[pid, r, :, 1, s, :]
    else:
        source_a = view[r, 0, pid, s, :, :]
        source_b = view[r, 1, pid, s, :, :]

    a = nl.ndarray((P, F), dtype=view.dtype, buffer=nl.sbuf)
    b = nl.ndarray((P, F), dtype=view.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=source_a)
    nisa.dma_copy(dst=b, src=source_b)

    result_a, result_b = _exchange_results(a, b, descending)
    nisa.dma_copy(dst=source_a, src=result_a)
    nisa.dma_copy(dst=source_b, src=result_b)


@nki.jit
def _global_stage(work, k, j, F):
    M = work.shape[0]
    pid = nl.program_id(0)
    Q = j // F
    descending = False

    if k < M:
        R = M // (2 * k)
        S = k // (2 * j)
        descending = pid == 1

        if R >= S and R >= Q:
            mode = 0
            P = min(128, R)
            view = work.reshape((R // P, P, 2, S, 2, Q, F))
            nr = R // P
            ns = S
            nq = Q
        elif S >= Q:
            mode = 1
            P = min(128, S)
            view = work.reshape((R, 2, S // P, P, 2, Q, F))
            nr = R
            ns = S // P
            nq = Q
        else:
            mode = 2
            P = min(128, Q)
            view = work.reshape((R, 2, S, 2, Q // P, P, F))
            nr = R
            ns = S
            nq = Q // P
    else:
        S = M // (2 * j)
        if S > Q:
            mode = 3
            P = min(128, S // 2)
            view = work.reshape((2, S // (2 * P), P, 2, Q, F))
            nr = S // (2 * P)
            ns = Q
            nq = 1
        else:
            mode = 4
            P = min(128, Q // 2)
            view = work.reshape((S, 2, 2, Q // (2 * P), P, F))
            nr = S
            ns = Q // (2 * P)
            nq = 1

    if nr >= ns and nr >= nq:
        for s in range(ns):
            for q in range(nq):
                def body_r(r):
                    _global_tile(
                        view, mode, pid, r, s, q, P, F, descending
                    )

                nl.fori_loop(0, nr, body_r)
    elif ns >= nq:
        for r in range(nr):
            for q in range(nq):
                def body_s(s):
                    _global_tile(
                        view, mode, pid, r, s, q, P, F, descending
                    )

                nl.fori_loop(0, ns, body_s)
    else:
        for r in range(nr):
            for s in range(ns):
                def body_q(q):
                    _global_tile(
                        view, mode, pid, r, s, q, P, F, descending
                    )

                nl.fori_loop(0, nq, body_q)

    return work


def _tail_tile(view, mode, pid, r, t, P, B, descending):
    if mode == 0:
        source = view[r, pid, t, :, :]
    elif mode == 1:
        source = view[r, :, pid, t, :]
    else:
        source = view[pid, r, :, :]

    values = nl.ndarray((P, B), dtype=view.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=values, src=source)

    with nl.no_reorder():
        j = B // 2
        while j > 0:
            pattern = [
                [B, P],
                [2 * j, B // (2 * j)],
                [1, j],
            ]
            a = values.ap(pattern=pattern, offset=0)
            b = values.ap(pattern=pattern, offset=j)
            _local_exchange(a, b, descending)
            j //= 2

        nisa.dma_copy(dst=source, src=values)


@nki.jit
def _local_tail(work, k, B):
    M = work.shape[0]
    pid = nl.program_id(0)
    descending = False

    if k < M:
        R = M // (2 * k)
        S = k // B
        descending = pid == 1

        if S >= R:
            mode = 0
            P = min(128, S)
            view = work.reshape((R, 2, S // P, P, B))
            nr = R
            nt = S // P
        else:
            mode = 1
            P = min(128, R)
            view = work.reshape((R // P, P, 2, S, B))
            nr = R // P
            nt = S
    else:
        mode = 2
        P = 128
        nr = M // (2 * P * B)
        nt = 1
        view = work.reshape((2, nr, P, B))

    if nr >= nt:
        for t in range(nt):
            def body_r(r):
                _tail_tile(view, mode, pid, r, t, P, B, descending)

            nl.fori_loop(0, nr, body_r)
    else:
        for r in range(nr):
            def body_t(t):
                _tail_tile(view, mode, pid, r, t, P, B, descending)

            nl.fori_loop(0, nt, body_t)

    return work


def run(data: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    if N <= 1:
        _LAST_CONFIG = {"algorithm": "copy", "length": N}
        return data.clone()

    M = 1 << ((N - 1).bit_length())
    B = min(16384, M // 512)
    F = 1024
    L = M.bit_length() - 1
    local_log = B.bit_length() - 1
    remaining_levels = L - local_log
    global_passes = remaining_levels * (remaining_levels + 1) // 2

    _LAST_CONFIG = {
        "algorithm": "strict_in_place_bitonic",
        "lnc": 2,
        "length": N,
        "padded_length": M,
        "local_block": B,
        "initial_partitions": 128,
        "initial_tiles_per_core": M // (512 * B),
        "global_free_width": F,
        "global_partition_limit": 128,
        "global_partition_axis": "largest_of_regions_pairs_chunks",
        "network_stages": L * (L + 1) // 2,
        "global_passes": global_passes,
        "local_tail_groups": remaining_levels,
        "synchronization": "kernel_boundaries",
        "kernel_launches": 1 + global_passes + remaining_levels,
    }

    # Every kernel operates on this same allocation. Kernel boundaries
    # globally order stages that cannot be confined to an on-chip row.
    work = torch.empty((M,), dtype=data.dtype, device=data.device)
    work = _initialize[2](data, work, N, B)

    k = 2 * B
    while k <= M:
        j = k // 2
        while j >= B:
            work = _global_stage[2](work, k, j, F)
            j //= 2

        work = _local_tail[2](work, k, B)
        k *= 2

    return work[:N]


def get_last_config() -> dict:
    return _LAST_CONFIG
