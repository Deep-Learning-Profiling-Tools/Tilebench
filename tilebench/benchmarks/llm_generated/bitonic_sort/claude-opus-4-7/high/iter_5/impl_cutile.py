import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _initial_local_sort_ct(data, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    base = pid * CHUNK
    x = ct.load(data, index=(pid,), shape=(CHUNK,))

    for kp in range(1, LOG2_CHUNK + 1):
        k_v = 1 << kp
        for jp in range(kp - 1, -1, -1):
            j = 1 << jp
            G = CHUNK // (2 * j)
            x_r = x.reshape((G, 2, j))
            lo = ct.extract(x_r, (0, 0, 0), (G, 1, j)).reshape((G, j))
            hi = ct.extract(x_r, (0, 1, 0), (G, 1, j)).reshape((G, j))
            mn = ct.minimum(lo, hi)
            mx = ct.maximum(lo, hi)
            group_offs = ct.arange(G, dtype=np.int32) * (2 * j) + base
            asc_g = (group_offs & k_v) == 0
            new_lo = ct.where(asc_g[:, None], mn, mx)
            new_hi = ct.where(asc_g[:, None], mx, mn)
            combined = ct.cat(
                (new_lo.reshape((G, 1, j)), new_hi.reshape((G, 1, j))),
                axis=1,
            )
            x = combined.reshape((CHUNK,))

    ct.store(data, index=(pid,), tile=x)


@ct.kernel
def _step_kernel_ct(data, k_val, j_val, BLOCK: ConstInt):
    pid = ct.bid(0)
    tid = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    i = (tid // j_val) * (2 * j_val) + (tid % j_val)
    ixj = i + j_val
    ai = ct.gather(data, i, check_bounds=False)
    aj = ct.gather(data, ixj, check_bounds=False)
    asc = (i & k_val) == 0
    mn = ct.minimum(ai, aj)
    mx = ct.maximum(ai, aj)
    ct.scatter(data, i, ct.where(asc, mn, mx), check_bounds=False)
    ct.scatter(data, ixj, ct.where(asc, mx, mn), check_bounds=False)


@ct.kernel
def _fused_local_merge_ct(data, k_val, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    base = pid * CHUNK
    x = ct.load(data, index=(pid,), shape=(CHUNK,))
    asc = (base & k_val) == 0  # scalar bool: uniform within chunk (since k_val > CHUNK)

    for jp in range(LOG2_CHUNK - 1, -1, -1):
        j = 1 << jp
        G = CHUNK // (2 * j)
        x_r = x.reshape((G, 2, j))
        lo = ct.extract(x_r, (0, 0, 0), (G, 1, j)).reshape((G, j))
        hi = ct.extract(x_r, (0, 1, 0), (G, 1, j)).reshape((G, j))
        mn = ct.minimum(lo, hi)
        mx = ct.maximum(lo, hi)
        new_lo = ct.where(asc, mn, mx)
        new_hi = ct.where(asc, mx, mn)
        combined = ct.cat(
            (new_lo.reshape((G, 1, j)), new_hi.reshape((G, 1, j))),
            axis=1,
        )
        x = combined.reshape((CHUNK,))

    ct.store(data, index=(pid,), tile=x)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    CHUNK = 1024
    LOG2_CHUNK = 10
    BLOCK_STEP = 1024

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, CHUNK, 2 * BLOCK_STEP)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_chunks = M // CHUNK
    n_pairs = M // 2
    n_step_blocks = max(n_pairs // BLOCK_STEP, 1)

    ct.launch(stream, (n_chunks, 1, 1),
              _initial_local_sort_ct, (work, CHUNK, LOG2_CHUNK))

    k = 2 * CHUNK
    while k <= M:
        j = k // 2
        while j >= CHUNK:
            ct.launch(stream, (n_step_blocks, 1, 1),
                      _step_kernel_ct, (work, k, j, BLOCK_STEP))
            j //= 2
        ct.launch(stream, (n_chunks, 1, 1),
                  _fused_local_merge_ct, (work, k, CHUNK, LOG2_CHUNK))
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({"CHUNK": CHUNK, "BLOCK_STEP": BLOCK_STEP})
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
