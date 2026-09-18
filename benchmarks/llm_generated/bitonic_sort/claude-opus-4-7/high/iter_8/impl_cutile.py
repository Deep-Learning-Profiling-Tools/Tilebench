import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _local_sort(data, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    tile = ct.load(data, index=(pid,), shape=(CHUNK,))
    base = pid * CHUNK

    for kp in range(1, LOG2_CHUNK + 1):
        K_VAL = 1 << kp
        for jp in range(kp - 1, -1, -1):
            J_VAL = 1 << jp
            TWO_J = J_VAL * 2
            NUM_G = CHUNK // TWO_J

            t3 = ct.reshape(tile, (NUM_G, 2, J_VAL))
            low = ct.extract(t3, (0, 0, 0), shape=(NUM_G, 1, J_VAL))
            high = ct.extract(t3, (0, 1, 0), shape=(NUM_G, 1, J_VAL))
            mn = ct.minimum(low, high)
            mx = ct.maximum(low, high)

            g_arange = ct.arange(NUM_G, dtype=np.int32)
            offs_g = base + g_arange * TWO_J
            asc = (offs_g & K_VAL) == 0
            asc_3d = ct.reshape(asc, (NUM_G, 1, 1))
            asc_b = ct.broadcast_to(asc_3d, (NUM_G, 1, J_VAL))

            new_low = ct.where(asc_b, mn, mx)
            new_high = ct.where(asc_b, mx, mn)
            combined = ct.cat((new_low, new_high), axis=1)
            tile = ct.reshape(combined, (CHUNK,))

    ct.store(data, index=(pid,), tile=tile)


@ct.kernel
def _bitonic_step(data, k, j, BLOCK: ConstInt):
    pid = ct.bid(0)
    tid = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    i = (tid // j) * (2 * j) + (tid % j)
    ixj = i + j

    ai = ct.gather(data, i, check_bounds=False)
    aj = ct.gather(data, ixj, check_bounds=False)
    asc = (i & k) == 0
    mn = ct.minimum(ai, aj)
    mx = ct.maximum(ai, aj)
    ct.scatter(data, i, ct.where(asc, mn, mx), check_bounds=False)
    ct.scatter(data, ixj, ct.where(asc, mx, mn), check_bounds=False)


@ct.kernel
def _local_merge(data, k_val, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    tile = ct.load(data, index=(pid,), shape=(CHUNK,))
    base = pid * CHUNK

    for jp in range(LOG2_CHUNK - 1, -1, -1):
        J_VAL = 1 << jp
        TWO_J = J_VAL * 2
        NUM_G = CHUNK // TWO_J

        t3 = ct.reshape(tile, (NUM_G, 2, J_VAL))
        low = ct.extract(t3, (0, 0, 0), shape=(NUM_G, 1, J_VAL))
        high = ct.extract(t3, (0, 1, 0), shape=(NUM_G, 1, J_VAL))
        mn = ct.minimum(low, high)
        mx = ct.maximum(low, high)

        g_arange = ct.arange(NUM_G, dtype=np.int32)
        offs_g = base + g_arange * TWO_J
        asc = (offs_g & k_val) == 0
        asc_3d = ct.reshape(asc, (NUM_G, 1, 1))
        asc_b = ct.broadcast_to(asc_3d, (NUM_G, 1, J_VAL))

        new_low = ct.where(asc_b, mn, mx)
        new_high = ct.where(asc_b, mx, mn)
        combined = ct.cat((new_low, new_high), axis=1)
        tile = ct.reshape(combined, (CHUNK,))

    ct.store(data, index=(pid,), tile=tile)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    CHUNK = 1024
    LOG2_CHUNK = 10
    BLOCK_STEP = 1024

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, 2 * CHUNK)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_chunks = M // CHUNK
    n_pairs = M // 2
    grid_local = (n_chunks, 1, 1)
    grid_step = (max(n_pairs // BLOCK_STEP, 1), 1, 1)

    ct.launch(stream, grid_local, _local_sort, (work, CHUNK, LOG2_CHUNK))

    k = 2 * CHUNK
    while k <= M:
        j = k // 2
        while j >= CHUNK:
            ct.launch(stream, grid_step, _bitonic_step, (work, k, j, BLOCK_STEP))
            j //= 2
        ct.launch(stream, grid_local, _local_merge, (work, k, CHUNK, LOG2_CHUNK))
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "CHUNK": CHUNK,
        "BLOCK_STEP": BLOCK_STEP,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
