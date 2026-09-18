import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _zero_kernel(accum, TILE: ConstInt):
    bid = ct.bid(0)
    z = ct.zeros((TILE,), dtype=np.float32)
    ct.store(accum, index=(bid,), tile=z)


@ct.kernel
def _cast_kernel(accum, out, TILE: ConstInt):
    bid = ct.bid(0)
    x = ct.load(accum, index=(bid,), shape=(TILE,),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False)
    ct.store(out, index=(bid,), tile=ct.astype(x, out.dtype))


@ct.kernel
def _streamk_split_kernel(a, b, accum,
                          M: ConstInt, N: ConstInt,
                          K_TILES: ConstInt,
                          CHUNK_TILES: ConstInt,
                          SPLIT_K: ConstInt,
                          TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid = ct.bid(0)

    split_id = bid % SPLIT_K
    tile_id = bid // SPLIT_K

    num_pid_n = ct.cdiv(N, TN)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id - pid_m * num_pid_n

    offs_m = pid_m * TM + ct.arange(TM, dtype=np.int32)[:, None]
    offs_n = pid_n * TN + ct.arange(TN, dtype=np.int32)[None, :]
    flat_idx = offs_m * N + offs_n
    valid = (offs_m < M) & (offs_n < N)
    flat_idx = ct.where(valid, flat_idx, M * N)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    k0 = split_id * CHUNK_TILES

    for j in range(0, CHUNK_TILES):
        kt = k0 + j
        a_tile = ct.load(a, index=(pid_m, kt), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(kt, pid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    ct.atomic_add(accum, flat_idx, acc,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.DEVICE)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    total = m * n

    output = torch.empty((total,), device=a.device, dtype=a.dtype)
    accum = torch.empty((total,), device=a.device, dtype=torch.float32)

    TM = 128
    TN = 128
    TK = 64
    SPLIT_K = 2
    k_tiles = (k + TK - 1) // TK
    chunk_tiles = (k_tiles + SPLIT_K - 1) // SPLIT_K

    VEC_TILE = 1024
    vec_occupancy = 8
    mma_occupancy = 1

    stream = torch.cuda.current_stream()

    zero_kernel = _zero_kernel.with_hints(occupancy=vec_occupancy)
    ct.launch(stream, ((total + VEC_TILE - 1) // VEC_TILE, 1, 1),
              zero_kernel, (accum, VEC_TILE))

    num_tiles_m = (m + TM - 1) // TM
    num_tiles_n = (n + TN - 1) // TN
    grid = (num_tiles_m * num_tiles_n * SPLIT_K, 1, 1)

    streamk_kernel = _streamk_split_kernel.with_hints(occupancy=mma_occupancy)
    ct.launch(stream, grid, streamk_kernel,
              (a, b, accum, m, n, k_tiles, chunk_tiles, SPLIT_K, TM, TN, TK))

    cast_kernel = _cast_kernel.with_hints(occupancy=vec_occupancy)
    ct.launch(stream, ((total + VEC_TILE - 1) // VEC_TILE, 1, 1),
              cast_kernel, (accum, output, VEC_TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM,
        "TN": TN,
        "TK": TK,
        "SPLIT_K": SPLIT_K,
        "CHUNK_TILES": chunk_tiles,
        "VEC_TILE": VEC_TILE,
        "mma_occupancy": mma_occupancy,
        "vec_occupancy": vec_occupancy,
    })
    return output.view(m, n)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
