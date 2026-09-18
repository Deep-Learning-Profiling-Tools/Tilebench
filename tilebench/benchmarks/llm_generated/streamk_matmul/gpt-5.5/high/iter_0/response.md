```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _zero_kernel(x_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(x_ptr + offs, z, mask=mask)


@triton.jit
def _cast_kernel(acc_ptr, out_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def _streamk_kernel(a_ptr, b_ptr, acc_ptr,
                    M, N, K,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    total_iters = num_tiles * iters_per_tile

    start_iter = (pid * total_iters) // num_progs
    end_iter = ((pid + 1) * total_iters) // num_progs

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    while start_iter < end_iter:
        tile_id = start_iter // iters_per_tile
        k_start = start_iter - tile_id * iters_per_tile
        tile_iter_end = tl.minimum(end_iter, (tile_id + 1) * iters_per_tile)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tile_in_group = tile_id - group_id * num_pid_in_group
        pid_m = first_pid_m + (tile_in_group % group_size_m)
        pid_n = tile_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + offs_m_base
        offs_n = pid_n * BLOCK_N + offs_n_base

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kk in tl.range(k_start, tile_iter_end, 1):
            offs_k = kk * BLOCK_K + offs_k_base

            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc, input_precision="tf32")

        c_offsets = offs_m[:, None] * N + offs_n[None, :]
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        if (k_start == 0) & (tile_iter_end == iters_per_tile):
            tl.store(acc_ptr + c_offsets, acc, mask=c_mask)
        else:
            tl.atomic_add(acc_ptr + c_offsets, acc, sem="relaxed", mask=c_mask)

        start_iter = tile_iter_end


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    total = M * N

    output = torch.empty((total,), device=a.device, dtype=a.dtype)
    accum = torch.empty((total,), device=a.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    NUM_SMS = 148
    num_warps = 8
    num_stages = 2

    ZERO_BLOCK_SIZE = 1024
    CAST_BLOCK_SIZE = 1024
    vec_warps = 4
    vec_stages = 2

    _zero_kernel[(triton.cdiv(total, ZERO_BLOCK_SIZE),)](
        accum, total,
        BLOCK_SIZE=ZERO_BLOCK_SIZE,
        num_warps=vec_warps,
        num_stages=vec_stages,
    )

    _streamk_kernel[(NUM_SMS,)](
        a, b, accum,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _cast_kernel[(triton.cdiv(total, CAST_BLOCK_SIZE),)](
        accum, output, total,
        BLOCK_SIZE=CAST_BLOCK_SIZE,
        num_warps=vec_warps,
        num_stages=vec_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "NUM_SMS": NUM_SMS,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "ZERO_BLOCK_SIZE": ZERO_BLOCK_SIZE,
        "CAST_BLOCK_SIZE": CAST_BLOCK_SIZE,
        "vec_warps": vec_warps,
        "vec_stages": vec_stages,
    })
    return output.view(M, N)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
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
```
