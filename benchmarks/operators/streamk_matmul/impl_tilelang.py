import math

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "threads": 256,
    "num_stages": 4,
}
_last_autotune_config: dict = {}


def streamk_configs():
    return [
        dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=32, GROUP_M=8, threads=nt, num_stages=3)
        for bm in [64, 128]
        for bn in [128, 256]
        for nt in [128, 256]
    ]


def _streamk_partition(M: int, N: int, BLOCK_M: int, BLOCK_N: int, NUM_SMS: int):
    total_tiles = math.ceil(M / BLOCK_M) * math.ceil(N / BLOCK_N)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    return total_tiles, streamk_tiles


@tilelang.autotune(configs=streamk_configs(), warmup=3, rep=10, timeout=60)
@tilelang.jit
def first_wave_kernel(
    A,
    B,
    C,
    dtype,
    NUM_SMS: int,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
    GROUP_M: int = 8,
    threads: int = 256,
    num_stages: int = 4,
):
    M, K, N = T.const("M, K, N")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), "float32")

    grid_m = T.ceildiv(M, BLOCK_M)
    grid_n = T.ceildiv(N, BLOCK_N)
    total_tiles = grid_m * grid_n
    iters_per_tile = T.ceildiv(K, BLOCK_K)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS

    total_iters_streamk = streamk_tiles * iters_per_tile
    total_full_iters = total_iters_streamk // NUM_SMS
    total_partial_iters = total_iters_streamk % NUM_SMS

    with T.Kernel(NUM_SMS, threads=threads) as pid:
        a_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

        start_iter = T.alloc_var("int32")
        last_iter = T.alloc_var("int32")
        end_iter = T.alloc_var("int32")
        tile_id = T.alloc_var("int32")
        iter_in_tile = T.alloc_var("int32")
        pid_m = T.alloc_var("int32")
        pid_n = T.alloc_var("int32")
        group_id = T.alloc_var("int32")
        first_pid_m = T.alloc_var("int32")
        group_size_m = T.alloc_var("int32")

        start_iter = pid * total_full_iters + T.min(pid, total_partial_iters)
        last_iter = (pid + 1) * total_full_iters + T.min(pid + 1, total_partial_iters)

        while start_iter < last_iter:
            end_iter = T.min(
                start_iter + (iters_per_tile - start_iter % iters_per_tile),
                last_iter,
            )
            tile_id = start_iter // iters_per_tile

            group_id = tile_id // (GROUP_M * grid_n)
            first_pid_m = group_id * GROUP_M
            group_size_m = T.min(grid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % (GROUP_M * grid_n)) // group_size_m

            T.clear(acc)
            iter_in_tile = start_iter % iters_per_tile
            for k_iter in T.Pipelined(end_iter - start_iter, num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, (iter_in_tile + k_iter) * BLOCK_K], a_shared)
                T.copy(B[(iter_in_tile + k_iter) * BLOCK_K, pid_n * BLOCK_N], b_shared)
                T.gemm(a_shared, b_shared, acc)

            T.atomic_add(C[pid_m * BLOCK_M, pid_n * BLOCK_N], acc)
            start_iter = end_iter


@tilelang.jit
def full_tiles_kernel(
    A,
    B,
    C,
    dtype,
    NUM_SMS: int,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
    GROUP_M: int = 8,
    threads: int = 256,
    num_stages: int = 4,
):
    M, K, N = T.const("M, K, N")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), "float32")

    grid_m = T.ceildiv(M, BLOCK_M)
    grid_n = T.ceildiv(N, BLOCK_N)
    total_tiles = grid_m * grid_n
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS

    blocking_tiles = total_tiles - streamk_tiles

    with T.Kernel(blocking_tiles, threads=threads) as pid:
        a_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

        tile_id = pid + streamk_tiles
        group_id = tile_id // (GROUP_M * grid_n)
        first_pid_m = group_id * GROUP_M
        group_size_m = T.min(grid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % (GROUP_M * grid_n)) // group_size_m

        T.clear(acc)
        for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
            T.copy(A[pid_m * BLOCK_M, k_tile * BLOCK_K], a_shared)
            T.copy(B[k_tile * BLOCK_K, pid_n * BLOCK_N], b_shared)
            T.gemm(a_shared, b_shared, acc)

        T.copy(acc, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


def run(a: torch.Tensor, b: torch.Tensor,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype

    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    dtype = T.tfloat32 if a.dtype == torch.float32 else str(a.dtype).removeprefix("torch.")

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if a.dtype == torch.float32:
        c = out
        c.zero_()
    else:
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    if autotune:
        with set_autotune_inputs(a, b, c):
            first_kernel = first_wave_kernel.compile(a, b, c, dtype=dtype, NUM_SMS=NUM_SMS)
        cfg = dict(first_kernel.config or {})
        _last_autotune_config.clear()
        _last_autotune_config.update(cfg)
        c.zero_()
        first_kernel(a, b, c)
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
        # Unlike Triton, TileLang doesn't shrink the pipeline when the staged
        # A/B tiles exceed the SM's shared-memory budget (128 KB needed for
        # fp32 at ns=4 vs e.g. 99 KB on sm_89). Clamp stages to what fits.
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        smem_budget = getattr(props, "shared_memory_per_block_optin", 99 * 1024)
        stage_bytes = (cfg["BLOCK_M"] + cfg["BLOCK_N"]) * cfg["BLOCK_K"] * a.element_size()
        cfg["num_stages"] = max(1, min(cfg["num_stages"], smem_budget // stage_bytes))
        first_wave_kernel(
            a,
            b,
            c,
            dtype,
            NUM_SMS=NUM_SMS,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=cfg["GROUP_M"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    total_tiles, streamk_tiles = _streamk_partition(M, N, cfg["BLOCK_M"], cfg["BLOCK_N"], NUM_SMS)
    blocking_tiles = total_tiles - streamk_tiles
    if blocking_tiles > 0:
        full_tiles_kernel(
            a,
            b,
            c,
            dtype,
            NUM_SMS=NUM_SMS,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=cfg["GROUP_M"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    if c is not out:
        out.copy_(c)
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
