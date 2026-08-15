import math

import torch
import tilelang
import tilelang.language as T
from triton.testing import do_bench


_DEFAULT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "GROUP_M": 8,
    "threads": 256,
    "num_stages": 3,
}
_autotune_cache: dict = {}
_last_autotune_config: dict = {}


_WRONG = frozenset({
    (128, 256, 32, 256), (128, 256, 64, 128), (128, 256, 64, 256),
})


def streamk_configs():
    def produces_wrong_results(bm, bn, bk, nt):
        return (bm, bn, bk, nt) in _WRONG

    return [
        dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, threads=nt, num_stages=3)
        for bm in [64, 128]
        for bn in [128, 256]
        for bk in [32, 64]
        for nt in [128, 256]
        if not produces_wrong_results(bm, bn, bk, nt)
    ]


def _streamk_partition(M: int, N: int, BLOCK_M: int, BLOCK_N: int, NUM_SMS: int):
    total_tiles = math.ceil(M / BLOCK_M) * math.ceil(N / BLOCK_N)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    return total_tiles, streamk_tiles


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)
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
    num_stages: int = 3,
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
        use_tmem = dtype != T.tfloat32
        if use_tmem:
            acc_tmem = T.alloc_tmem((BLOCK_M, BLOCK_N), "float32")
            mbar = T.alloc_barrier(1)

        start_iter = T.alloc_var("int32")
        last_iter = T.alloc_var("int32")
        end_iter = T.alloc_var("int32")
        tile_id = T.alloc_var("int32")
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

            for current_iter in T.Pipelined(
                start_iter,
                end_iter,
                num_stages=1 if use_tmem else num_stages,
            ):
                if not use_tmem and current_iter == start_iter:
                    T.clear(acc)
                k_tile = current_iter % iters_per_tile
                T.copy(A[pid_m * BLOCK_M, k_tile * BLOCK_K], a_shared)
                T.copy(B[k_tile * BLOCK_K, pid_n * BLOCK_N], b_shared)
                if use_tmem:
                    T.gemm(
                        a_shared,
                        b_shared,
                        acc_tmem,
                        mbar=mbar,
                        clear_accum=current_iter == start_iter,
                    )
                    T.sync_threads()
                else:
                    T.gemm(a_shared, b_shared, acc)

            if use_tmem:
                T.copy(acc_tmem, acc)
                T.sync_threads()
            T.atomic_add(C[pid_m * BLOCK_M, pid_n * BLOCK_N], acc)
            T.sync_threads()
            start_iter = end_iter


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)
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
    num_stages: int = 3,
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
        use_tmem = dtype != T.tfloat32
        if use_tmem:
            acc_tmem = T.alloc_tmem((BLOCK_M, BLOCK_N), "float32")
            mbar = T.alloc_barrier(1)
        else:
            T.clear(acc)

        tile_id = pid + streamk_tiles
        group_id = tile_id // (GROUP_M * grid_n)
        first_pid_m = group_id * GROUP_M
        group_size_m = T.min(grid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % (GROUP_M * grid_n)) // group_size_m

        for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
            T.copy(A[pid_m * BLOCK_M, k_tile * BLOCK_K], a_shared)
            T.copy(B[k_tile * BLOCK_K, pid_n * BLOCK_N], b_shared)
            if use_tmem:
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_tmem,
                    mbar=mbar,
                    clear_accum=k_tile == 0,
                )
                T.sync_threads()
            else:
                T.gemm(a_shared, b_shared, acc)

        if use_tmem:
            T.copy(acc_tmem, acc)
        T.copy(acc, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


def _compile_first_wave(a, b, c, dtype, NUM_SMS, cfg):
    return first_wave_kernel.compile(
        a,
        b,
        c,
        dtype=dtype,
        NUM_SMS=NUM_SMS,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        threads=cfg["threads"],
        num_stages=cfg["num_stages"],
    )


def _compile_full_tiles(a, b, c, dtype, NUM_SMS, cfg):
    return full_tiles_kernel.compile(
        a,
        b,
        c,
        dtype=dtype,
        NUM_SMS=NUM_SMS,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        threads=cfg["threads"],
        num_stages=cfg["num_stages"],
    )


def _tune_pipeline(a, b, M, N, K, dtype, NUM_SMS) -> dict:
    key = (M, N, K, str(a.dtype))
    cached = _autotune_cache.get(key)
    if cached is not None:
        return cached

    ref = a @ b
    torch.cuda.synchronize()
    scratch = torch.empty((M, N), device=a.device, dtype=torch.float32)
    best_cfg, best_ms, failures = None, float("inf"), []
    for cfg in streamk_configs():
        total_tiles, streamk_tiles = _streamk_partition(
            M, N, cfg["BLOCK_M"], cfg["BLOCK_N"], NUM_SMS)
        blocking_tiles = total_tiles - streamk_tiles
        try:
            first_kernel = _compile_first_wave(a, b, scratch, dtype, NUM_SMS, cfg)
            full_kernel = None
            if blocking_tiles > 0:
                full_kernel = _compile_full_tiles(a, b, scratch, dtype, NUM_SMS, cfg)

            def _pipeline():
                first_kernel(a, b, scratch)
                if full_kernel is not None:
                    full_kernel(a, b, scratch)

            scratch.zero_()
            _pipeline()
            torch.cuda.synchronize()
            candidate = scratch if a.dtype == torch.float32 else scratch.to(a.dtype)
            if not torch.allclose(candidate, ref, atol=1.0, rtol=1e-2):
                failures.append((cfg, "correctness check failed"))
                continue
            ms = do_bench(_pipeline, warmup=1, rep=3)
        except Exception as e:
            failures.append((cfg, f"{type(e).__name__}: {e}"))
            continue
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    if best_cfg is None:
        raise RuntimeError(
            f"streamk_matmul: all {len(streamk_configs())} pipeline configs "
            f"failed to run; first error: {failures[0][1] if failures else 'n/a'}")
    if failures:
        print(f"  streamk_matmul tilelang autotune: skipped {len(failures)} "
              f"failing config(s), e.g. {failures[0][1][:80]}")
    _autotune_cache[key] = best_cfg
    return best_cfg


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
        cfg = _tune_pipeline(a, b, M, N, K, dtype, NUM_SMS)
        _last_autotune_config.clear()
        _last_autotune_config.update(cfg)
        c.zero_()
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
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
