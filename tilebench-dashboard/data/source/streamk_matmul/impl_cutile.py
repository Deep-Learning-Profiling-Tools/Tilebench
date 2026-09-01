from types import SimpleNamespace

import cuda.tile as ct
import torch

from triton.testing import do_bench

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(
    tm=128, tn=128, tk=64, group_m=8, occupancy=8,
)


_SEARCH_SPACE = [
    SimpleNamespace(tm=tm, tn=tn, tk=tk, group_m=8, occupancy=occ)
    for tm in (64, 128)
    for tn in (128, 256)
    for tk in (32, 64)
    for occ in (4, 8, 16)
]


def _swizzle_tile_coords(tile_id, M, N, TM, TN, GROUP_M):
    grid_m = (M + TM - 1) // TM
    grid_n = (N + TN - 1) // TN
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = ct.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


def _streamk_partition(M, N, TM, TN, NUM_SMS):
    total_tiles = ((M + TM - 1) // TM) * ((N + TN - 1) // TN)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    return total_tiles, streamk_tiles


@ct.kernel
def first_wave(
    A, B, C,
    K: ConstInt,
    NUM_SMS: ConstInt,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_M: ConstInt,
):
    M = A.shape[0]
    N = B.shape[1]

    grid_m = (M + TM - 1) // TM
    grid_n = (N + TN - 1) // TN
    total_tiles = grid_m * grid_n
    iters_per_tile = (K + TK - 1) // TK
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles = streamk_tiles + NUM_SMS
    total_iters_streamk = streamk_tiles * iters_per_tile
    total_full_iters = total_iters_streamk // NUM_SMS
    total_partial_iters = total_iters_streamk % NUM_SMS

    pid = ct.bid(0)
    start_iter = pid * total_full_iters + ct.minimum(pid, total_partial_iters)
    last_iter = (pid + 1) * total_full_iters + ct.minimum(pid + 1, total_partial_iters)

    mma_dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    while start_iter < last_iter:
        rem = iters_per_tile - (start_iter % iters_per_tile)
        end_iter = ct.minimum(start_iter + rem, last_iter)

        tile_id = start_iter // iters_per_tile
        pid_m, pid_n = _swizzle_tile_coords(tile_id, M, N, TM, TN, GROUP_M)

        acc = ct.full((TM, TN), 0.0, dtype=ct.float32)


        k_chunk = start_iter % iters_per_tile
        current_iter = start_iter
        while current_iter < end_iter:
            a_tile = ct.load(A, index=(pid_m, k_chunk), shape=(TM, TK),
                             padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
            b_tile = ct.load(B, index=(k_chunk, pid_n), shape=(TK, TN),
                             padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
            acc = ct.mma(a_tile, b_tile, acc)
            k_chunk = k_chunk + 1
            current_iter = current_iter + 1

        acc_casted = ct.astype(acc, C.dtype)


        row_indices = ct.expand_dims(pid_m * TM + ct.arange(TM, dtype=ct.int32), 1)
        col_indices = ct.expand_dims(pid_n * TN + ct.arange(TN, dtype=ct.int32), 0)
        ct.atomic_add(C, (row_indices, col_indices), acc_casted)

        start_iter = end_iter


@ct.kernel
def full_tiles(
    A, B, C,
    K: ConstInt,
    NUM_SMS: ConstInt,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_M: ConstInt,
):
    M = A.shape[0]
    N = B.shape[1]

    grid_m = (M + TM - 1) // TM
    grid_n = (N + TN - 1) // TN
    total_tiles = grid_m * grid_n
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles = streamk_tiles + NUM_SMS

    tile_id = ct.bid(0) + streamk_tiles
    if tile_id >= total_tiles:
        return

    pid_m, pid_n = _swizzle_tile_coords(tile_id, M, N, TM, TN, GROUP_M)

    mma_dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    acc = ct.full((TM, TN), 0.0, dtype=ct.float32)

    num_tiles_k = (K + TK - 1) // TK
    for k in range(num_tiles_k):
        a_tile = ct.load(A, index=(pid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        b_tile = ct.load(B, index=(k, pid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        acc = ct.mma(a_tile, b_tile, acc)

    acc_casted = ct.astype(acc, C.dtype)
    ct.store(C, index=(pid_m, pid_n), tile=acc_casted)


_first_wave_tuner = CutileAutotuner(first_wave)
_full_tiles_tuner = CutileAutotuner(full_tiles)


_pipeline_tune_cache: dict = {}


def _tune_pipeline(a, b, scratch, M, N, K, NUM_SMS, stream) -> SimpleNamespace:
    key = (M, N, K, str(a.dtype))
    cached = _pipeline_tune_cache.get(key)
    if cached is not None:
        return cached

    best_cfg, best_ms, failures = None, float("inf"), []
    for cfg in _SEARCH_SPACE:
        total_tiles, streamk_tiles = _streamk_partition(M, N, cfg.tm, cfg.tn, NUM_SMS)
        blocking_tiles = total_tiles - streamk_tiles
        args = (a, b, scratch, K, NUM_SMS, cfg.tm, cfg.tn, cfg.tk, cfg.group_m)

        def _pipeline():


            fk = _first_wave_tuner.kernel_with_hints(occupancy=cfg.occupancy)
            ct.launch(stream, (NUM_SMS, 1, 1), fk, args)
            if blocking_tiles > 0:
                lk = _full_tiles_tuner.kernel_with_hints(occupancy=cfg.occupancy)
                ct.launch(stream, (blocking_tiles, 1, 1), lk, args)

        try:
            ms = do_bench(_pipeline, warmup=1, rep=3)
        except Exception as e:
            failures.append(f"{type(e).__name__}: {e}")
            continue
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    if best_cfg is None:
        raise RuntimeError(
            f"streamk_matmul: all {len(_SEARCH_SPACE)} pipeline configs "
            f"failed to run; first error: {failures[0] if failures else 'n/a'}")
    if failures:
        print(f"  streamk_matmul cuTile autotune: skipped {len(failures)} "
              f"failing config(s), e.g. {failures[0][:80]}")
    _pipeline_tune_cache[key] = best_cfg
    return best_cfg


def _device_sm_count() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def run(a: torch.Tensor, b: torch.Tensor,
        block_size: int = None, autotune: bool = False, **kwargs):

    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape
    NUM_SMS = _device_sm_count()

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)


    if a.dtype == torch.float32:
        c = out
        c.zero_()
    else:
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()


    if autotune:


        scratch = torch.empty_like(c)
        cfg = _tune_pipeline(a, b, scratch, M, N, K, NUM_SMS, stream)
    else:
        cfg = _DEFAULT_CONFIG


    first_kernel = _first_wave_tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, (NUM_SMS, 1, 1), first_kernel,
              (a, b, c, K, NUM_SMS, cfg.tm, cfg.tn, cfg.tk, cfg.group_m))


    total_tiles, streamk_tiles = _streamk_partition(M, N, cfg.tm, cfg.tn, NUM_SMS)
    blocking_tiles = total_tiles - streamk_tiles
    if blocking_tiles > 0:
        full_kernel = _full_tiles_tuner.kernel_with_hints(occupancy=cfg.occupancy)
        ct.launch(stream, (blocking_tiles, 1, 1), full_kernel,
                  (a, b, c, K, NUM_SMS, cfg.tm, cfg.tn, cfg.tk, cfg.group_m))

    if autotune:
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tm": cfg.tm, "tn": cfg.tn, "tk": cfg.tk,
            "group_m": cfg.group_m, "occupancy": cfg.occupancy,
        })

    if c is not out:
        out.copy_(c)
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
