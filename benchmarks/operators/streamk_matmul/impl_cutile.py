"""cuTile Stream-K GEMM (Osama et al., PPoPP 2023).

Mirrors impl_triton.py: two kernels, scheduler computed INSIDE each
kernel from M/N/K/NUM_SMS/TM/TN/TK so the autotuner can sweep TM/TN/TK
freely without desynchronising any host-side scheduler state.

  first_wave_kernel - launches NUM_SMS programs; each owns a contiguous
                      range of K-iterations spanning multiple (M, N)
                      tiles. Combines partials via ct.atomic_add into a
                      pre-zeroed C — order-independent, no locks.
  full_tiles_kernel - data-parallel; one program per leftover whole
                      tile. Writes pre-zeroed disjoint regions via
                      ct.store (non-atomic).

Only first_wave_kernel is autotuned; full_tiles_kernel reuses the
winning TM/TN/TK so both kernels' tile partitions agree.

Boundary handling: `ct.load(..., padding_mode=ZERO)` returns 0 for OOB
lanes; tile-space `ct.store` and per-element `ct.atomic_add` silently
drop OOB writes — equivalent to Triton's masked load / store.

Precision: fp32 inputs are cast to `ct.tfloat32` before `ct.mma`,
matching Triton's `tl.dot(input_precision="tf32")`.

Output staging: cuTile's `ct.atomic_add` does not support bfloat16. To
keep one code path across dtypes, all kernel writes go to an fp32
staging buffer, then we cast back to the input dtype at the end. fp32
inputs skip the staging (the output buffer is already fp32).
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(
    tm=128, tn=128, tk=32, group_m=8, occupancy=16,
)
_SEARCH_SPACE = [
    SimpleNamespace(tm=tm, tn=tn, tk=32, group_m=8, occupancy=occ)
    for tm in (64, 128)
    for tn in (128, 256)
    for occ in (4, 8, 16)
]


def _swizzle_tile_coords(tile_id, M, N, TM, TN, GROUP_M):
    """Plain-Python helper inlined by cuTile's tracer when called from a
    @ct.kernel — no decorator needed."""
    grid_m = (M + TM - 1) // TM
    grid_n = (N + TN - 1) // TN
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = ct.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


def _streamk_partition(M, N, TM, TN, NUM_SMS):
    """Mirror the kernel's scheduler. Returns (total_tiles, streamk_tiles)."""
    total_tiles = ((M + TM - 1) // TM) * ((N + TN - 1) // TN)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    return total_tiles, streamk_tiles


@ct.kernel
def first_wave_kernel(
    A, B, C,
    NUM_SMS: ConstInt,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_M: ConstInt,
):
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

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

        current_iter = start_iter
        while current_iter < end_iter:
            k_chunk = current_iter % iters_per_tile
            a_tile = ct.load(A, index=(pid_m, k_chunk), shape=(TM, TK),
                             padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
            b_tile = ct.load(B, index=(k_chunk, pid_n), shape=(TK, TN),
                             padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
            acc = ct.mma(a_tile, b_tile, acc)
            current_iter = current_iter + 1

        acc_casted = ct.astype(acc, C.dtype)

        # All Stream-K writes go through atomic_add into pre-zeroed C —
        # order-independent partial-tile accumulation, no locks needed.
        row_indices = ct.expand_dims(pid_m * TM + ct.arange(TM, dtype=ct.int32), 1)
        col_indices = ct.expand_dims(pid_n * TN + ct.arange(TN, dtype=ct.int32), 0)
        ct.atomic_add(C, (row_indices, col_indices), acc_casted)

        start_iter = end_iter


@ct.kernel
def full_tiles_kernel(
    A, B, C,
    NUM_SMS: ConstInt,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_M: ConstInt,
):
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

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
    k = 0
    while k < num_tiles_k:
        a_tile = ct.load(A, index=(pid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        b_tile = ct.load(B, index=(k, pid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        acc = ct.mma(a_tile, b_tile, acc)
        k = k + 1

    acc_casted = ct.astype(acc, C.dtype)
    ct.store(C, index=(pid_m, pid_n), tile=acc_casted)


_first_wave_tuner = CutileAutotuner(first_wave_kernel)
_full_tiles_tuner = CutileAutotuner(full_tiles_kernel)


def _device_sm_count() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def run(a: torch.Tensor, b: torch.Tensor,
        block_size: int = None, autotune: bool = False, **kwargs):

    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape
    NUM_SMS = _device_sm_count()

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # cuTile's ct.atomic_add doesn't support bfloat16; route all writes
    # through an fp32 staging buffer for non-fp32 dtypes.
    if a.dtype == torch.float32:
        c = out
        c.zero_()
    else:
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    # ---- Stage 1: first_wave (Stream-K), autotuned ----
    if autotune:
        # Tune against a scratch buffer so the kernel's atomic_add
        # accumulation across timing samples doesn't pollute the real
        # output `c`. After tuning, we launch on real `c` once below.
        scratch = torch.empty_like(c)
        first_cfg = _first_wave_tuner.tune_or_cached(
            shape_key=(M, N, K, str(a.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (NUM_SMS, 1, 1),
            args_fn=lambda cfg: (a, b, scratch, NUM_SMS, cfg.tm, cfg.tn, cfg.tk, cfg.group_m),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
    else:
        first_cfg = _DEFAULT_CONFIG

    first_kernel = _first_wave_tuner.kernel_with_hints(occupancy=first_cfg.occupancy)
    ct.launch(stream, (NUM_SMS, 1, 1), first_kernel,
              (a, b, c, NUM_SMS, first_cfg.tm, first_cfg.tn, first_cfg.tk, first_cfg.group_m))

    # ---- Stage 2: full_tiles, reuses Stage 1's TM/TN/TK ----
    total_tiles, streamk_tiles = _streamk_partition(M, N, first_cfg.tm, first_cfg.tn, NUM_SMS)
    blocking_tiles = total_tiles - streamk_tiles
    if blocking_tiles > 0:
        full_kernel = _full_tiles_tuner.kernel_with_hints(occupancy=first_cfg.occupancy)
        ct.launch(stream, (blocking_tiles, 1, 1), full_kernel,
                  (a, b, c, NUM_SMS, first_cfg.tm, first_cfg.tn, first_cfg.tk, first_cfg.group_m))

    if autotune:
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tm": first_cfg.tm, "tn": first_cfg.tn, "tk": first_cfg.tk,
            "group_m": first_cfg.group_m, "occupancy": first_cfg.occupancy,
        })

    if c is not out:
        out.copy_(c)
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
