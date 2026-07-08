from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(
    tile_m=128, tile_n=128, tile_k=64, occupancy=4, group_size=1,
)
_SEARCH_SPACE = [
    SimpleNamespace(tile_m=tm, tile_n=tn, tile_k=tk, occupancy=occ, group_size=gs)
    for tm in [32, 64, 128]
    for tn in [32, 64, 128]
    for tk in [32, 64]
    for gs in [1, 8]   # gs=1 == no swizzle (row-major), gives autotuner an escape hatch
    for occ in [4, 8, 16]

]
_last_autotune_config: dict = {}


@ct.kernel
def bmm_kernel(a_3d, b_3d, c_3d,
                K_TILES:    ConstInt,
                TILE_M:     ConstInt,
                TILE_N:     ConstInt,
                TILE_K:     ConstInt,
                GRID_M:     ConstInt,   # cdiv(M, TILE_M)
                GRID_N:     ConstInt,   # cdiv(N, TILE_N)
                GROUP_SIZE: ConstInt):
    """
    Batched tiled GEMM — direct mirror of Triton's tl.dot method, with swizzle.

    Grid: (GRID_M * GRID_N, BATCH). The (M, N) tile axes are collapsed into a
    single linear bid(0), then remapped to (bid_m, bid_n) via swizzle math —
    the same (bid % num_pid_in_group) trick Triton performs inside
    tl.swizzle2d. This reorders CTA launch into GROUP_SIZE-row blocks so that
    concurrently-scheduled CTAs share more rows of A and columns of B,
    improving L2 hit rate.

    GROUP_SIZE = 1 degenerates to the default row-major order (no swizzle).

    cuTile's ct.mma is the direct analog of tl.dot: native-dtype inputs with an
    fp32 accumulator, producing the same hardware MMA op. OOB lanes in the
    last M/N/K tile are zeroed via padding_mode=ZERO, and the final ct.store
    silently drops OOB writes — matching Triton's masked-store semantics
    without needing ct.scatter.
    """
    linear_bid = ct.bid(0)
    bid_b      = ct.bid(1)

    # --- swizzle 2d: math-equivalent to tl.swizzle2d(pid0, pid1, ..., GROUP_SIZE) ---
    num_pid_in_group = GROUP_SIZE * GRID_N
    group_id         = linear_bid // num_pid_in_group
    first_pid_m      = group_id * GROUP_SIZE
    # handle the tail group when GRID_M % GROUP_SIZE != 0
    group_size_m     = min(GRID_M - first_pid_m, GROUP_SIZE)
    bid_m = first_pid_m + ((linear_bid % num_pid_in_group) % group_size_m)
    bid_n = (linear_bid % num_pid_in_group) // group_size_m
    # ------------------------------------------------------------------------------

    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

    # Cast to TF32 for fp32 inputs so ct.mma uses Tensor Cores; no-op for
    # fp16/bf16 inputs (cast to same dtype). Matches Triton's
    # input_precision="tf32" kwarg in impl_triton.py.
    mma_dtype = ct.tfloat32 if a_3d.dtype == ct.float32 else a_3d.dtype

    for bid_k in range(K_TILES):  # compile-time unrolled (cuTile DSL constraint)
        a_tile = ct.load(
            a_3d, index=(bid_b, bid_m, bid_k),
            shape=(1, TILE_M, TILE_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            b_3d, index=(bid_b, bid_k, bid_n),
            shape=(1, TILE_K, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )

        a_2d = ct.reshape(a_tile, (TILE_M, TILE_K)).astype(mma_dtype)
        b_2d = ct.reshape(b_tile, (TILE_K, TILE_N)).astype(mma_dtype)

        # ct.mma(x, y, acc) = acc + x @ y with native-dtype inputs + fp32 acc,
        # same underlying hardware op as Triton's tl.dot(a, b) + accumulator.
        acc = ct.mma(a_2d, b_2d, acc)

    acc_3d = ct.reshape(acc, (1, TILE_M, TILE_N))
    ct.store(c_3d, index=(bid_b, bid_m, bid_n),
             tile=ct.astype(acc_3d, c_3d.dtype))


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(bmm_kernel)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile batched matrix multiplication — direct mirror of Triton's tl.dot tiled GEMM
    including the tl.swizzle2d L2-locality trick:
      grid (GRID_M * GRID_N, BATCH); a linear bid(0) is remapped in-kernel to
      (bid_m, bid_n) via GROUP_SIZE-row swizzle, then each block accumulates
      K_TILES (TILE_M, TILE_K) x (TILE_K, TILE_N) matmuls along K via ct.mma.

    Algorithm and scheduling are both equivalent to the Triton version:
      - tiled GEMM + K-loop accumulation (algorithm)
      - GROUP_SIZE swizzle for L2-friendly CTA ordering (scheduling)
    The only remaining cross-DSL differences are compiler codegen and the
    autotune search space.
    """

    a_3d = A.contiguous().view(BATCH, M, K)
    b_3d = B.contiguous().view(BATCH, K, N)
    c_3d = torch.empty(BATCH, M, N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(BATCH, M, N, K),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                ct.cdiv(M, cfg.tile_m) * ct.cdiv(N, cfg.tile_n),
                BATCH,
                1,
            ),
            args_fn=lambda cfg: (
                a_3d, b_3d, c_3d,
                ct.cdiv(K, cfg.tile_k),
                cfg.tile_m, cfg.tile_n, cfg.tile_k,
                ct.cdiv(M, cfg.tile_m),   # GRID_M
                ct.cdiv(N, cfg.tile_n),   # GRID_N
                cfg.group_size,           # GROUP_SIZE
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile_m":     cfg.tile_m,
            "tile_n":     cfg.tile_n,
            "tile_k":     cfg.tile_k,
            "occupancy":  cfg.occupancy,
            "group_size": cfg.group_size,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid_m = ct.cdiv(M, cfg.tile_m)
    grid_n = ct.cdiv(N, cfg.tile_n)
    grid = (grid_m * grid_n, BATCH, 1)
    K_TILES = ct.cdiv(K, cfg.tile_k)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel, (
        a_3d, b_3d, c_3d,
        K_TILES,
        cfg.tile_m, cfg.tile_n, cfg.tile_k,
        grid_m, grid_n, cfg.group_size,
    ))

    return c_3d.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
