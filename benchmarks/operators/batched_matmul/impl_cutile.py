from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile_m=64, tile_n=64, tile_k=32, occupancy=2)
_SEARCH_SPACE = [
    SimpleNamespace(tile_m=tm, tile_n=tn, tile_k=tk, occupancy=occ)
    for tm in [32, 64, 128]
    for tn in [32, 64, 128]
    for tk in [32, 64]
    for occ in [1, 2, 4]
]
_last_autotune_config = None


@ct.kernel
def _bmm_kernel(a_3d, b_3d, c_3d,
                K_TILES: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                TILE_K: ConstInt):
    """
    Batched tiled GEMM — direct mirror of Triton's tl.dot method.
    Grid: (M_tiles, N_tiles, BATCH). Each program computes one
    (TILE_M, TILE_N) output block for one batch by accumulating K_TILES
    matmul results along the K dimension (same as Triton's K-loop).

    cuTile's ct.mma is the direct analog of tl.dot: native-dtype inputs
    with an fp32 accumulator, producing the same hardware MMA op.
    OOB lanes in the last M/N/K tile are zeroed via padding_mode=ZERO,
    and the final ct.store silently drops OOB writes — matching
    Triton's masked-store semantics without needing ct.scatter.
    """
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    bid_b = ct.bid(2)

    acc = ct.zeros((TILE_M, TILE_N), dtype=np.float32)

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

        a_2d = ct.reshape(a_tile, (TILE_M, TILE_K))
        b_2d = ct.reshape(b_tile, (TILE_K, TILE_N))

        # ct.mma(x, y, acc) = acc + x @ y with native-dtype inputs + fp32 acc,
        # same underlying hardware op as Triton's tl.dot(a, b) + accumulator.
        acc = ct.mma(a_2d, b_2d, acc)

    acc_3d = ct.reshape(acc, (1, TILE_M, TILE_N))
    ct.store(c_3d, index=(bid_b, bid_m, bid_n),
             tile=ct.astype(acc_3d, c_3d.dtype))


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile batched matrix multiplication — direct mirror of Triton's tl.dot tiled GEMM:
      grid (M/TILE_M, N/TILE_N, BATCH); each block accumulates K_TILES (TILE_M, TILE_K) x
      (TILE_K, TILE_N) matmuls along the K dimension via ct.mma.

    Note: Triton uses tl.swizzle2d for L2-locality-friendly scheduling of (pid_m, pid_n).
    cuTile has no direct swizzle primitive, so the block schedule is the default
    (bid_m, bid_n, bid_b) linearization. This is a Triton-specific scheduling trick, not
    a method change — the algorithm (tiled GEMM + K-loop accumulation) is identical.
    """
    global _last_autotune_config

    a_3d = A.contiguous().view(BATCH, M, K)
    b_3d = B.contiguous().view(BATCH, K, N)
    c_3d = torch.empty(BATCH, M, N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (
                ct.cdiv(M, cfg.tile_m),
                ct.cdiv(N, cfg.tile_n),
                BATCH,
            ),
            kernel=_bmm_kernel,
            args_fn=lambda cfg: (
                a_3d, b_3d, c_3d,
                ct.cdiv(K, cfg.tile_k),
                cfg.tile_m, cfg.tile_n, cfg.tile_k,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile_m":    result.tuned_config.tile_m,
            "tile_n":    result.tuned_config.tile_n,
            "tile_k":    result.tuned_config.tile_k,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        grid = (
            ct.cdiv(M, cfg.tile_m),
            ct.cdiv(N, cfg.tile_n),
            BATCH,
        )
        K_TILES = ct.cdiv(K, cfg.tile_k)
        ct.launch(stream, grid, _bmm_kernel,
                  (a_3d, b_3d, c_3d, K_TILES, cfg.tile_m, cfg.tile_n, cfg.tile_k))

    return c_3d.reshape(-1)


def get_last_config() -> dict | None:
    return _last_autotune_config
