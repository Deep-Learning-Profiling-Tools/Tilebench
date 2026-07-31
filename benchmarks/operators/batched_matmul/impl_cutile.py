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
    for gs in [1, 8]
    for occ in [4, 8, 16]

]
_last_autotune_config: dict = {}


@ct.kernel
def bmm_kernel(a_3d, b_3d, c_3d,
                K_TILES:    ConstInt,
                TILE_M:     ConstInt,
                TILE_N:     ConstInt,
                TILE_K:     ConstInt,
                GRID_M:     ConstInt,
                GRID_N:     ConstInt,
                GROUP_SIZE: ConstInt):
    linear_bid = ct.bid(0)
    bid_b      = ct.bid(1)


    num_pid_in_group = GROUP_SIZE * GRID_N
    group_id         = linear_bid // num_pid_in_group
    first_pid_m      = group_id * GROUP_SIZE

    group_size_m     = min(GRID_M - first_pid_m, GROUP_SIZE)
    bid_m = first_pid_m + ((linear_bid % num_pid_in_group) % group_size_m)
    bid_n = (linear_bid % num_pid_in_group) // group_size_m


    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)


    mma_dtype = ct.tfloat32 if a_3d.dtype == ct.float32 else a_3d.dtype

    for bid_k in range(K_TILES):
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


        acc = ct.mma(a_2d, b_2d, acc)

    acc_3d = ct.reshape(acc, (1, TILE_M, TILE_N))
    ct.store(c_3d, index=(bid_b, bid_m, bid_n),
             tile=ct.astype(acc_3d, c_3d.dtype))


_tuner = CutileAutotuner(bmm_kernel)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):

    a_3d = A.contiguous().view(BATCH, M, K)
    b_3d = B.contiguous().view(BATCH, K, N)
    c_3d = torch.empty(BATCH, M, N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(BATCH, M, N, K, str(A.dtype)),
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
                ct.cdiv(M, cfg.tile_m),
                ct.cdiv(N, cfg.tile_n),
                cfg.group_size,
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
