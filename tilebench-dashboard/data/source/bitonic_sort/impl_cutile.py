from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=4)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048, 4096]
    for occ in [4, 8, 16, 32]
]
_last_autotune_config: dict = {}


def _next_pow2(n: int) -> int:
    return 1 << ((n - 1).bit_length()) if n > 1 else 1


@ct.kernel
def pad_kernel(data_ptr, work_ptr, N, M, TILE: ConstInt):
    bid = ct.bid(0)


    vals = ct.load(data_ptr, index=(bid,), shape=(TILE,),
                   padding_mode=ct.PaddingMode.POS_INF)

    ct.store(work_ptr, index=(bid,), tile=vals)


@ct.kernel
def bitonic_step_kernel(work_ptr, k, j, M, TILE: ConstInt):
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    ixj = offs ^ j

    active = (ixj > offs) & (ixj < M) & (offs < M)


    a = ct.load(work_ptr, index=(bid,), shape=(TILE,),
                padding_mode=ct.PaddingMode.ZERO)
    b = ct.gather(work_ptr, ixj, padding_value=0.0)

    ascending = (offs & k) == 0
    swap = ct.where(ascending, a > b, a < b)
    new_a = ct.where(swap, b, a)
    new_b = ct.where(swap, a, b)


    offs_safe = ct.where(active, offs, M)
    ixj_safe = ct.where(active, ixj, M)

    ct.scatter(work_ptr, offs_safe, new_a)
    ct.scatter(work_ptr, ixj_safe, new_b)


_tuner = CutileAutotuner(bitonic_step_kernel)


def run(data: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):

    if N <= 1:
        return data.clone()

    M = _next_pow2(N)
    work = torch.empty((M,), device=data.device, dtype=data.dtype)
    stream = torch.cuda.current_stream()


    default_tile = _DEFAULT_CONFIG.tile
    pad_grid = (ct.cdiv(M, default_tile), 1, 1)
    ct.launch(stream, pad_grid, pad_kernel, (data, work, N, M, default_tile))


    if autotune:


        k0, j0 = 2, 1
        cfg = _tuner.tune_or_cached(
            shape_key=(M, str(data.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(M, cfg.tile), 1, 1),
            args_fn=lambda cfg: (work, k0, j0, M, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    grid = (ct.cdiv(M, cfg.tile), 1, 1)
    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            ct.launch(stream, grid, kernel, (work, k, j, M, cfg.tile))
            j //= 2
        k *= 2

    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
