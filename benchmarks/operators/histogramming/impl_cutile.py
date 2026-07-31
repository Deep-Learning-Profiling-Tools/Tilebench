from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}


_DEFAULT_CONFIG = SimpleNamespace(
    partial_block_size=1024, partial_occupancy=8,
    reduce_block_rows=64, reduce_block_bins=256, reduce_occupancy=8,
)
_NUM_PARTIAL = 256


_PARTIAL_SEARCH_SPACE = [
    SimpleNamespace(block_size=bs, occupancy=occ)
    for bs in [1024, 2048]
    for occ in [4, 8]
]

_REDUCE_SEARCH_SPACE = [
    SimpleNamespace(block_rows=br, block_bins=bb, occupancy=occ)
    for br in [64, 128]
    for bb in [64, 128]
    for occ in [4, 8]
]


@ct.kernel
def histogram_partial_kernel(
    input_ptr,
    partial_ptr,
    N,
    num_bins,
    num_partials,
    BLOCK_SIZE: ConstInt,
):
    pid = ct.bid(0)

    num_chunks = ct.cdiv(N, BLOCK_SIZE)
    for chunk_idx in range(pid, num_chunks, num_partials):
        vals = ct.load(
            input_ptr,
            index=(chunk_idx,),
            shape=(BLOCK_SIZE,),
            padding_mode=ct.PaddingMode.ZERO,
        )

        offs = chunk_idx * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
        valid = offs < N
        in_range = ct.bitwise_and(vals >= 0, vals < num_bins)
        valid = ct.bitwise_and(valid, in_range)


        row_idx = ct.full((BLOCK_SIZE,), pid, dtype=ct.int32)
        bin_idx = ct.where(valid, vals, 0)
        update = ct.astype(valid, ct.int32)

        ct.atomic_add(partial_ptr, (row_idx, bin_idx), update)


@ct.kernel
def histogram_reduce_kernel(
    partial_ptr,
    hist_ptr,
    num_partials,
    num_bins,
    BLOCK_ROWS: ConstInt,
    BLOCK_BINS: ConstInt,
):
    pid_b = ct.bid(0)

    acc = ct.full((BLOCK_BINS,), 0, dtype=ct.int32)

    num_row_tiles = ct.cdiv(num_partials, BLOCK_ROWS)
    for row_tile in range(num_row_tiles):
        tile = ct.load(
            partial_ptr,
            index=(row_tile, pid_b),
            shape=(BLOCK_ROWS, BLOCK_BINS),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = acc + ct.sum(tile, axis=0)

    ct.store(hist_ptr, index=(pid_b,), tile=acc)


_partial_tuner = CutileAutotuner(histogram_partial_kernel)
_reduce_tuner = CutileAutotuner(histogram_reduce_kernel)


def run(input: torch.Tensor, N: int, num_bins: int,
        block_size: int = None, autotune: bool = False, **kwargs):

    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    input = input.contiguous()
    histogram = torch.empty((num_bins,), device=input.device, dtype=torch.int32)


    default_block_size = (
        int(block_size) if block_size is not None else _DEFAULT_CONFIG.partial_block_size
    )
    num_partials = min(_NUM_PARTIAL, (N + default_block_size - 1) // default_block_size)


    partial = torch.zeros((num_partials, num_bins), device=input.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()


    if autotune:


        scratch = torch.empty_like(partial)
        partial_cfg = _partial_tuner.tune_or_cached(
            shape_key=(N, num_bins, num_partials, str(input.dtype)),
            search_space=_PARTIAL_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (num_partials, 1, 1),
            args_fn=lambda cfg: (
                input, scratch, N, num_bins, num_partials, cfg.block_size,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
    else:
        partial_cfg = SimpleNamespace(
            block_size=default_block_size,
            occupancy=_DEFAULT_CONFIG.partial_occupancy,
        )

    partial_kernel = _partial_tuner.kernel_with_hints(occupancy=partial_cfg.occupancy)
    ct.launch(
        stream, (num_partials, 1, 1), partial_kernel,
        (input, partial, N, num_bins, num_partials, partial_cfg.block_size),
    )


    if autotune:
        reduce_cfg = _reduce_tuner.tune_or_cached(
            shape_key=(num_partials, num_bins, str(input.dtype)),
            search_space=_REDUCE_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((num_bins + cfg.block_bins - 1) // cfg.block_bins, 1, 1),
            args_fn=lambda cfg: (
                partial, histogram, num_partials, num_bins,
                cfg.block_rows, cfg.block_bins,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
    else:
        reduce_cfg = SimpleNamespace(
            block_rows=_DEFAULT_CONFIG.reduce_block_rows,
            block_bins=_DEFAULT_CONFIG.reduce_block_bins,
            occupancy=_DEFAULT_CONFIG.reduce_occupancy,
        )

    reduce_kernel = _reduce_tuner.kernel_with_hints(occupancy=reduce_cfg.occupancy)
    grid_reduce = ((num_bins + reduce_cfg.block_bins - 1) // reduce_cfg.block_bins, 1, 1)
    ct.launch(
        stream, grid_reduce, reduce_kernel,
        (partial, histogram, num_partials, num_bins,
         reduce_cfg.block_rows, reduce_cfg.block_bins),
    )

    if autotune:
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "partial_block_size": partial_cfg.block_size,
            "partial_occupancy":  partial_cfg.occupancy,
            "reduce_block_rows":  reduce_cfg.block_rows,
            "reduce_block_bins":  reduce_cfg.block_bins,
            "reduce_occupancy":   reduce_cfg.occupancy,
        })

    return histogram


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
