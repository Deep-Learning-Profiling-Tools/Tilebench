from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(occupancy=4)
_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [4, 8, 16, 32]]
_last_autotune_config: dict = {}

_BLOCK_SIZE = 1024
_RADIX_BITS = 2
_RADIX = 1 << _RADIX_BITS
_FIELD_BITS = 16
_FIELD_MASK = (1 << _FIELD_BITS) - 1


@ct.kernel
def radix_histogram_kernel(input_ptr, hist_ptr, N, K, shift,
                           TILE: ConstInt, RADIX: ConstInt,
                           FIELD_BITS: ConstInt, FIELD_MASK: ConstInt):
    bid = ct.bid(0)
    offset = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    mask = offset < N

    block = ct.load(input_ptr, index=(bid,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO)
    digit = ct.astype((block >> shift) & (RADIX - 1), ct.int32)


    packed = ct.astype(mask, ct.int64) << ct.astype(digit * FIELD_BITS, ct.int64)
    total = ct.sum(packed, axis=0)


    lanes = ct.arange(RADIX, dtype=ct.int32)
    counts = ct.astype((total >> ct.astype(lanes * FIELD_BITS, ct.int64)) & FIELD_MASK,
                       ct.int32)
    ct.scatter(hist_ptr, lanes * K + bid, counts)


@ct.kernel
def radix_sum_chunks_kernel(src_ptr, dst_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(src_ptr, index=(bid,), shape=(TILE,),
                   padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst_ptr, index=(bid,), tile=ct.sum(vals, axis=0, keepdims=True))


@ct.kernel
def radix_scan_chunk_sums_kernel(sums_ptr, TILE_BB: ConstInt):

    vals = ct.load(sums_ptr, index=(0,), shape=(TILE_BB,),
                   padding_mode=ct.PaddingMode.ZERO)
    excl = ct.cumsum(vals, axis=0) - vals
    ct.store(sums_ptr, index=(0,), tile=excl)


@ct.kernel
def radix_scan_chunks_kernel(src_ptr, chunk_offsets_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(src_ptr, index=(bid,), shape=(TILE,),
                   padding_mode=ct.PaddingMode.ZERO)
    base = ct.load(chunk_offsets_ptr, index=(bid,), shape=())
    excl = ct.cumsum(vals, axis=0) - vals + base
    ct.store(src_ptr, index=(bid,), tile=excl)


@ct.kernel
def radix_scatter_kernel(input_ptr, output_ptr, hist_ptr, N, K, shift,
                         TILE: ConstInt, RADIX: ConstInt,
                         FIELD_BITS: ConstInt, FIELD_MASK: ConstInt):
    bid = ct.bid(0)
    offset = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    mask = offset < N

    block = ct.load(input_ptr, index=(bid,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO)
    digit = ct.astype((block >> shift) & (RADIX - 1), ct.int32)


    field = ct.astype(digit * FIELD_BITS, ct.int64)
    packed = ct.astype(mask, ct.int64) << field
    excl = ct.cumsum(packed, axis=0) - packed
    rank = ct.astype((excl >> field) & FIELD_MASK, ct.int32)

    base = ct.gather(hist_ptr, digit * K + bid, padding_value=0)

    dest = ct.where(mask, base + rank, N)
    ct.scatter(output_ptr, dest, block)


_HELPER_OCC = 8
_hist_kernel = radix_histogram_kernel.replace_hints(occupancy=_HELPER_OCC)
_sum_chunks_kernel = radix_sum_chunks_kernel.replace_hints(occupancy=_HELPER_OCC)
_scan_chunk_sums_kernel = radix_scan_chunk_sums_kernel.replace_hints(occupancy=_HELPER_OCC)
_scan_chunks_kernel = radix_scan_chunks_kernel.replace_hints(occupancy=_HELPER_OCC)


_tuner = CutileAutotuner(radix_scatter_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    if N <= 1:
        return input.clone()

    work = input.clone()
    output = torch.empty_like(input)

    K = (N + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    M = _RADIX * K
    G2 = (M + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    BB = max(1024, 1 << (G2 - 1).bit_length())

    hist = torch.empty((M,), dtype=torch.int32, device=input.device)
    chunk_sums = torch.empty((G2,), dtype=torch.int32, device=input.device)

    stream = torch.cuda.current_stream()
    grid = (K, 1, 1)
    grid_chunks = (G2, 1, 1)
    grid_one = (1, 1, 1)


    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (
                work, output, hist, N, K, 0,
                _BLOCK_SIZE, _RADIX, _FIELD_BITS, _FIELD_MASK,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    scatter_kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)

    for shift in range(0, 32, _RADIX_BITS):
        ct.launch(stream, grid, _hist_kernel,
                  (work, hist, N, K, shift,
                   _BLOCK_SIZE, _RADIX, _FIELD_BITS, _FIELD_MASK))
        ct.launch(stream, grid_chunks, _sum_chunks_kernel,
                  (hist, chunk_sums, _BLOCK_SIZE))
        ct.launch(stream, grid_one, _scan_chunk_sums_kernel,
                  (chunk_sums, BB))
        ct.launch(stream, grid_chunks, _scan_chunks_kernel,
                  (hist, chunk_sums, _BLOCK_SIZE))
        ct.launch(stream, grid, scatter_kernel,
                  (work, output, hist, N, K, shift,
                   _BLOCK_SIZE, _RADIX, _FIELD_BITS, _FIELD_MASK))


        work, output = output, work

    return work


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
