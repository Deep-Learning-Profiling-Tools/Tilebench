from typing import Annotated

import torch
import cuda.tile as ct


StaticVector = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(0,)),
]
StaticMatrix = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(0, 1)),
]


@ct.kernel(num_ctas=1, occupancy=4, opt_level=3)
def _count_private(
    input: StaticVector,
    partials: StaticMatrix,
    N: ct.Constant[int],
    P: ct.Constant[int],
    CHUNK: ct.Constant[int],
    NUM_BINS: ct.Constant[int],
):
    pid = ct.bid(0)

    ct.store(
        partials,
        (pid, 0),
        ct.zeros((1, NUM_BINS), ct.int32),
        allow_tma=False,
    )

    values = ct.load(
        input,
        (pid,),
        (CHUNK,),
        allow_tma=False,
    )

    # Issue the next chunk's loads before counting the current chunk.
    for chunk_id in range(pid + P, N // CHUNK, P):
        next_values = ct.load(
            input,
            (chunk_id,),
            (CHUNK,),
            allow_tma=False,
        )

        ct.atomic_add(
            partials,
            (pid, values),
            1,
            check_bounds=True,
            memory_order=ct.MemoryOrder.RELAXED,
            memory_scope=ct.MemoryScope.BLOCK,
        )
        values = next_values

    ct.atomic_add(
        partials,
        (pid, values),
        1,
        check_bounds=True,
        memory_order=ct.MemoryOrder.RELAXED,
        memory_scope=ct.MemoryScope.BLOCK,
    )


@ct.kernel(num_ctas=1, occupancy=4, opt_level=3)
def _reduce_private(
    partials: StaticMatrix,
    output: StaticVector,
    P: ct.Constant[int],
    ROWS: ct.Constant[int],
    BINS: ct.Constant[int],
):
    bin_tile = ct.bid(0)

    accumulator = ct.load(
        partials,
        (0, bin_tile),
        (ROWS, BINS),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )

    for row_tile in range(1, ct.cdiv(P, ROWS)):
        values = ct.load(
            partials,
            (row_tile, bin_tile),
            (ROWS, BINS),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        )
        accumulator = accumulator + values

    counts = ct.sum(accumulator, axis=0)
    ct.store(output, (bin_tile,), counts, allow_tma=False)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.int32
    assert len(input.shape) == 1
    assert input.shape[0] == N
    assert input.stride() == (1,)
    assert N == 67108864
    assert num_bins == 4096

    partial_cap = 592
    chunk = 2048
    reduce_rows = 128
    reduce_bins = 8
    P = min(partial_cap, ct.cdiv(N, chunk))

    output = torch.empty((num_bins,), dtype=torch.int32, device=input.device)
    partials = torch.empty((P, num_bins), dtype=torch.int32, device=input.device)
    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (P,),
        _count_private,
        (input, partials, N, P, chunk, num_bins),
    )
    ct.launch(
        stream,
        (ct.cdiv(num_bins, reduce_bins),),
        _reduce_private,
        (partials, output, P, reduce_rows, reduce_bins),
    )
    return output


def get_last_config() -> dict:
    return {
        "partial_cap": 592,
        "count_chunk": 2048,
        "count_pipeline_lookahead": 1,
        "count_num_ctas": 1,
        "count_occupancy": 4,
        "count_opt_level": 3,
        "atomic_order": "relaxed",
        "atomic_scope": "block",
        "scratch_initialization": "owning_program",
        "zero_bins": 4096,
        "reduce_rows": 128,
        "reduce_bins": 8,
        "reduce_num_ctas": 1,
        "reduce_occupancy": 4,
        "reduce_opt_level": 3,
        "reduce_structure": "tile_accumulation_then_row_sum",
        "vector_static_shape_dims": (0,),
        "scratch_static_shape_dims": (0, 1),
        "allow_tma": False,
    }
