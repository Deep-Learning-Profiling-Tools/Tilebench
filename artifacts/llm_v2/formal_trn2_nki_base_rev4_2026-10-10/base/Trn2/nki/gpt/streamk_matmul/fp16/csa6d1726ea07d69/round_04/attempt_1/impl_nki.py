import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _zero_acc(a, b):
    M = a.shape[0]
    N = b.shape[1]
    cacc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = M // programs

    zeros = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0.0)

    for block in range(rows_per_program // 128):
        row = pid * rows_per_program + block * 128
        nisa.dma_copy(dst=cacc[row:row + 128, :], src=zeros)

    return cacc


def _pack_lhs(a, first_row, group_rows, iterations):
    K = a.shape[1]
    packed = nl.ndarray(
        (128, group_rows, iterations, 128),
        dtype=a.dtype,
        buffer=nl.sbuf,
    )
    native = nl.ndarray((128, K), dtype=a.dtype, buffer=nl.sbuf)

    for member in range(group_rows):
        row = first_row + member * 128
        nisa.dma_copy(dst=native, src=a[row:row + 128, :])

        for slab in range(iterations):
            transposed = nl.ndarray(
                (128, 128), dtype=a.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=transposed,
                data=native[:, slab * 128:(slab + 1) * 128],
                engine=nisa.engine.tensor,
            )
            nisa.tensor_copy(
                dst=packed[:, member, slab, :],
                src=transposed,
            )

    return packed


def _load_rhs(b, column_tile, tile_n, iterations, dynamic_column):
    N = b.shape[1]
    packed = nl.ndarray(
        (128, iterations, tile_n),
        dtype=b.dtype,
        buffer=nl.sbuf,
    )

    if dynamic_column:
        # The indirect dimension belongs to the underlying tensor view.
        # Its stride, and the corresponding AP stride, are both tile_n.
        column_view = b.reshape(
            (iterations, 128, N // tile_n, tile_n)
        )
        source = column_view.ap(
            pattern=[
                [N, 128],
                [128 * N, iterations],
                [tile_n, 1],
                [1, tile_n],
            ],
            offset=0,
            scalar_offset=column_tile,
            indirect_dim=2,
        )
    else:
        source = b.ap(
            pattern=[
                [N, 128],
                [128 * N, iterations],
                [1, tile_n],
            ],
            offset=column_tile * tile_n,
        )

    nisa.dma_copy(dst=packed, src=source)
    return packed


def _grouped_tile(tile_id, row_tiles, column_tiles):
    group_span = 8 * column_tiles
    group_id = tile_id // group_span
    first_row_tile = group_id * 8
    group_rows = min(8, row_tiles - first_row_tile)
    within_group = tile_id % group_span

    row_tile = first_row_tile + within_group % group_rows
    column_tile = within_group // group_rows
    return row_tile, column_tile


def _atomic_add_tile(cacc, partial, first_row, first_column, tile_n):
    N = cacc.shape[1]

    indices = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(
        dst=indices,
        pattern=[[1, 1]],
        offset=0,
        channel_multiplier=1,
    )

    # Row addressing is carried by the indirect indices, not duplicated
    # by a second affine partition stride.
    target = cacc.ap(
        pattern=[[0, 128], [1, tile_n]],
        offset=first_row * N + first_column,
        vector_offset=indices,
        indirect_dim=0,
    )
    nisa.dma_compute(
        dst=target,
        srcs=[target, partial],
        reduce_op=nl.add,
        unique_indices=True,
    )


def _stream_wave(a, b, cacc, tile_n, stream_tiles):
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    iterations = K // 128
    column_tiles = N // tile_n

    pid = nl.program_id(0)
    programs = nl.num_programs()
    total_iterations = stream_tiles * iterations
    quotient = total_iterations // programs
    remainder = total_iterations % programs

    cursor = pid * quotient + min(pid, remainder)
    limit = cursor + quotient
    if pid < remainder:
        limit = limit + 1

    while cursor < limit:
        tile_id = cursor // iterations
        first_k = cursor % iterations
        segment_end = min(limit, (tile_id + 1) * iterations)
        last_k = segment_end - tile_id * iterations

        row_tile, column_tile = _grouped_tile(
            tile_id, M // 128, column_tiles
        )
        lhs = _pack_lhs(a, row_tile * 128, 1, iterations)
        rhs = _load_rhs(
            b, column_tile, tile_n, iterations, False
        )

        acc = nl.ndarray(
            (128, tile_n), dtype=nl.float32, buffer=nl.psum
        )
        for slab in range(first_k, last_k):
            nisa.nc_matmul(
                dst=acc,
                stationary=lhs[:, 0, slab, :],
                moving=rhs[:, slab, :],
                accumulate=(slab > first_k),
            )

        partial = nl.ndarray(
            (128, tile_n), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=partial, src=acc)
        _atomic_add_tile(
            cacc,
            partial,
            row_tile * 128,
            column_tile * tile_n,
            tile_n,
        )
        cursor = segment_end


def _dp_column(
    lhs, b, destination, column_tile, group_rows,
    first_member, tile_n, iterations, dynamic_column
):
    rhs = _load_rhs(
        b, column_tile, tile_n, iterations, dynamic_column
    )

    # Reserve one PSUM bank for each row tile.
    acc = nl.ndarray(
        (128, group_rows, 512),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    for slab in range(iterations):
        for member in range(first_member, group_rows):
            nisa.nc_matmul(
                dst=acc[:, member, 0:tile_n],
                stationary=lhs[:, member, slab, :],
                moving=rhs[:, slab, :],
                accumulate=(slab > 0),
            )

    result = nl.ndarray(
        (128, group_rows, tile_n),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    N = b.shape[1]

    if dynamic_column:
        column_view = destination.reshape(
            (group_rows * 128, N // tile_n, tile_n)
        )

    for member in range(first_member, group_rows):
        nisa.tensor_copy(
            dst=result[:, member, :],
            src=acc[:, member, 0:tile_n],
        )

        if dynamic_column:
            target = column_view.ap(
                pattern=[[N, 128], [tile_n, 1], [1, tile_n]],
                offset=member * 128 * N,
                scalar_offset=column_tile,
                indirect_dim=1,
            )
        else:
            target = destination.ap(
                pattern=[[N, 128], [1, tile_n]],
                offset=member * 128 * N + column_tile * tile_n,
            )

        nisa.dma_copy(dst=target, src=result[:, member, :])


def _dp_group(a, b, cacc, first_row, group_rows, tile_n, stream_tiles):
    iterations = a.shape[1] // 128
    column_tiles = b.shape[1] // tile_n

    lhs = _pack_lhs(a, first_row, group_rows, iterations)
    destination = cacc[
        first_row:first_row + group_rows * 128, :
    ]

    # In the declared domain the Stream-K region is the first two
    # grouped row tiles in column zero.
    skip = min(group_rows, max(0, stream_tiles - first_row // 128))
    if skip < group_rows:
        _dp_column(
            lhs, b, destination, 0, group_rows, skip,
            tile_n, iterations, False
        )

    def column_body(column_tile):
        _dp_column(
            lhs, b, destination, column_tile, group_rows, 0,
            tile_n, iterations, True
        )

    nl.fori_loop(1, column_tiles, column_body)


@nki.jit
def _hybrid_matmul(a, b, cacc, tile_n, stream_tiles):
    pid = nl.program_id(0)
    programs = nl.num_programs()

    _stream_wave(a, b, cacc, tile_n, stream_tiles)

    groups = a.shape[0] // 1024
    full_pairs = groups // programs

    for pair in range(full_pairs):
        first_row = (pair * programs + pid) * 1024
        _dp_group(
            a, b, cacc, first_row, 8, tile_n, stream_tiles
        )

    if groups % programs != 0:
        first_row = full_pairs * programs * 1024 + pid * 512
        _dp_group(
            a, b, cacc, first_row, 4, tile_n, stream_tiles
        )

    return cacc


@nki.jit
def _final_cast(cacc):
    M, N = cacc.shape
    out = nl.ndarray(
        (M, N), dtype=nl.float16, buffer=nl.shared_hbm
    )

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = M // programs

    tile = nl.ndarray(
        (128, N), dtype=nl.float16, buffer=nl.sbuf
    )
    for block in range(rows_per_program // 128):
        row = pid * rows_per_program + block * 128
        nisa.dma_copy(dst=tile, src=cacc[row:row + 128, :])
        nisa.dma_copy(dst=out[row:row + 128, :], src=tile)

    return out


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    global _LAST_CONFIG

    M, K = a.shape
    N = b.shape[1]
    assert b.shape[0] == K
    assert K == 4096
    assert M % 1024 == 0 and N % 256 == 0
    assert a.dtype == torch.float16 and b.dtype == torch.float16

    tile_m = 128
    tile_k = 128
    tile_n = 512 if N % 512 == 0 else 256
    programs = 2

    tiles = (M // tile_m) * (N // tile_n)
    iterations = K // tile_k
    stream_tiles = tiles % programs
    if tiles - stream_tiles > programs:
        stream_tiles += programs

    _LAST_CONFIG = {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "programs": programs,
        "group_m": 8,
        "odd_group_rows_per_program": 4,
        "k_iterations": iterations,
        "streamk_tiles": stream_tiles,
        "data_parallel_tiles": tiles - stream_tiles,
        "swizzle": "grouped_m",
        "lhs_layout": "per_group_sbuf_transpose",
        "rhs_layout": "partition_k_slab",
        "rhs_indirect_view_rank": 4,
        "column_loop": "fori_loop",
        "accumulation_dtype": "float32",
        "atomic_reduction": "indirect_dma_compute_add",
        "atomic_partitions": 128,
        "atomic_affine_partition_stride": 0,
        "atomic_unique_indices": True,
        "zero_fill_rows": 128,
        "final_cast_rows": 128,
        "lnc": 2,
    }

    cacc = _zero_acc[2](a, b)
    cacc = _hybrid_matmul[2](a, b, cacc, tile_n, stream_tiles)
    return _final_cast[2](cacc)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
