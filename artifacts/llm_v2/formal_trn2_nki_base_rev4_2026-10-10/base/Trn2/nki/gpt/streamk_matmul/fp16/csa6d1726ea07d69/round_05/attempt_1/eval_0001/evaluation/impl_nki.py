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
    local_rows = M // programs

    zeros = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0.0)

    for block in range(local_rows // 128):
        row = pid * local_rows + block * 128
        nisa.dma_copy(dst=cacc[row:row + 128, :], src=zeros)

    return cacc


def _pack_lhs(a, first_row, group_rows, iterations):
    K = a.shape[1]
    packed = nl.ndarray(
        (128, group_rows, iterations, 128),
        dtype=a.dtype,
        buffer=nl.sbuf,
    )

    for member in range(group_rows):
        row = first_row + member * 128
        native = nl.ndarray((128, K), dtype=a.dtype, buffer=nl.sbuf)
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


def _grouped_tile(tile_id, row_tiles, column_tiles):
    group_span = 4 * column_tiles
    group_id = tile_id // group_span
    first_row = group_id * 4
    group_rows = min(4, row_tiles - first_row)
    within_group = tile_id % group_span

    row_tile = first_row + within_group % group_rows
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
        first_row = row_tile * 128
        first_column = column_tile * tile_n

        lhs = _pack_lhs(a, first_row, 1, iterations)
        acc = nl.ndarray(
            (128, tile_n), dtype=nl.float32, buffer=nl.psum
        )

        for slab in range(first_k, last_k):
            rhs = nl.ndarray(
                (128, tile_n), dtype=b.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=rhs,
                src=b[
                    slab * 128:(slab + 1) * 128,
                    first_column:first_column + tile_n,
                ],
            )
            nisa.nc_matmul(
                dst=acc,
                stationary=lhs[:, 0, slab, :],
                moving=rhs,
                accumulate=(slab > first_k),
            )

        partial = nl.ndarray(
            (128, tile_n), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=partial, src=acc)
        _atomic_add_tile(
            cacc, partial, first_row, first_column, tile_n
        )
        cursor = segment_end


def _dp_column(
    lhs,
    rhs_column,
    destination_column,
    first_row,
    group_rows,
    first_member,
    tile_n,
    iterations,
):
    # Each row tile occupies a separate, bank-aligned PSUM region.
    acc = nl.ndarray(
        (128, group_rows, 512),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    for slab in range(iterations):
        rhs = nl.ndarray(
            (128, tile_n), dtype=rhs_column.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=rhs,
            src=rhs_column[slab * 128:(slab + 1) * 128, :],
        )

        for member in range(first_member, group_rows):
            nisa.nc_matmul(
                dst=acc[:, member, 0:tile_n],
                stationary=lhs[:, member, slab, :],
                moving=rhs,
                accumulate=(slab > 0),
            )

    for member in range(first_member, group_rows):
        result = nl.ndarray(
            (128, tile_n), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(
            dst=result,
            src=acc[:, member, 0:tile_n],
        )
        row = first_row + member * 128
        nisa.dma_copy(
            dst=destination_column[row:row + 128, :],
            src=result,
        )


def _dp_group(
    a,
    rhs_columns,
    destination_columns,
    first_row,
    tile_n,
    stream_tiles,
):
    iterations = a.shape[1] // 128
    column_tiles = rhs_columns.shape[0]
    group_rows = 4
    lhs = _pack_lhs(a, first_row, group_rows, iterations)

    # The configured domain has exactly two Stream-K tiles, both
    # in column zero of the first grouped row block.
    skip = min(group_rows, max(0, stream_tiles - first_row // 128))
    if skip < group_rows:
        _dp_column(
            lhs,
            rhs_columns[0],
            destination_columns[0],
            first_row,
            group_rows,
            skip,
            tile_n,
            iterations,
        )

    remaining_rhs = rhs_columns[1:column_tiles, :, :]
    remaining_destination = destination_columns[1:column_tiles, :, :]

    def column_body(column_index):
        _dp_column(
            lhs,
            remaining_rhs[column_index],
            remaining_destination[column_index],
            first_row,
            group_rows,
            0,
            tile_n,
            iterations,
        )

    nl.fori_loop(0, column_tiles - 1, column_body)


@nki.jit
def _hybrid_matmul(a, b, cacc, tile_n, stream_tiles):
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    column_tiles = N // tile_n

    pid = nl.program_id(0)
    programs = nl.num_programs()

    _stream_wave(a, b, cacc, tile_n, stream_tiles)

    # Runtime indexing is only on the outer column-tile dimension.
    # These are logical HBM views; neither operand is repacked here.
    rhs_columns = b.reshape((K, column_tiles, tile_n)).permute((1, 0, 2))
    destination_columns = cacc.reshape(
        (M, column_tiles, tile_n)
    ).permute((1, 0, 2))

    groups_per_program = M // (programs * 512)
    for group in range(groups_per_program):
        first_row = (group * programs + pid) * 512
        _dp_group(
            a,
            rhs_columns,
            destination_columns,
            first_row,
            tile_n,
            stream_tiles,
        )

    return cacc


@nki.jit
def _final_cast(cacc):
    M, N = cacc.shape
    out = nl.ndarray((M, N), dtype=nl.float16, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    local_rows = M // programs

    tile = nl.ndarray((128, N), dtype=nl.float16, buffer=nl.sbuf)
    for block in range(local_rows // 128):
        row = pid * local_rows + block * 128
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
        "lnc": 2,
        "group_m": 4,
        "k_iterations": iterations,
        "streamk_tiles": stream_tiles,
        "data_parallel_tiles": tiles - stream_tiles,
        "swizzle": "grouped_m",
        "lhs_layout": "group_sbuf_tensor_transpose",
        "rhs_layout": "native_strided_column_views",
        "rhs_load_rows": 128,
        "column_loop": "zero_based_fori_loop",
        "accumulation_dtype": "float32",
        "psum_bank_stride": 512,
        "atomic_reduction": "indirect_dma_compute_add",
        "atomic_unique_indices": True,
        "zero_fill_rows": 128,
        "final_cast_rows": 128,
    }

    cacc = _zero_acc[2](a, b)
    cacc = _hybrid_matmul[2](a, b, cacc, tile_n, stream_tiles)
    return _final_cast[2](cacc)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
