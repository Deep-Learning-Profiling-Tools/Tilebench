import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}
_STREAMING_BYTES = 24 * 1024 * 1024


@nki.jit
def _matrix_transpose(x, tile_n, pipeline_depth):
    m, n = x.shape
    out = nl.ndarray((n, m), dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()

    row_lanes = 16
    column_lanes = 128
    row_groups = m // row_lanes
    column_groups = tile_n // column_lanes
    column_tiles = n // (programs * tile_n)
    column_group_stride = programs * column_lanes

    assert m % row_lanes == 0
    assert tile_n % column_lanes == 0
    assert n % (programs * tile_n) == 0
    assert 1 <= pipeline_depth <= 12

    tile_shape = (column_lanes, column_groups, row_groups, row_lanes)
    buffer0 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 1:
        buffer1 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 2:
        buffer2 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 3:
        buffer3 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 4:
        buffer4 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 5:
        buffer5 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 6:
        buffer6 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 7:
        buffer7 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 8:
        buffer8 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 9:
        buffer9 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 10:
        buffer10 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 11:
        buffer11 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)

    for block in range(column_tiles):
        column_start = block * programs * tile_n + pid * column_lanes
        slot = block % pipeline_depth

        if slot == 0:
            transposed = buffer0
        elif slot == 1:
            transposed = buffer1
        elif slot == 2:
            transposed = buffer2
        elif slot == 3:
            transposed = buffer3
        elif slot == 4:
            transposed = buffer4
        elif slot == 5:
            transposed = buffer5
        elif slot == 6:
            transposed = buffer6
        elif slot == 7:
            transposed = buffer7
        elif slot == 8:
            transposed = buffer8
        elif slot == 9:
            transposed = buffer9
        elif slot == 10:
            transposed = buffer10
        else:
            transposed = buffer11

        source = x.ap(
            pattern=[
                [n, row_lanes],
                [column_group_stride, column_groups],
                [row_lanes * n, row_groups],
                [1, column_lanes],
            ],
            offset=column_start,
        )
        nisa.dma_transpose(
            dst=transposed,
            src=source,
            axes=(3, 1, 2, 0),
            dge_mode=nisa.dge_mode.hwdge,
        )

        destination = out.ap(
            pattern=[
                [m, column_lanes],
                [column_group_stride * m, column_groups],
                [1, m],
            ],
            offset=column_start * m,
        )
        nisa.dma_copy(
            dst=destination,
            src=transposed.reshape((column_lanes, column_groups, m)),
        )

    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG

    m, n = x.shape

    # FP16 input is divided evenly between two physical cores.
    per_core_input_bytes = m * n

    # Finer packets reduce pipeline startup for inputs whose entire
    # per-core working set fits in the streaming window.
    tile_n = 512
    if per_core_input_bytes <= _STREAMING_BYTES:
        tile_n = 256
    if per_core_input_bytes <= 4 * 1024 * 1024:
        tile_n = 128

    column_tiles_per_core = n // (2 * tile_n)
    tile_bytes = m * tile_n * 2
    pipeline_depth = min(
        column_tiles_per_core,
        _STREAMING_BYTES // tile_bytes,
    )

    _LAST_CONFIG = {
        "tile_m": m,
        "tile_n": tile_n,
        "row_lanes": 16,
        "column_lanes": 128,
        "column_group_stride": 256,
        "transpose": "dma_hwdge_4d",
        "lnc": 2,
        "split_axis": "striped_input_columns",
        "stripe_columns": 128,
        "row_tiles_per_core": 1,
        "column_tiles_per_core": column_tiles_per_core,
        "pipeline_depth": pipeline_depth,
        "streaming_bytes_per_core": pipeline_depth * tile_bytes,
        "loop": "unrolled_columns",
    }

    return _matrix_transpose[2](x, tile_n, pipeline_depth)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
