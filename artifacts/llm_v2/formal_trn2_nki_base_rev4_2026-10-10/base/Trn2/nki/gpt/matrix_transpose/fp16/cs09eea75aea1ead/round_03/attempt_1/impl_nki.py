import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _matrix_transpose(x, tile_n, pipeline_depth):
    m, n = x.shape
    out = nl.ndarray((n, m), dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    columns_per_program = n // programs

    row_lanes = 16
    column_lanes = 128
    row_groups = m // row_lanes
    column_groups = tile_n // column_lanes
    column_tiles = columns_per_program // tile_n

    assert m % row_lanes == 0
    assert n % programs == 0
    assert columns_per_program % tile_n == 0
    assert tile_n % column_lanes == 0

    tile_shape = (column_lanes, column_groups, row_groups, row_lanes)
    buffer0 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 1:
        buffer1 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 2:
        buffer2 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
    if pipeline_depth > 3:
        buffer3 = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)

    for block in range(column_tiles):
        column_start = pid * columns_per_program + block * tile_n
        slot = block % pipeline_depth

        if slot == 0:
            transposed = buffer0
        elif slot == 1:
            transposed = buffer1
        elif slot == 2:
            transposed = buffer2
        else:
            transposed = buffer3

        source = x.ap(
            pattern=[
                [n, row_lanes],
                [column_lanes, column_groups],
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
                [column_lanes * m, column_groups],
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
    tile_n = 512
    column_tiles_per_core = n // (2 * tile_n)
    pipeline_depth = min(4, column_tiles_per_core)

    _LAST_CONFIG = {
        "tile_m": m,
        "tile_n": tile_n,
        "row_lanes": 16,
        "column_lanes": 128,
        "transpose": "dma_hwdge_4d",
        "lnc": 2,
        "split_axis": "input_columns",
        "row_tiles_per_core": 1,
        "column_tiles_per_core": column_tiles_per_core,
        "pipeline_depth": pipeline_depth,
        "loop": "unrolled_columns",
    }

    return _matrix_transpose[2](x, tile_n, pipeline_depth)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
