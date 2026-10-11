import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _matrix_transpose(x, tile_m, tile_n):
    m, n = x.shape
    out = nl.ndarray((n, m), dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = m // programs

    row_lanes = 16
    column_lanes = 128
    row_groups = tile_m // row_lanes
    column_groups = tile_n // column_lanes

    assert m % programs == 0
    assert rows_per_program % tile_m == 0
    assert n % tile_n == 0

    def column_body(column_start):
        for block in range(rows_per_program // tile_m):
            row_start = pid * rows_per_program + block * tile_m

            # Logical source coordinates:
            #   row = row_start + lane + 16 * row_group
            #   col = column_start + 128 * column_group + col_lane
            source = x.ap(
                pattern=[
                    [n, row_lanes],
                    [column_lanes, column_groups],
                    [row_lanes * n, row_groups],
                    [1, column_lanes],
                ],
                offset=row_start * n,
                scalar_offset=column_start,
                indirect_dim=3,
            )

            transposed = nl.ndarray(
                (column_lanes, column_groups, row_groups, row_lanes),
                dtype=x.dtype,
                buffer=nl.sbuf,
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
                    [1, tile_m],
                ],
                offset=row_start,
                scalar_offset=column_start,
                indirect_dim=0,
            )
            nisa.dma_copy(
                dst=destination,
                src=transposed.reshape(
                    (column_lanes, column_groups, tile_m)
                ),
            )

    nl.fori_loop(0, n, column_body, step=tile_n)
    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG

    m, n = x.shape
    tile_m = 1024
    tile_n = 1024

    _LAST_CONFIG = {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "row_lanes": 16,
        "column_lanes": 128,
        "transpose": "dma_hwdge_4d",
        "lnc": 2,
        "split_axis": "input_rows",
        "row_tiles_per_core": m // (2 * tile_m),
        "column_tiles": n // tile_n,
        "loop": "fori_columns",
    }

    return _matrix_transpose[2](x, tile_m, tile_n)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
