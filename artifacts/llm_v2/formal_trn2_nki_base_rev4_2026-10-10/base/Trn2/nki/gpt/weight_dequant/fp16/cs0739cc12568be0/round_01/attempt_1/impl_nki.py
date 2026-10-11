import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _weight_dequant(X, S, M, N, T, COL_TILE):
    Y = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    P = 128
    pid = nl.program_id(0)
    programs = nl.num_programs()
    row_tiles_per_core = (M // P) // programs
    scale_cols = (N + T - 1) // T

    for ri in range(row_tiles_per_core):
        r0 = (pid * row_tiles_per_core + ri) * P
        scale_row = r0 // T

        # Hoist the compact scale row and broadcast it across partitions.
        scales = nl.ndarray(
            (P, scale_cols), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=scales,
            src=S.ap(
                pattern=[[0, P], [1, scale_cols]],
                offset=scale_row * scale_cols,
            ),
        )

        for ci in range((N + COL_TILE - 1) // COL_TILE):
            c0 = ci * COL_TILE
            width = min(COL_TILE, N - c0)

            tile = nl.ndarray(
                (P, width), dtype=X.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=tile,
                src=X[r0:r0 + P, c0:c0 + width],
            )

            for bi in range(width // T):
                b0 = bi * T
                sc = c0 // T + bi
                nisa.tensor_scalar(
                    dst=tile[:, b0:b0 + T],
                    data=tile[:, b0:b0 + T],
                    op0=nl.multiply,
                    operand0=scales[:, sc:sc + 1],
                    engine=nisa.engine.vector,
                )

            nisa.dma_copy(
                dst=Y[r0:r0 + P, c0:c0 + width],
                src=tile,
            )

    return Y


def run(
    X: torch.Tensor,
    S: torch.Tensor,
    M: int,
    N: int,
    TILE_SIZE: int,
    **kwargs,
):
    global _last_config

    column_tile = min(N, 2048)
    _last_config = {
        "kernel": "weight_dequant",
        "lnc": 2,
        "row_tile": 128,
        "column_tile": column_tile,
        "scale_block": TILE_SIZE,
        "scale_dtype": "float32",
        "multiply_engine": "vector",
        "row_tiles_per_core": M // 256,
    }

    return _weight_dequant[2](X, S, M, N, TILE_SIZE, column_tile)


def get_last_config() -> dict:
    return dict(_last_config)
