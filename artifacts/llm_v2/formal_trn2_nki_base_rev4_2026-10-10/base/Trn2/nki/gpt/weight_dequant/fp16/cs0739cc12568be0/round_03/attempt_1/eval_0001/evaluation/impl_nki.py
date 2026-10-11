import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _weight_dequant(
    X, S, M, N, T, COL_TILE, STRIPE_GROUPS, MIN_STRIPED_BLOCKS
):
    Y = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    P = 128
    pid = nl.program_id(0)
    programs = nl.num_programs()
    row_tiles_per_core = (M // P) // programs
    scale_cols = (N + T - 1) // T
    stripe_rows = P // STRIPE_GROUPS

    for ri in range(row_tiles_per_core):
        r0 = (pid * row_tiles_per_core + ri) * P
        scale_row = r0 // T

        for ci in range((N + COL_TILE - 1) // COL_TILE):
            c0 = ci * COL_TILE
            width = min(COL_TILE, N - c0)
            blocks = width // T

            if blocks >= MIN_STRIPED_BLOCKS:
                partitions = STRIPE_GROUPS * blocks
                scales = nl.ndarray(
                    (partitions, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                tile = nl.ndarray(
                    (partitions, stripe_rows * T),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                # Each DMA has exactly "blocks" outer elements, matching
                # the partition count of its destination slice.
                for group in range(STRIPE_GROUPS):
                    p0 = group * blocks
                    stripe_r0 = r0 + group * stripe_rows

                    nisa.dma_copy(
                        dst=scales[p0:p0 + blocks, :],
                        src=S.ap(
                            pattern=[[1, blocks], [0, 1]],
                            offset=scale_row * scale_cols + c0 // T,
                        ),
                    )
                    nisa.dma_copy(
                        dst=tile[p0:p0 + blocks, :].reshape(
                            (blocks, stripe_rows, T)
                        ),
                        src=X.ap(
                            pattern=[
                                [T, blocks],
                                [N, stripe_rows],
                                [1, T],
                            ],
                            offset=stripe_r0 * N + c0,
                        ),
                    )

                # A partition contains a row stripe from one scale block.
                nisa.tensor_scalar(
                    dst=tile,
                    data=tile,
                    op0=nl.multiply,
                    operand0=scales,
                    engine=nisa.engine.vector,
                )

                for group in range(STRIPE_GROUPS):
                    p0 = group * blocks
                    stripe_r0 = r0 + group * stripe_rows
                    nisa.dma_copy(
                        dst=Y.ap(
                            pattern=[
                                [T, blocks],
                                [N, stripe_rows],
                                [1, T],
                            ],
                            offset=stripe_r0 * N + c0,
                        ),
                        src=tile[p0:p0 + blocks, :].reshape(
                            (blocks, stripe_rows, T)
                        ),
                    )
            else:
                # Narrow tiles retain full row-partition utilization.
                scales = nl.ndarray(
                    (P, blocks), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=scales,
                    src=S.ap(
                        pattern=[[0, P], [1, blocks]],
                        offset=scale_row * scale_cols + c0 // T,
                    ),
                )

                tile = nl.ndarray(
                    (P, width), dtype=X.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=tile,
                    src=X[r0:r0 + P, c0:c0 + width],
                )

                for block in range(blocks):
                    b0 = block * T
                    nisa.tensor_scalar(
                        dst=tile[:, b0:b0 + T],
                        data=tile[:, b0:b0 + T],
                        op0=nl.multiply,
                        operand0=scales[:, block:block + 1],
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

    column_tile = min(N, 32 * TILE_SIZE)
    stripe_groups = 4
    min_striped_blocks = 24

    _last_config = {
        "kernel": "weight_dequant",
        "lnc": 2,
        "row_tile": 128,
        "column_tile": column_tile,
        "max_column_blocks": 32,
        "stripe_groups": stripe_groups,
        "stripe_rows": 128 // stripe_groups,
        "min_striped_blocks": min_striped_blocks,
        "striped_free_tile": (128 // stripe_groups) * TILE_SIZE,
        "scale_block": TILE_SIZE,
        "scale_dtype": "float32",
        "multiply_engine": "vector",
        "row_tiles_per_core": M // 256,
    }

    return _weight_dequant[2](
        X,
        S,
        M,
        N,
        TILE_SIZE,
        column_tile,
        stripe_groups,
        min_striped_blocks,
    )


def get_last_config() -> dict:
    return dict(_last_config)
