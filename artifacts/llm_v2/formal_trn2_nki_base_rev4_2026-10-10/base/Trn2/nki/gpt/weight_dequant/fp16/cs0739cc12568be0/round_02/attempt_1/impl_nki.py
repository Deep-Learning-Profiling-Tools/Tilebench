import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _weight_dequant(X, S, M, N, T, BLOCKS_PER_TILE):
    Y = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    row_tiles_per_core = (M // T) // programs
    scale_cols = N // T

    # Partition lanes combine a row stripe and a column block.
    # Every element within a lane therefore uses the same scale.
    stripe_groups = 128 // BLOCKS_PER_TILE
    stripe_rows = T // stripe_groups
    column_tile = BLOCKS_PER_TILE * T

    for ri in range(row_tiles_per_core):
        scale_row = pid * row_tiles_per_core + ri
        r0 = scale_row * T

        for ci in range((N + column_tile - 1) // column_tile):
            c0 = ci * column_tile
            blocks = min(BLOCKS_PER_TILE, (N - c0) // T)
            partitions = stripe_groups * blocks

            scales = nl.ndarray(
                (partitions, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=scales,
                src=S.ap(
                    pattern=[[0, stripe_groups], [1, blocks]],
                    offset=scale_row * scale_cols + c0 // T,
                ),
            )

            tile = nl.ndarray(
                (partitions, stripe_rows * T),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            # Flatten (stripe group, column block) into the partition
            # axis, and (row within stripe, column within block) into
            # the free axis.
            nisa.dma_copy(
                dst=tile,
                src=X.ap(
                    pattern=[
                        [stripe_rows * N, stripe_groups],
                        [T, blocks],
                        [N, stripe_rows],
                        [1, T],
                    ],
                    offset=r0 * N + c0,
                ),
            )

            nisa.tensor_scalar(
                dst=tile,
                data=tile,
                op0=nl.multiply,
                operand0=scales,
                engine=nisa.engine.vector,
            )

            nisa.dma_copy(
                dst=Y.ap(
                    pattern=[
                        [stripe_rows * N, stripe_groups],
                        [T, blocks],
                        [N, stripe_rows],
                        [1, T],
                    ],
                    offset=r0 * N + c0,
                ),
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

    blocks_per_tile = 1
    scale_columns = N // TILE_SIZE
    while blocks_per_tile < scale_columns and blocks_per_tile < 32:
        blocks_per_tile *= 2

    stripe_groups = 128 // blocks_per_tile
    stripe_rows = TILE_SIZE // stripe_groups

    _last_config = {
        "kernel": "weight_dequant",
        "lnc": 2,
        "row_tile": TILE_SIZE,
        "column_tile": blocks_per_tile * TILE_SIZE,
        "blocks_per_tile": blocks_per_tile,
        "stripe_groups": stripe_groups,
        "stripe_rows": stripe_rows,
        "partition_tile": 128,
        "free_tile": stripe_rows * TILE_SIZE,
        "scale_block": TILE_SIZE,
        "scale_dtype": "float32",
        "multiply_engine": "vector",
        "row_tiles_per_core": M // (2 * TILE_SIZE),
    }

    return _weight_dequant[2](
        X, S, M, N, TILE_SIZE, blocks_per_tile
    )


def get_last_config() -> dict:
    return dict(_last_config)
