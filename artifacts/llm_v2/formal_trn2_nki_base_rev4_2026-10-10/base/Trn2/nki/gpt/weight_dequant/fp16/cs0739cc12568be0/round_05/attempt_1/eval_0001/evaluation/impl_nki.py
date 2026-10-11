import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


def _apply_chunk(X, Y, scale, N, T, r0, c0, blocks):
    # Partition p represents (row_chunk, column_block). Each partition
    # contains several rows from one scale block, so one broadcast
    # multiplication handles the entire column chunk.
    row_chunks = 128 // blocks
    tile = nl.ndarray(
        (128, blocks * T), dtype=X.dtype, buffer=nl.sbuf
    )

    nisa.dma_copy(
        dst=tile,
        src=X.ap(
            pattern=[
                [N * blocks, row_chunks],
                [T, blocks],
                [N, blocks],
                [1, T],
            ],
            offset=r0 * N + c0,
        ),
    )

    nisa.tensor_scalar(
        dst=tile,
        data=tile,
        op0=nl.multiply,
        operand0=scale,
        engine=nisa.engine.vector,
    )

    nisa.dma_copy(
        dst=Y.ap(
            pattern=[
                [N * blocks, row_chunks],
                [T, blocks],
                [N, blocks],
                [1, T],
            ],
            offset=r0 * N + c0,
        ),
        src=tile,
    )


@nki.jit
def _weight_dequant(X, S, M, N, T, BLOCKS_PER_CHUNK):
    Y = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    P = 128
    pid = nl.program_id(0)
    programs = nl.num_programs()
    row_tiles_per_core = (M // P) // programs

    scale_cols = N // T
    full_chunks = scale_cols // BLOCKS_PER_CHUNK
    row_chunks = P // BLOCKS_PER_CHUNK

    for ri in range(row_tiles_per_core):
        r0 = (pid * row_tiles_per_core + ri) * P
        scale_row = r0 // T

        # Hoist compact scales for all full column chunks. The partition
        # order agrees with the swizzled weight tiles below.
        scales = nl.ndarray(
            (P, full_chunks), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=scales,
            src=S.ap(
                pattern=[
                    [0, row_chunks],
                    [1, BLOCKS_PER_CHUNK],
                    [BLOCKS_PER_CHUNK, full_chunks],
                ],
                offset=scale_row * scale_cols,
            ),
        )

        for ci in range(full_chunks):
            c0 = ci * BLOCKS_PER_CHUNK * T
            _apply_chunk(
                X,
                Y,
                scales[:, ci:ci + 1],
                N,
                T,
                r0,
                c0,
                BLOCKS_PER_CHUNK,
            )

        # Decompose the remaining block columns into descending powers
        # of two, retaining all 128 compute partitions for every chunk.
        next_block = full_chunks * BLOCKS_PER_CHUNK
        remaining = scale_cols - next_block
        tail_blocks = BLOCKS_PER_CHUNK // 2

        while tail_blocks >= 1:
            if remaining >= tail_blocks:
                tail_row_chunks = P // tail_blocks
                tail_scale = nl.ndarray(
                    (P, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=tail_scale,
                    src=S.ap(
                        pattern=[
                            [0, tail_row_chunks],
                            [1, tail_blocks],
                            [0, 1],
                        ],
                        offset=scale_row * scale_cols + next_block,
                    ),
                )

                _apply_chunk(
                    X,
                    Y,
                    tail_scale,
                    N,
                    T,
                    r0,
                    next_block * T,
                    tail_blocks,
                )

                next_block = next_block + tail_blocks
                remaining = remaining - tail_blocks

            tail_blocks = tail_blocks // 2

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

    scale_cols = N // TILE_SIZE
    blocks_per_chunk = 1
    limit = min(scale_cols, 16)
    while blocks_per_chunk * 2 <= limit:
        blocks_per_chunk *= 2

    _last_config = {
        "kernel": "weight_dequant",
        "lnc": 2,
        "mapping": "row_chunk_column_block_partitions",
        "row_tile": 128,
        "column_tile": blocks_per_chunk * TILE_SIZE,
        "blocks_per_chunk": blocks_per_chunk,
        "rows_per_partition": blocks_per_chunk,
        "row_chunks": 128 // blocks_per_chunk,
        "full_column_chunks": scale_cols // blocks_per_chunk,
        "tail_column_blocks": scale_cols % blocks_per_chunk,
        "tail_rule": "descending_powers_of_two",
        "scale_block": TILE_SIZE,
        "scale_dtype": "float32",
        "multiply_engine": "vector",
        "row_tiles_per_core": M // 256,
        "in_place_sbuf": True,
    }

    return _weight_dequant[2](
        X, S, M, N, TILE_SIZE, blocks_per_chunk
    )


def get_last_config() -> dict:
    return dict(_last_config)
