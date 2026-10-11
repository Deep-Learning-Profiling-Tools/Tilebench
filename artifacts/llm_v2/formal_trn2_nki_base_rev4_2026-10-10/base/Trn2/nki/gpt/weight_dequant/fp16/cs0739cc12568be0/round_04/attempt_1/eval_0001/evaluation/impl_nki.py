import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _weight_dequant(X, S, M, N, T, SCALAR_PERIOD):
    Y = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    P = 128
    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_core = (M // P) // programs
    scale_cols = N // T

    for ri in range(rows_per_core):
        r0 = (pid * rows_per_core + ri) * P
        scale_row = r0 // T

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

        data = nl.ndarray((P, N), dtype=X.dtype, buffer=nl.sbuf)
        result = nl.ndarray((P, N), dtype=X.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=data, src=X[r0:r0 + P, :])

        for block in range(scale_cols):
            c0 = block * T
            if block % SCALAR_PERIOD == SCALAR_PERIOD - 1:
                nisa.tensor_scalar(
                    dst=result[:, c0:c0 + T],
                    data=data[:, c0:c0 + T],
                    op0=nl.multiply,
                    operand0=scales[:, block:block + 1],
                    engine=nisa.engine.scalar,
                )
            else:
                nisa.tensor_scalar(
                    dst=result[:, c0:c0 + T],
                    data=data[:, c0:c0 + T],
                    op0=nl.multiply,
                    operand0=scales[:, block:block + 1],
                    engine=nisa.engine.vector,
                )

        nisa.dma_copy(dst=Y[r0:r0 + P, :], src=result)

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

    scalar_period = 4
    _last_config = {
        "kernel": "weight_dequant",
        "lnc": 2,
        "row_tile": 128,
        "column_tile": N,
        "scale_block": TILE_SIZE,
        "scale_dtype": "float32",
        "multiply_engines": ["vector", "scalar"],
        "scalar_period": scalar_period,
        "scalar_block_remainder": scalar_period - 1,
        "row_tiles_per_core": M // 256,
        "separate_result_tile": True,
    }

    return _weight_dequant[2](
        X, S, M, N, TILE_SIZE, scalar_period
    )


def get_last_config() -> dict:
    return dict(_last_config)
