import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _mean_rows(x, rows_per_partition):
    M, N = x.shape
    P = 128
    G = rows_per_partition

    programs = nl.num_programs()
    pid = nl.program_id(0)
    blocks = M // (P * G)
    blocks_per_program = blocks // programs

    out = nl.ndarray((M,), dtype=nl.float32, buffer=nl.shared_hbm)

    # Each free-axis group represents an independent row.
    xv = x.reshape((blocks, P, G, N))
    ov = out.reshape((blocks, P, G))

    for i in range(blocks_per_program):
        block = pid * blocks_per_program + i

        values = nl.ndarray((P, G, N), dtype=x.dtype, buffer=nl.sbuf)
        totals = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)

        nisa.dma_copy(dst=values, src=xv[block])

        # VectorE converts inputs to FP32 before arithmetic. The FP32
        # destination also preserves the full-precision row totals.
        nisa.tensor_reduce(
            dst=totals,
            op=nl.add,
            data=values,
            axis=2,
        )
        nisa.tensor_scalar(
            dst=totals,
            data=totals,
            op0=nl.multiply,
            operand0=1.0 / N,
            engine=nisa.engine.scalar,
        )
        nisa.dma_copy(dst=ov[block], src=totals)

    return out


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    global _last_config

    if x.ndim != 2 or dim != 1:
        raise ValueError("Expected a two-dimensional input and dim=1.")

    M, N = x.shape
    if M != 2048 or N <= 0:
        raise ValueError("Expected M=2048 and a positive row length.")
    if x.dtype != torch.float16:
        raise ValueError("Expected a float16 input.")

    # Pack short rows into the free dimensions, while retaining enough
    # independent tiles to pipeline larger reductions.
    groups = 1
    while groups < 8 and (2 * groups * N) <= 8192:
        groups *= 2

    _last_config = {
        "lnc": 2,
        "partition_rows": 128,
        "rows_per_partition": groups,
        "rows_per_tile": 128 * groups,
        "column_tile": N,
        "tiles_per_core": M // (256 * groups),
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
        "reduction_axis": 2,
    }

    return _mean_rows[2](x, groups)


def get_last_config() -> dict:
    return dict(_last_config)
