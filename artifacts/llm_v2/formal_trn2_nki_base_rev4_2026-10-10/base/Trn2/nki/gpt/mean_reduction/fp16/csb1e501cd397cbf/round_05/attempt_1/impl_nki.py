import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _mean_rows(x, groups, vector_columns):
    M, N = x.shape
    P = 128
    C = 512
    blocks = M // (P * groups)

    out = nl.ndarray((M,), dtype=nl.float32, buffer=nl.shared_hbm)
    xv = x.reshape((blocks, groups, P, N)).permute((0, 2, 1, 3))
    ov = out.reshape((blocks, groups, P)).permute((0, 2, 1))

    indices = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    zero = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    weights = nl.ndarray((P, P), dtype=x.dtype, buffer=nl.sbuf)
    nisa.iota(
        dst=indices,
        pattern=[[-1, P]],
        offset=0,
        channel_multiplier=1,
    )
    nisa.memset(dst=zero, value=0.0)
    nisa.tensor_tensor(
        dst=weights,
        data1=indices,
        data2=zero,
        op=nl.equal,
    )

    programs = nl.num_programs()
    pid = nl.program_id(0)
    blocks_per_program = blocks // programs
    tensor_chunks = (N - vector_columns) // C

    for i in range(blocks_per_program):
        block = pid * blocks_per_program + i
        values = nl.ndarray((P, groups, N), dtype=x.dtype, buffer=nl.sbuf)
        vector_totals = nl.ndarray(
            (P, groups), dtype=nl.float32, buffer=nl.sbuf
        )
        tensor_totals = nl.ndarray(
            (P, groups), dtype=nl.float32, buffer=nl.sbuf
        )
        acc = nl.ndarray(
            (P, groups, C), dtype=nl.float32, buffer=nl.psum
        )

        nisa.dma_copy(dst=values, src=xv[block])

        # VectorE sums one disjoint part of each row in float32 while
        # TensorE accumulates the remaining columns in float32 PSUM.
        nisa.tensor_reduce(
            dst=vector_totals,
            op=nl.add,
            data=values[:, :, 0:vector_columns],
            axis=2,
        )

        for g in range(groups):
            for c in range(tensor_chunks):
                c0 = vector_columns + c * C
                nisa.nc_matmul(
                    dst=acc[:, g, :],
                    stationary=weights,
                    moving=values[:, g, c0:c0 + C],
                    is_stationary_onezero=True,
                    accumulate=(c > 0),
                )

        nisa.tensor_reduce(
            dst=tensor_totals,
            op=nl.add,
            data=acc,
            axis=2,
        )
        nisa.tensor_tensor(
            dst=vector_totals,
            data1=vector_totals,
            data2=tensor_totals,
            op=nl.add,
        )
        nisa.tensor_scalar(
            dst=vector_totals,
            data=vector_totals,
            op0=nl.multiply,
            operand0=1.0 / N,
            engine=nisa.engine.scalar,
        )
        nisa.dma_copy(dst=ov[block], src=vector_totals)

    return out


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    global _last_config

    if x.ndim != 2 or dim != 1:
        raise ValueError("Expected a two-dimensional input and dim=1.")

    M, N = x.shape
    if M != 2048 or N < 1024 or N > 20480 or N % 1024 != 0:
        raise ValueError("Expected M=2048 and N in 1024..20480, divisible by 1024.")
    if x.dtype != torch.float16:
        raise ValueError("Expected a float16 input.")

    groups = 2
    while groups < 8 and 2 * groups * N <= 8192:
        groups *= 2
    vector_columns = N // 2

    _last_config = {
        "lnc": 2,
        "partition_size": 128,
        "rows_per_group": 128,
        "groups_per_tile": groups,
        "rows_per_tile": 128 * groups,
        "tiles_per_core": M // (256 * groups),
        "vector_columns": vector_columns,
        "tensor_columns": N - vector_columns,
        "tensor_column_tile": 512,
        "reduction_engine": "parallel_tensor_and_vector",
        "direct_psum_reduction": True,
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
    }
    return _mean_rows[2](x, groups, vector_columns)


def get_last_config() -> dict:
    return dict(_last_config)
