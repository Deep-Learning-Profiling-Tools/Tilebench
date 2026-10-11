import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _mean_rows(x, groups):
    M, N = x.shape
    P = 128
    C = 512
    rows_per_block = P * groups
    blocks = M // rows_per_block

    out = nl.ndarray((M,), dtype=nl.float32, buffer=nl.shared_hbm)
    xv = x.reshape((blocks, groups, P, N)).permute((0, 2, 1, 3))
    ov = out.reshape((blocks, groups, P)).permute((0, 2, 1))

    # Exact identity weights preserve row independence. TensorE converts
    # the fp16 values into its float32 accumulation path.
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

    for i in range(blocks_per_program):
        block = pid * blocks_per_program + i
        values = nl.ndarray((P, groups, N), dtype=x.dtype, buffer=nl.sbuf)
        totals = nl.ndarray((P, groups), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=values, src=xv[block])

        for g in range(groups):
            acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.psum)

            # Accumulate disjoint column chunks into float32 column-lane
            # partials, then reduce those partials to one total per row.
            for c in range((N + C - 1) // C):
                c0 = c * C
                count = min(C, N - c0)
                nisa.nc_matmul(
                    dst=acc[:, 0:count],
                    stationary=weights,
                    moving=values[:, g, c0:c0 + count],
                    is_stationary_onezero=True,
                    accumulate=(c > 0),
                )

            partials = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=partials, src=acc)
            nisa.tensor_reduce(
                dst=totals[:, g:g + 1],
                op=nl.add,
                data=partials,
                axis=1,
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
    if M != 2048 or N < 1024 or N > 20480 or N % 1024 != 0:
        raise ValueError("Expected M=2048 and N in 1024..20480, divisible by 1024.")
    if x.dtype != torch.float16:
        raise ValueError("Expected a float16 input.")

    groups = 2
    while groups < 8 and 2 * groups * N <= 8192:
        groups *= 2

    _last_config = {
        "lnc": 2,
        "partition_size": 128,
        "column_segments": 1,
        "rows_per_group": 128,
        "groups_per_tile": groups,
        "rows_per_tile": 128 * groups,
        "segment_width": N,
        "tensor_column_tile": 512,
        "tiles_per_core": M // (256 * groups),
        "reduction_engine": "tensor_then_vector",
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
    }
    return _mean_rows[2](x, groups)


def get_last_config() -> dict:
    return dict(_last_config)
