import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _mean_rows(x, groups):
    M, N = x.shape

    P = 128
    segments = 4
    rows = P // segments
    width = N // segments
    reduction_width = min(512, width)
    rows_per_block = rows * groups
    blocks = M // rows_per_block

    out = nl.ndarray((M,), dtype=nl.float32, buffer=nl.shared_hbm)

    # Each partition contains one of four disjoint column segments.
    # Free-axis groups pack independent sets of 32 rows.
    xv = x.reshape((blocks, groups, P, width)).permute((0, 2, 1, 3))
    ov = out.reshape((blocks, groups, rows)).permute((0, 2, 1))

    # Block-diagonal 0/1 weights combine only segments of the same row.
    indices = nl.ndarray((P, rows), dtype=nl.float32, buffer=nl.sbuf)
    distance = nl.ndarray((P, rows), dtype=nl.float32, buffer=nl.sbuf)
    bound = nl.ndarray((P, rows), dtype=nl.float32, buffer=nl.sbuf)
    weights = nl.ndarray((P, rows), dtype=x.dtype, buffer=nl.sbuf)

    nisa.iota(
        dst=indices,
        pattern=[[-segments, rows]],
        offset=-(segments // 2),
        channel_multiplier=1,
    )
    nisa.activation(
        dst=distance,
        op=nl.abs,
        data=indices,
        bias=0.5,
    )
    nisa.memset(dst=bound, value=float(segments) / 2.0)
    nisa.tensor_tensor(
        dst=weights,
        data1=distance,
        data2=bound,
        op=nl.less,
    )

    programs = nl.num_programs()
    pid = nl.program_id(0)
    blocks_per_program = blocks // programs

    for i in range(blocks_per_program):
        block = pid * blocks_per_program + i

        values = nl.ndarray((P, groups, width), dtype=x.dtype, buffer=nl.sbuf)
        totals = nl.ndarray((rows, groups), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=values, src=xv[block])

        for g in range(groups):
            acc = nl.ndarray(
                (rows, reduction_width),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            # Multiplication by exact 0/1 weights introduces no narrowing;
            # every addition and all partial accumulation are float32.
            for c in range((width + reduction_width - 1) // reduction_width):
                c0 = c * reduction_width
                count = min(reduction_width, width - c0)
                nisa.nc_matmul(
                    dst=acc[:, 0:count],
                    stationary=weights,
                    moving=values[:, g, c0:c0 + count],
                    is_stationary_onezero=True,
                    accumulate=(c > 0),
                )

            partials = nl.ndarray(
                (rows, reduction_width),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
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
    if M != 2048 or N < 1024 or N % 1024 != 0:
        raise ValueError("Expected M=2048 and N to be a positive multiple of 1024.")
    if x.dtype != torch.float16:
        raise ValueError("Expected a float16 input.")

    width = N // 4
    groups = 8
    while groups < 32 and 2 * groups * width <= 8192:
        groups *= 2

    _last_config = {
        "lnc": 2,
        "partition_size": 128,
        "column_segments": 4,
        "rows_per_group": 32,
        "groups_per_tile": groups,
        "rows_per_tile": 32 * groups,
        "segment_width": width,
        "tensor_column_tile": min(512, width),
        "tiles_per_core": M // (64 * groups),
        "reduction_engine": "tensor_then_vector",
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
    }

    return _mean_rows[2](x, groups)


def get_last_config() -> dict:
    return dict(_last_config)
