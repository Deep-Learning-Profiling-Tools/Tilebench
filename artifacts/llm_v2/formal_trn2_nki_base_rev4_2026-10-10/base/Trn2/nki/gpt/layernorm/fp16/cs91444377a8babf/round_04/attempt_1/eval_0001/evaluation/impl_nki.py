import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _layernorm_grouped(x, weight, bias, eps, row_tile, row_group, column_tile):
    batch, m, k = x.shape
    rows = batch * m
    inv_k = 1.0 / k
    chunks = (k + column_tile - 1) // column_tile
    group_rows = row_tile * row_group
    assert rows % group_rows == 0

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x2 = x.reshape((rows, k))
    y2 = out.reshape((rows, k))

    w = nl.ndarray((row_tile, k), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((row_tile, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w,
        src=weight.reshape((1, k)).ap(
            pattern=[[0, row_tile], [1, k]], offset=0
        ),
    )
    nisa.dma_copy(
        dst=b,
        src=bias.reshape((1, k)).ap(
            pattern=[[0, row_tile], [1, k]], offset=0
        ),
    )

    groups = rows // group_rows
    pid = nl.program_id(0)
    programs = nl.num_programs()
    first = pid * groups // programs
    last = (pid + 1) * groups // programs

    for group in range(first, last):
        r0 = group * group_rows

        # The extra free axis batches independent rows, allowing the
        # short statistics operations to be shared across row tiles.
        data = nl.ndarray(
            (row_tile, row_group, k), dtype=nl.float32, buffer=nl.sbuf
        )
        for j in range(row_group):
            start = r0 + j * row_tile
            nisa.dma_copy(
                dst=data[:, j, :],
                src=x2[start:start + row_tile, :],
            )

        negative_mean = nl.ndarray(
            (row_tile, row_group), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_reduce(
            dst=negative_mean,
            data=data,
            op=nl.add,
            axis=2,
            negate=True,
        )
        nisa.tensor_scalar(
            dst=negative_mean,
            data=negative_mean,
            op0=nl.multiply,
            operand0=inv_k,
            engine=nisa.engine.vector,
        )

        # Keep the small mean-square operation on VectorE, independent
        # of the long ScalarE square-and-sum stream.
        mean_square = nl.ndarray(
            (row_tile, row_group), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=mean_square,
            data1=negative_mean,
            data2=negative_mean,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )

        square_sums = nl.ndarray(
            (row_tile, row_group, chunks),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        for j in range(row_group):
            for chunk in range(chunks):
                c0 = chunk * column_tile
                nc = min(column_tile, k - c0)
                squared = nl.ndarray(
                    (row_tile, nc), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(
                    dst=squared,
                    op=nl.square,
                    data=data[:, j, c0:c0 + nc],
                    reduce_op=nl.add,
                    reduce_res=square_sums[:, j, chunk:chunk + 1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )

        if chunks == 1:
            sum_squares = square_sums.reshape((row_tile, row_group))
        else:
            sum_squares = nl.ndarray(
                (row_tile, row_group), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_reduce(
                dst=sum_squares,
                data=square_sums,
                op=nl.add,
                axis=2,
            )

        variance = nl.ndarray(
            (row_tile, row_group), dtype=nl.float32, buffer=nl.sbuf
        )
        if row_group == 1:
            nisa.tensor_scalar(
                dst=variance,
                data=sum_squares,
                op0=nl.multiply,
                operand0=inv_k,
                op1=nl.subtract,
                operand1=mean_square,
                engine=nisa.engine.vector,
            )
        else:
            nisa.tensor_scalar(
                dst=variance,
                data=sum_squares,
                op0=nl.multiply,
                operand0=inv_k,
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=variance,
                data1=variance,
                data2=mean_square,
                op=nl.subtract,
                engine=nisa.engine.vector,
            )

        nisa.tensor_scalar(
            dst=variance,
            data=variance,
            op0=nl.maximum,
            operand0=0.0,
            op1=nl.add,
            operand1=eps,
            engine=nisa.engine.vector,
        )

        rstd = nl.ndarray(
            (row_tile, row_group), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(dst=rstd, op=nl.rsqrt, data=variance)

        normalization_bias = nl.ndarray(
            (row_tile, row_group), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=normalization_bias,
            data1=negative_mean,
            data2=rstd,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )

        for j in range(row_group):
            start = r0 + j * row_tile
            for chunk in range(chunks):
                c0 = chunk * column_tile
                nc = min(column_tile, k - c0)

                nisa.activation(
                    dst=data[:, j, c0:c0 + nc],
                    op=nl.copy,
                    data=data[:, j, c0:c0 + nc],
                    scale=rstd[:, j:j + 1],
                    bias=normalization_bias[:, j:j + 1],
                )
                nisa.tensor_tensor(
                    dst=data[:, j, c0:c0 + nc],
                    data1=data[:, j, c0:c0 + nc],
                    data2=w[:, c0:c0 + nc],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )

                result = nl.ndarray(
                    (row_tile, nc), dtype=x.dtype, buffer=nl.sbuf
                )
                nisa.tensor_tensor(
                    dst=result,
                    data1=data[:, j, c0:c0 + nc],
                    data2=b[:, c0:c0 + nc],
                    op=nl.add,
                    engine=nisa.engine.vector,
                )
                nisa.dma_copy(
                    dst=y2[start:start + row_tile, c0:c0 + nc],
                    src=result,
                )

    return out


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    global _last_config

    rows = x.shape[0] * x.shape[1]
    k = x.shape[2]
    row_tile = 128

    available = 212984 - 20 * k - 1024
    column_tile = min(
        k, 2048, max(512, (available // (6 * 512)) * 512)
    )

    # Select a fixed grouping while retaining space for two groups,
    # affine parameters, chunk temporaries, and scheduling headroom.
    if (
        rows % (4 * row_tile) == 0
        and 40 * k + 6 * column_tile + 8192 <= 212984
    ):
        row_group = 4
    elif (
        rows % (2 * row_tile) == 0
        and 24 * k + 6 * column_tile + 8192 <= 212984
    ):
        row_group = 2
    else:
        row_group = 1

    _last_config = {
        "lnc": 2,
        "row_tile": row_tile,
        "row_group": row_group,
        "rows_per_group": row_tile * row_group,
        "row_blocks": rows // row_tile,
        "row_group_blocks": rows // (row_tile * row_group),
        "column_tile": column_tile,
        "column_chunks": (k + column_tile - 1) // column_tile,
        "statistics": "fp32_grouped_sum_and_sum_of_squares",
        "mean_square_engine": "vector",
        "normalization_engine": "scalar",
        "affine_engine": "vector",
        "parameter_load": "broadcast_once_per_core",
        "output_cast": "final_affine_add",
        "row_loop": "unrolled_grouped",
    }

    return _layernorm_grouped[2](
        x, weight, bias, float(eps), row_tile, row_group, column_tile
    )


def get_last_config() -> dict:
    return dict(_last_config)
