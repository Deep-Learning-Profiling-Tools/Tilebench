import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _layernorm_balanced(
    x, weight, bias, eps, row_tile, column_tile, normalization_period
):
    batch, m, k = x.shape
    rows = batch * m
    inv_k = 1.0 / k
    chunks = (k + column_tile - 1) // column_tile

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

    tiles = (rows + row_tile - 1) // row_tile
    pid = nl.program_id(0)
    programs = nl.num_programs()
    first = pid * tiles // programs
    last = (pid + 1) * tiles // programs

    for tile in range(first, last):
        r0 = tile * row_tile
        nr = min(row_tile, rows - r0)

        data = nl.ndarray((nr, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=data, src=x2[r0:r0 + nr, :])

        negative_mean = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=negative_mean,
            data=data,
            op=nl.add,
            axis=1,
            negate=True,
        )
        nisa.tensor_scalar(
            dst=negative_mean,
            data=negative_mean,
            op0=nl.multiply,
            operand0=inv_k,
            engine=nisa.engine.vector,
        )

        mean_square = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=mean_square,
            data1=negative_mean,
            data2=negative_mean,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )

        # ScalarE square-and-sum runs independently of the VectorE mean.
        square_sums = nl.ndarray((nr, chunks), dtype=nl.float32, buffer=nl.sbuf)
        for chunk in range(chunks):
            c0 = chunk * column_tile
            nc = min(column_tile, k - c0)
            squared = nl.ndarray((nr, nc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=squared,
                op=nl.square,
                data=data[:, c0:c0 + nc],
                reduce_op=nl.add,
                reduce_res=square_sums[:, chunk:chunk + 1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

        if chunks == 1:
            sum_squares = square_sums
        else:
            sum_squares = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=sum_squares,
                data=square_sums,
                op=nl.add,
                axis=1,
            )

        variance = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=variance,
            data=sum_squares,
            op0=nl.multiply,
            operand0=inv_k,
            op1=nl.subtract,
            operand1=mean_square,
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

        rstd = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=variance)

        normalization_bias = nl.ndarray(
            (nr, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=normalization_bias,
            data1=negative_mean,
            data2=rstd,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )

        for chunk in range(chunks):
            c0 = chunk * column_tile
            nc = min(column_tile, k - c0)

            # Distribute a fixed fraction of normalization to VectorE.
            # Rotating by row tile gives equal coverage for tail chunks.
            if (tile + chunk) % normalization_period == 0:
                nisa.tensor_scalar(
                    dst=data[:, c0:c0 + nc],
                    data=data[:, c0:c0 + nc],
                    op0=nl.multiply,
                    operand0=rstd,
                    op1=nl.add,
                    operand1=normalization_bias,
                    engine=nisa.engine.vector,
                )
            else:
                nisa.activation(
                    dst=data[:, c0:c0 + nc],
                    op=nl.copy,
                    data=data[:, c0:c0 + nc],
                    scale=rstd,
                    bias=normalization_bias,
                )

            nisa.tensor_tensor(
                dst=data[:, c0:c0 + nc],
                data1=data[:, c0:c0 + nc],
                data2=w[0:nr, c0:c0 + nc],
                op=nl.multiply,
                engine=nisa.engine.vector,
            )

            # Only the final affine add narrows to the output dtype.
            result = nl.ndarray((nr, nc), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=result,
                data1=data[:, c0:c0 + nc],
                data2=b[0:nr, c0:c0 + nc],
                op=nl.add,
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(
                dst=y2[r0:r0 + nr, c0:c0 + nc],
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
    normalization_period = 4

    # Budget for parameters, overlapped row tiles, and chunk temporaries.
    available = 212984 - 20 * k - 1024
    column_tile = min(
        k, 2048, max(512, (available // (6 * 512)) * 512)
    )

    _last_config = {
        "lnc": 2,
        "row_tile": row_tile,
        "column_tile": column_tile,
        "row_blocks": (rows + row_tile - 1) // row_tile,
        "column_chunks": (k + column_tile - 1) // column_tile,
        "statistics": "fp32_sum_and_sum_of_squares",
        "statistics_square_engine": "scalar",
        "mean_square_engine": "vector",
        "normalization_engine": "scalar_vector_interleaved",
        "vector_normalization_period": normalization_period,
        "vector_normalization_rule": "(row_tile + column_chunk) % period == 0",
        "affine_engine": "vector",
        "parameter_load": "broadcast_once_per_core",
        "output_cast": "final_affine_add",
        "row_loop": "unrolled",
    }

    return _layernorm_balanced[2](
        x,
        weight,
        bias,
        float(eps),
        row_tile,
        column_tile,
        normalization_period,
    )


def get_last_config() -> dict:
    return dict(_last_config)
