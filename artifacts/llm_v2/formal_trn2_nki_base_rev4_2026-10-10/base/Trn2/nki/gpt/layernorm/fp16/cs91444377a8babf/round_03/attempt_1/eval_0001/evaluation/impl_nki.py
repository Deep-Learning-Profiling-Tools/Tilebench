import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _layernorm_buffered(x, weight, bias, eps, row_tile, column_tile):
    batch, m, k = x.shape
    rows = batch * m
    inv_k = 1.0 / k

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x2 = x.reshape((rows, k))
    y2 = out.reshape((rows, k))

    # Read each affine parameter once, then broadcast on chip.
    weight_row = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=weight_row, src=weight.reshape((1, k)))
    w = nl.broadcast_to(weight_row, (row_tile, k))

    bias_row = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_row, src=bias.reshape((1, k)))
    b = nl.broadcast_to(bias_row, (row_tile, k))

    # Explicitly reuse two full-row buffers. This leaves enough SBUF for
    # a full-row square/reduction even at the largest configured K.
    data0 = nl.ndarray((row_tile, k), dtype=nl.float32, buffer=nl.sbuf)
    data1 = nl.ndarray((row_tile, k), dtype=nl.float32, buffer=nl.sbuf)
    squared = nl.ndarray((row_tile, k), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray(
        (row_tile, column_tile), dtype=x.dtype, buffer=nl.sbuf
    )

    tiles = (rows + row_tile - 1) // row_tile
    pid = nl.program_id(0)
    programs = nl.num_programs()
    first = pid * tiles // programs
    last = (pid + 1) * tiles // programs
    chunks = (k + column_tile - 1) // column_tile

    for tile in range(first, last):
        r0 = tile * row_tile
        nr = min(row_tile, rows - r0)

        if (tile - first) % 2 == 0:
            data = data0
        else:
            data = data1

        nisa.dma_copy(
            dst=data[0:nr, :],
            src=x2[r0:r0 + nr, :],
        )

        negative_mean = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=negative_mean,
            data=data[0:nr, :],
            op=nl.add,
            axis=1,
            negate=True,
        )

        # VectorE computes the sum while ScalarE computes square and sum.
        # One full-row activation avoids per-chunk statistics overhead.
        sum_squares = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=squared[0:nr, :],
            op=nl.square,
            data=data[0:nr, :],
            reduce_op=nl.add,
            reduce_res=sum_squares,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        nisa.tensor_scalar(
            dst=negative_mean,
            data=negative_mean,
            op0=nl.multiply,
            operand0=inv_k,
        )

        mean_square = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=mean_square,
            op=nl.square,
            data=negative_mean,
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

        # Chunked apply overlaps ScalarE normalization, VectorE affine
        # arithmetic and DMA. Only the final affine add narrows to fp16.
        for chunk in range(chunks):
            c0 = chunk * column_tile
            nc = min(column_tile, k - c0)

            nisa.activation(
                dst=data[0:nr, c0:c0 + nc],
                op=nl.copy,
                data=data[0:nr, c0:c0 + nc],
                scale=rstd,
                bias=normalization_bias,
            )
            nisa.tensor_tensor(
                dst=data[0:nr, c0:c0 + nc],
                data1=data[0:nr, c0:c0 + nc],
                data2=w[0:nr, c0:c0 + nc],
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=result[0:nr, 0:nc],
                data1=data[0:nr, c0:c0 + nc],
                data2=b[0:nr, c0:c0 + nc],
                op=nl.add,
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(
                dst=y2[r0:r0 + nr, c0:c0 + nc],
                src=result[0:nr, 0:nc],
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
    column_tile = min(k, 2048)

    _last_config = {
        "lnc": 2,
        "row_tile": row_tile,
        "column_tile": column_tile,
        "row_blocks": (rows + row_tile - 1) // row_tile,
        "column_chunks": (k + column_tile - 1) // column_tile,
        "statistics": "fp32_sum_and_sum_of_squares",
        "statistics_column_tile": k,
        "data_buffers": 2,
        "normalization_engine": "scalar",
        "affine_engine": "vector",
        "parameter_load": "single_row_then_on_chip_broadcast",
        "output_cast": "final_affine_add",
        "row_loop": "unrolled_reused_buffers",
    }

    return _layernorm_buffered[2](
        x, weight, bias, float(eps), row_tile, column_tile
    )


def get_last_config() -> dict:
    return dict(_last_config)
