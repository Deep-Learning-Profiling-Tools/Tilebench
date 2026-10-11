import torch
import torch_neuronx
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


def _load_channels_first(x, r0, rows, c0, channels, C):
    tile = nl.ndarray(
        (channels, rows, 1), dtype=x.dtype, buffer=nl.sbuf
    )
    nisa.dma_transpose(
        dst=tile,
        src=x.ap(
            pattern=[[rows * C, 1], [C, rows], [1, channels]],
            offset=r0 * C + c0,
        ),
        axes=(2, 1, 0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    return tile.reshape((channels, rows))


def _store_channels_first(out, tile, r0, rows, c0, channels):
    for j in range((rows + 127) // 128):
        start = j * 128
        height = min(128, rows - start)

        transposed = nl.ndarray(
            (height, channels), dtype=tile.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(
            dst=transposed,
            data=tile[:, start:start + height],
            engine=nisa.engine.tensor,
        )
        staging = nl.ndarray(
            (height, channels), dtype=tile.dtype, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=staging, src=transposed)
        nisa.dma_copy(
            dst=out[
                r0 + start:r0 + start + height,
                c0:c0 + channels,
            ],
            src=staging,
        )


@nki.jit
def _batch_normalization(
    x, gamma, beta, N, C, eps, statistics_rows, apply_rows
):
    assert nl.num_programs() == 2
    assert C % 256 == 0

    pid = nl.program_id(0)
    F = 128
    channels_per_core = C // 2
    channel_blocks = channels_per_core // F
    num_blocks = (N + statistics_rows - 1) // statistics_rows
    num_apply_blocks = (N + apply_rows - 1) // apply_rows

    out = nl.ndarray((N, C), dtype=x.dtype, buffer=nl.shared_hbm)
    partials = nl.ndarray(
        (2, num_blocks, C), dtype=nl.float32, buffer=nl.shared_hbm
    )
    moments = nl.ndarray(
        (2, C), dtype=nl.float32, buffer=nl.shared_hbm
    )

    # First traversal: channels occupy partitions, so each row block
    # reduces directly along its contiguous free dimension.
    for cb in range(channel_blocks):
        c0 = pid * channels_per_core + cb * F

        for b in range(num_blocks):
            r0 = b * statistics_rows
            rows = min(statistics_rows, N - r0)

            loaded = _load_channels_first(x, r0, rows, c0, F, C)
            values = nl.ndarray(
                (F, rows), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=values, src=loaded)

            block_sum = nl.ndarray(
                (F, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            block_sumsq = nl.ndarray(
                (F, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            squares = nl.ndarray(
                (F, rows), dtype=nl.float32, buffer=nl.sbuf
            )

            nisa.tensor_reduce(
                dst=block_sum, op=nl.add, data=values, axis=1
            )
            nisa.activation(
                dst=squares,
                op=nl.square,
                data=values,
                reduce_op=nl.add,
                reduce_res=block_sumsq,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                engine=nisa.engine.scalar,
            )

            nisa.dma_copy(
                dst=partials.ap(
                    pattern=[[1, F], [C, 1]],
                    offset=b * C + c0,
                ),
                src=block_sum,
            )
            nisa.dma_copy(
                dst=partials.ap(
                    pattern=[[1, F], [C, 1]],
                    offset=(num_blocks + b) * C + c0,
                ),
                src=block_sumsq,
            )

    # Every partial is complete before any combine or second traversal.
    nisa.core_barrier(data=partials, cores=(0, 1))

    for cb in range(channel_blocks):
        c0 = pid * channels_per_core + cb * F

        sums = nl.ndarray(
            (F, num_blocks), dtype=nl.float32, buffer=nl.sbuf
        )
        sumsqs = nl.ndarray(
            (F, num_blocks), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=sums,
            src=partials.ap(
                pattern=[[1, F], [C, num_blocks]], offset=c0
            ),
        )
        nisa.dma_copy(
            dst=sumsqs,
            src=partials.ap(
                pattern=[[1, F], [C, num_blocks]],
                offset=num_blocks * C + c0,
            ),
        )

        total = nl.ndarray((F, 1), dtype=nl.float32, buffer=nl.sbuf)
        totalsq = nl.ndarray((F, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=total, op=nl.add, data=sums, axis=1)
        nisa.tensor_reduce(
            dst=totalsq, op=nl.add, data=sumsqs, axis=1
        )

        mean = nl.ndarray((F, 1), dtype=nl.float32, buffer=nl.sbuf)
        variance = nl.ndarray(
            (F, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        mean_square = nl.ndarray(
            (F, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        inv_std = nl.ndarray(
            (F, 1), dtype=nl.float32, buffer=nl.sbuf
        )

        nisa.tensor_scalar(
            dst=mean, data=total, op0=nl.multiply, operand0=1.0 / N
        )
        nisa.tensor_scalar(
            dst=variance,
            data=totalsq,
            op0=nl.multiply,
            operand0=1.0 / N,
        )
        nisa.tensor_tensor(
            dst=mean_square, data1=mean, data2=mean, op=nl.multiply
        )
        nisa.tensor_tensor(
            dst=variance,
            data1=variance,
            data2=mean_square,
            op=nl.subtract,
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
        nisa.activation(dst=inv_std, op=nl.rsqrt, data=variance)

        nisa.dma_copy(
            dst=moments.ap(
                pattern=[[1, F], [C, 1]], offset=c0
            ),
            src=mean,
        )
        nisa.dma_copy(
            dst=moments.ap(
                pattern=[[1, F], [C, 1]], offset=C + c0
            ),
            src=inv_std,
        )

        gamma_t = nl.ndarray(
            (F, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        beta_t = nl.ndarray(
            (F, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=gamma_t,
            src=gamma.reshape((C, 1))[c0:c0 + F, :],
        )
        nisa.dma_copy(
            dst=beta_t,
            src=beta.reshape((C, 1))[c0:c0 + F, :],
        )

        scale = nl.ndarray((F, 1), dtype=nl.float32, buffer=nl.sbuf)
        shift = nl.ndarray((F, 1), dtype=nl.float32, buffer=nl.sbuf)
        mean_scale = nl.ndarray(
            (F, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=scale, data1=inv_std, data2=gamma_t, op=nl.multiply
        )
        nisa.tensor_tensor(
            dst=mean_scale, data1=mean, data2=scale, op=nl.multiply
        )
        nisa.tensor_tensor(
            dst=shift, data1=beta_t, data2=mean_scale, op=nl.subtract
        )

        # Second traversal. ScalarE promotes input values to fp32,
        # performs both affine operations, then casts the final result.
        for b in range(num_apply_blocks):
            r0 = b * apply_rows
            rows = min(apply_rows, N - r0)
            normalized = _load_channels_first(
                x, r0, rows, c0, F, C
            )
            nisa.tensor_scalar(
                dst=normalized,
                data=normalized,
                op0=nl.multiply,
                operand0=scale,
                op1=nl.add,
                operand1=shift,
                engine=nisa.engine.scalar,
            )
            _store_channels_first(
                out, normalized, r0, rows, c0, F
            )

    return out


def run(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    N: int,
    C: int,
    eps: float,
    **kwargs,
):
    global _last_config

    statistics_rows = 4096
    apply_rows = 2048
    _last_config = {
        "lnc": 2,
        "channel_tile": 128,
        "channels_per_core": C // 2,
        "statistics_row_block": statistics_rows,
        "apply_row_block": apply_rows,
        "partial_blocks": (N + statistics_rows - 1) // statistics_rows,
        "apply_blocks": (N + apply_rows - 1) // apply_rows,
        "statistics_dtype": "float32",
        "layout": "channels_partition_rows_free",
        "input_transpose": "hwdge_3d",
        "reduction": "free_axis_block_then_free_axis_partials",
        "square_reduction": "fused_scalar",
        "affine": "fp32_scalar_multiply_add",
        "output_transpose": "tensor_128",
        "tail": "clipped",
        "launches": 1,
        "global_barriers": 1,
    }
    return _batch_normalization[2](
        input, gamma, beta, N, C, eps, statistics_rows, apply_rows
    )


def get_last_config() -> dict:
    return dict(_last_config)
