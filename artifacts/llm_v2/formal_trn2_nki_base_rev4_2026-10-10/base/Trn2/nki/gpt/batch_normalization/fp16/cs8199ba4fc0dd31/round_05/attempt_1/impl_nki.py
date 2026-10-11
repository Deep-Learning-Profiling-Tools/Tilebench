import torch
import torch_neuronx
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


def _load_statistics_tile(x, r0, rows, c0, channels, C):
    tile = nl.ndarray(
        (channels, rows, 1), dtype=x.dtype, buffer=nl.sbuf
    )

    # HWDGE requires the number of source rows to be divisible by 16.
    aligned_rows = (rows // 16) * 16
    if aligned_rows > 0:
        nisa.dma_transpose(
            dst=tile[:, 0:aligned_rows, :],
            src=x.ap(
                pattern=[
                    [aligned_rows * C, 1],
                    [C, aligned_rows],
                    [1, channels],
                ],
                offset=r0 * C + c0,
            ),
            axes=(2, 1, 0),
            dge_mode=nisa.dge_mode.hwdge,
        )

    if aligned_rows < rows:
        tail_rows = rows - aligned_rows
        nisa.dma_copy(
            dst=tile[:, aligned_rows:rows, :],
            src=x.ap(
                pattern=[[1, channels], [C, tail_rows], [1, 1]],
                offset=(r0 + aligned_rows) * C + c0,
            ),
        )

    return tile.reshape((channels, rows))


@nki.jit
def _batch_normalization(
    x, gamma, beta, N, C, eps, statistics_rows, apply_groups
):
    assert nl.num_programs() == 2
    assert C % 256 == 0

    pid = nl.program_id(0)
    P = 128
    S = 128
    F = C // 2
    cbase = pid * F
    channel_blocks = F // S
    num_statistics_blocks = (N + statistics_rows - 1) // statistics_rows
    apply_rows = P * apply_groups
    num_apply_blocks = (N + apply_rows - 1) // apply_rows

    assert F <= 512

    out = nl.ndarray((N, C), dtype=x.dtype, buffer=nl.shared_hbm)
    partials = nl.ndarray(
        (2, num_statistics_blocks, C),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    moments = nl.ndarray(
        (2, C), dtype=nl.float32, buffer=nl.shared_hbm
    )

    # First traversal: channels are partitions and rows are the free axis.
    # VectorE and ScalarE promote the loaded values to fp32 internally;
    # both reductions and their destinations are fp32.
    for cb in range(channel_blocks):
        c0 = cbase + cb * S
        for b in range(num_statistics_blocks):
            r0 = b * statistics_rows
            rows = min(statistics_rows, N - r0)
            loaded = _load_statistics_tile(x, r0, rows, c0, S, C)

            block_sum = nl.ndarray(
                (S, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            block_sumsq = nl.ndarray(
                (S, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            squares = nl.ndarray(
                (S, rows), dtype=nl.float32, buffer=nl.sbuf
            )

            nisa.tensor_reduce(
                dst=block_sum, op=nl.add, data=loaded, axis=1
            )
            nisa.activation(
                dst=squares,
                op=nl.square,
                data=loaded,
                reduce_op=nl.add,
                reduce_res=block_sumsq,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                engine=nisa.engine.scalar,
            )

            nisa.dma_copy(
                dst=partials.ap(
                    pattern=[[1, S], [C, 1]],
                    offset=b * C + c0,
                ),
                src=block_sum,
            )
            nisa.dma_copy(
                dst=partials.ap(
                    pattern=[[1, S], [C, 1]],
                    offset=(num_statistics_blocks + b) * C + c0,
                ),
                src=block_sumsq,
            )

    # No combine or second traversal starts before all partials are written.
    nisa.core_barrier(data=partials, cores=(0, 1))

    scale_vector = nl.ndarray(
        (1, F), dtype=nl.float32, buffer=nl.sbuf
    )
    shift_vector = nl.ndarray(
        (1, F), dtype=nl.float32, buffer=nl.sbuf
    )

    # Fixed-order second-level reduction, followed by the affine parameters.
    for cb in range(channel_blocks):
        c0 = cbase + cb * S

        sums = nl.ndarray(
            (S, num_statistics_blocks),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        sumsqs = nl.ndarray(
            (S, num_statistics_blocks),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(
            dst=sums,
            src=partials.ap(
                pattern=[[1, S], [C, num_statistics_blocks]],
                offset=c0,
            ),
        )
        nisa.dma_copy(
            dst=sumsqs,
            src=partials.ap(
                pattern=[[1, S], [C, num_statistics_blocks]],
                offset=num_statistics_blocks * C + c0,
            ),
        )

        total = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        totalsq = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=total, op=nl.add, data=sums, axis=1)
        nisa.tensor_reduce(dst=totalsq, op=nl.add, data=sumsqs, axis=1)

        mean = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        variance = nl.ndarray(
            (S, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        mean_square = nl.ndarray(
            (S, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        inv_std = nl.ndarray(
            (S, 1), dtype=nl.float32, buffer=nl.sbuf
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
            engine=nisa.engine.vector,
        )
        nisa.tensor_scalar(
            dst=variance, data=variance, op0=nl.add, operand0=eps
        )
        nisa.activation(dst=inv_std, op=nl.rsqrt, data=variance)

        nisa.dma_copy(
            dst=moments.ap(
                pattern=[[1, S], [C, 1]], offset=c0
            ),
            src=mean,
        )
        nisa.dma_copy(
            dst=moments.ap(
                pattern=[[1, S], [C, 1]], offset=C + c0
            ),
            src=inv_std,
        )

        gamma_t = nl.ndarray(
            (S, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        beta_t = nl.ndarray(
            (S, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=gamma_t,
            src=gamma.reshape((C, 1))[c0:c0 + S, :],
        )
        nisa.dma_copy(
            dst=beta_t,
            src=beta.reshape((C, 1))[c0:c0 + S, :],
        )

        scale = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        shift = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        mean_scale = nl.ndarray(
            (S, 1), dtype=nl.float32, buffer=nl.sbuf
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

        # Only the small affine vectors need an on-chip transpose.
        scale_psum = nl.ndarray(
            (1, S), dtype=nl.float32, buffer=nl.psum
        )
        shift_psum = nl.ndarray(
            (1, S), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_transpose(
            dst=scale_psum, data=scale, engine=nisa.engine.tensor
        )
        nisa.nc_transpose(
            dst=shift_psum, data=shift, engine=nisa.engine.tensor
        )
        nisa.tensor_copy(
            dst=scale_vector[:, cb * S:(cb + 1) * S],
            src=scale_psum,
        )
        nisa.tensor_copy(
            dst=shift_vector[:, cb * S:(cb + 1) * S],
            src=shift_psum,
        )

    scale_rows = nl.broadcast_to(scale_vector, (P, F))
    shift_rows = nl.broadcast_to(shift_vector, (P, F))
    scale_tile = nl.ndarray(
        (P, apply_groups, F), dtype=nl.float32, buffer=nl.sbuf
    )
    shift_tile = nl.ndarray(
        (P, apply_groups, F), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=scale_tile[:, 0, :], src=scale_rows)
    nisa.tensor_copy(dst=shift_tile[:, 0, :], src=shift_rows)

    filled = 1
    while filled < apply_groups:
        nisa.tensor_copy(
            dst=scale_tile[:, filled:2 * filled, :],
            src=scale_tile[:, 0:filled, :],
        )
        nisa.tensor_copy(
            dst=shift_tile[:, filled:2 * filled, :],
            src=shift_tile[:, 0:filled, :],
        )
        filled = filled * 2

    # Second traversal: row-major DMA and fp32 elementwise arithmetic.
    for b in range(num_apply_blocks):
        r0 = b * apply_rows
        rows = min(apply_rows, N - r0)
        p = rows // apply_groups

        if p > 0:
            yt = nl.ndarray(
                (p, apply_groups, F),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=yt,
                src=x.ap(
                    pattern=[
                        [apply_groups * C, p],
                        [C, apply_groups],
                        [1, F],
                    ],
                    offset=r0 * C + cbase,
                ),
            )
            nisa.tensor_tensor(
                dst=yt,
                data1=yt,
                data2=scale_tile[0:p, :, :],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=yt,
                data1=yt,
                data2=shift_tile[0:p, :, :],
                op=nl.add,
            )
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[
                        [apply_groups * C, p],
                        [C, apply_groups],
                        [1, F],
                    ],
                    offset=r0 * C + cbase,
                ),
                src=yt,
            )

        tail_rows = rows - p * apply_groups
        if tail_rows > 0:
            tail_r0 = r0 + p * apply_groups
            tail = nl.ndarray(
                (1, tail_rows, F),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=tail,
                src=x.ap(
                    pattern=[[tail_rows * C, 1], [C, tail_rows], [1, F]],
                    offset=tail_r0 * C + cbase,
                ),
            )
            nisa.tensor_tensor(
                dst=tail,
                data1=tail,
                data2=scale_tile[0:1, 0:tail_rows, :],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=tail,
                data1=tail,
                data2=shift_tile[0:1, 0:tail_rows, :],
                op=nl.add,
            )
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[tail_rows * C, 1], [C, tail_rows], [1, F]],
                    offset=tail_r0 * C + cbase,
                ),
                src=tail,
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
    apply_groups = 16
    apply_rows = 128 * apply_groups

    _last_config = {
        "lnc": 2,
        "statistics_channel_tile": 128,
        "statistics_row_block": statistics_rows,
        "partial_blocks": (N + statistics_rows - 1) // statistics_rows,
        "statistics_dtype": "float32",
        "statistics_layout": "channels_partition_rows_free",
        "statistics_load": "hwdge_3d_aligned16_with_dma_tail",
        "square_reduction": "fused_scalar_fp32",
        "combine": "free_axis_fixed_order",
        "apply_channel_tile": C // 2,
        "apply_partition_rows": 128,
        "apply_rows_per_partition": apply_groups,
        "apply_row_block": apply_rows,
        "apply_blocks": (N + apply_rows - 1) // apply_rows,
        "apply_layout": "row_major_grouped",
        "affine_expansion": "doubling",
        "tail": "clipped",
        "launches": 1,
        "global_barriers": 1,
    }

    return _batch_normalization[2](
        input, gamma, beta, N, C, eps, statistics_rows, apply_groups
    )


def get_last_config() -> dict:
    return dict(_last_config)
