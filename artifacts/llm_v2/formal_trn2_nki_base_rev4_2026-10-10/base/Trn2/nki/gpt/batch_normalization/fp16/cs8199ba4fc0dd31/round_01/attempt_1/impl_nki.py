import torch
import torch_neuronx
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _batch_normalization(x, gamma, beta, N, C, eps, groups):
    pid = nl.program_id(0)
    nprog = nl.num_programs()
    assert nprog == 2

    P = 128
    F = C // nprog
    block_rows = P * groups
    num_blocks = (N + block_rows - 1) // block_rows
    c0 = pid * F

    assert F <= 512
    assert num_blocks <= 128
    assert N % groups == 0

    # Fresh output and the two levels of fp32 statistics scratch.
    out = nl.ndarray((N, C), dtype=x.dtype, buffer=nl.shared_hbm)
    partials = nl.ndarray(
        (2, num_blocks, C), dtype=nl.float32, buffer=nl.shared_hbm
    )
    moments = nl.ndarray((2, C), dtype=nl.float32, buffer=nl.shared_hbm)

    row_ones = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=row_ones, value=1.0)

    # First traversal: each core owns half the channels. Within each
    # contiguous row block, fold groups locally, then reduce partitions.
    for b in range(num_blocks):
        r0 = b * block_rows
        rows = min(block_rows, N - r0)
        p = rows // groups

        xt = nl.ndarray((p, groups, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=xt,
            src=x.ap(
                pattern=[[groups * C, p], [C, groups], [1, F]],
                offset=r0 * C + c0,
            ),
        )

        squares = nl.ndarray(
            (p, groups, F), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=squares, data1=xt, data2=xt, op=nl.multiply
        )

        local_sum = nl.ndarray((p, F), dtype=nl.float32, buffer=nl.sbuf)
        local_sumsq = nl.ndarray((p, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=local_sum,
            data1=xt[:, 0, :],
            data2=xt[:, 1, :],
            op=nl.add,
        )
        nisa.tensor_tensor(
            dst=local_sumsq,
            data1=squares[:, 0, :],
            data2=squares[:, 1, :],
            op=nl.add,
        )
        for g in range(2, groups):
            nisa.tensor_tensor(
                dst=local_sum,
                data1=local_sum,
                data2=xt[:, g, :],
                op=nl.add,
            )
            nisa.tensor_tensor(
                dst=local_sumsq,
                data1=local_sumsq,
                data2=squares[:, g, :],
                op=nl.add,
            )

        sum_psum = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.psum)
        sumsq_psum = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=sum_psum,
            stationary=row_ones[0:p, :],
            moving=local_sum,
            accumulate=False,
        )
        nisa.nc_matmul(
            dst=sumsq_psum,
            stationary=row_ones[0:p, :],
            moving=local_sumsq,
            accumulate=False,
        )

        block_sum = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
        block_sumsq = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=block_sum, src=sum_psum)
        nisa.tensor_copy(dst=block_sumsq, src=sumsq_psum)
        nisa.dma_copy(
            dst=partials[0, b:b + 1, c0:c0 + F], src=block_sum
        )
        nisa.dma_copy(
            dst=partials[1, b:b + 1, c0:c0 + F], src=block_sumsq
        )

    # No apply work can begin until both cores finish all partials.
    nisa.core_barrier(data=partials, cores=(0, 1))

    sums = nl.ndarray((num_blocks, F), dtype=nl.float32, buffer=nl.sbuf)
    sumsqs = nl.ndarray((num_blocks, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sums, src=partials[0, :, c0:c0 + F])
    nisa.dma_copy(dst=sumsqs, src=partials[1, :, c0:c0 + F])

    block_ones = nl.ndarray(
        (num_blocks, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(dst=block_ones, value=1.0)
    total_psum = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.psum)
    totalsq_psum = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=total_psum,
        stationary=block_ones,
        moving=sums,
        accumulate=False,
    )
    nisa.nc_matmul(
        dst=totalsq_psum,
        stationary=block_ones,
        moving=sumsqs,
        accumulate=False,
    )

    total = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    totalsq = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=total, src=total_psum)
    nisa.tensor_copy(dst=totalsq, src=totalsq_psum)

    mean = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    variance = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    mean_square = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    inv_std = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_scalar(
        dst=mean, data=total, op0=nl.multiply, operand0=1.0 / N
    )
    nisa.tensor_scalar(
        dst=variance, data=totalsq, op0=nl.multiply, operand0=1.0 / N
    )
    nisa.tensor_tensor(
        dst=mean_square, data1=mean, data2=mean, op=nl.multiply
    )
    nisa.tensor_tensor(
        dst=variance, data1=variance, data2=mean_square, op=nl.subtract
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

    nisa.dma_copy(dst=moments[0:1, c0:c0 + F], src=mean)
    nisa.dma_copy(dst=moments[1:2, c0:c0 + F], src=inv_std)

    gamma_t = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    beta_t = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gamma_t, src=gamma.reshape((1, C))[:, c0:c0 + F]
    )
    nisa.dma_copy(
        dst=beta_t, src=beta.reshape((1, C))[:, c0:c0 + F]
    )

    scale = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    shift = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    mean_scale = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scale, data1=inv_std, data2=gamma_t, op=nl.multiply
    )
    nisa.tensor_tensor(
        dst=mean_scale, data1=mean, data2=scale, op=nl.multiply
    )
    nisa.tensor_tensor(
        dst=shift, data1=beta_t, data2=mean_scale, op=nl.subtract
    )

    # Construct reusable affine tiles once, outside the second traversal.
    scale_rows = nl.broadcast_to(scale, (P, F))
    shift_rows = nl.broadcast_to(shift, (P, F))
    scale_tile = nl.ndarray(
        (P, groups, F), dtype=nl.float32, buffer=nl.sbuf
    )
    shift_tile = nl.ndarray(
        (P, groups, F), dtype=nl.float32, buffer=nl.sbuf
    )
    for g in range(groups):
        nisa.tensor_copy(dst=scale_tile[:, g, :], src=scale_rows)
        nisa.tensor_copy(dst=shift_tile[:, g, :], src=shift_rows)

    # Second traversal: fp32 affine normalization, cast only at the store.
    for b in range(num_blocks):
        r0 = b * block_rows
        rows = min(block_rows, N - r0)
        p = rows // groups
        yt = nl.ndarray((p, groups, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=yt,
            src=x.ap(
                pattern=[[groups * C, p], [C, groups], [1, F]],
                offset=r0 * C + c0,
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
                pattern=[[groups * C, p], [C, groups], [1, F]],
                offset=r0 * C + c0,
            ),
            src=yt,
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

    groups = 8
    block_rows = 128 * groups
    result = _batch_normalization[2](
        input, gamma, beta, N, C, eps, groups
    )
    _last_config = {
        "lnc": 2,
        "channel_tile": C // 2,
        "partition_rows": 128,
        "rows_per_partition": groups,
        "row_block": block_rows,
        "partial_blocks": (N + block_rows - 1) // block_rows,
        "statistics_dtype": "float32",
        "reduction": "group_fold_then_tensor_sum",
        "launches": 1,
        "global_barriers": 1,
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
