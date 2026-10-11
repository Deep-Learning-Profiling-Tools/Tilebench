import torch
import torch_neuronx
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


def _fold_groups(data, partitions, groups, channels):
    width = groups
    current = data
    while width > 1:
        half = width // 2
        reduced = nl.ndarray(
            (partitions, half, channels),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_tensor(
            dst=reduced,
            data1=current[:, 0:width:2, :],
            data2=current[:, 1:width:2, :],
            op=nl.add,
        )
        current = reduced
        width = half
    return current.reshape((partitions, channels))


@nki.jit
def _batch_normalization(x, gamma, beta, N, C, eps, groups):
    pid = nl.program_id(0)
    assert nl.num_programs() == 2

    P = 128
    F = C // 2
    block_rows = P * groups
    num_blocks = (N + block_rows - 1) // block_rows
    c0 = pid * F

    assert F <= 512
    assert num_blocks <= 128
    assert N % groups == 0

    out = nl.ndarray((N, C), dtype=x.dtype, buffer=nl.shared_hbm)
    partials = nl.ndarray(
        (2, num_blocks, C), dtype=nl.float32, buffer=nl.shared_hbm
    )
    moments = nl.ndarray(
        (2, C), dtype=nl.float32, buffer=nl.shared_hbm
    )

    row_ones = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=row_ones, value=1.0)

    # First traversal. Independent ScalarE squaring and VectorE sum
    # folding can execute concurrently.
    for b in range(num_blocks):
        r0 = b * block_rows
        rows = min(block_rows, N - r0)
        p = rows // groups

        xt = nl.ndarray(
            (p, groups, F), dtype=nl.float32, buffer=nl.sbuf
        )
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
        nisa.activation(
            dst=squares,
            op=nl.square,
            data=xt,
            engine=nisa.engine.scalar,
        )

        local_sum = _fold_groups(xt, p, groups, F)
        local_sumsq = _fold_groups(squares, p, groups, F)

        sum_psum = nl.ndarray(
            (1, F), dtype=nl.float32, buffer=nl.psum
        )
        sumsq_psum = nl.ndarray(
            (1, F), dtype=nl.float32, buffer=nl.psum
        )
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

        block_sum = nl.ndarray(
            (1, F), dtype=nl.float32, buffer=nl.sbuf
        )
        block_sumsq = nl.ndarray(
            (1, F), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=block_sum, src=sum_psum)
        nisa.tensor_copy(dst=block_sumsq, src=sumsq_psum)
        nisa.dma_copy(
            dst=partials[0, b:b + 1, c0:c0 + F], src=block_sum
        )
        nisa.dma_copy(
            dst=partials[1, b:b + 1, c0:c0 + F], src=block_sumsq
        )

    # Both cores finish every partial before combine or apply.
    nisa.core_barrier(data=partials, cores=(0, 1))

    sums = nl.ndarray(
        (num_blocks, F), dtype=nl.float32, buffer=nl.sbuf
    )
    sumsqs = nl.ndarray(
        (num_blocks, F), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=sums, src=partials[0, :, c0:c0 + F])
    nisa.dma_copy(dst=sumsqs, src=partials[1, :, c0:c0 + F])

    block_ones = nl.ndarray(
        (num_blocks, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(dst=block_ones, value=1.0)

    total_psum = nl.ndarray(
        (1, F), dtype=nl.float32, buffer=nl.psum
    )
    totalsq_psum = nl.ndarray(
        (1, F), dtype=nl.float32, buffer=nl.psum
    )
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
    mean_square = nl.ndarray(
        (1, F), dtype=nl.float32, buffer=nl.sbuf
    )
    inv_std = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_scalar(
        dst=mean,
        data=total,
        op0=nl.multiply,
        operand0=1.0 / N,
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
    mean_scale = nl.ndarray(
        (1, F), dtype=nl.float32, buffer=nl.sbuf
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

    scale_rows = nl.broadcast_to(scale, (P, F))
    shift_rows = nl.broadcast_to(shift, (P, F))
    scale_tile = nl.ndarray(
        (P, groups, F), dtype=nl.float32, buffer=nl.sbuf
    )
    shift_tile = nl.ndarray(
        (P, groups, F), dtype=nl.float32, buffer=nl.sbuf
    )

    # Expand reusable affine tiles by doubling rather than issuing
    # one separate copy for every group.
    nisa.tensor_copy(dst=scale_tile[:, 0, :], src=scale_rows)
    nisa.tensor_copy(dst=shift_tile[:, 0, :], src=shift_rows)
    filled = 1
    while filled < groups:
        nisa.tensor_copy(
            dst=scale_tile[:, filled:2 * filled, :],
            src=scale_tile[:, 0:filled, :],
        )
        nisa.tensor_copy(
            dst=shift_tile[:, filled:2 * filled, :],
            src=shift_tile[:, 0:filled, :],
        )
        filled = filled * 2

    # Second traversal: all arithmetic remains fp32 until output DMA.
    for b in range(num_blocks):
        r0 = b * block_rows
        rows = min(block_rows, N - r0)
        p = rows // groups

        yt = nl.ndarray(
            (p, groups, F), dtype=nl.float32, buffer=nl.sbuf
        )
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

    groups = 16 if N % 16 == 0 else 8
    block_rows = 128 * groups
    _last_config = {
        "lnc": 2,
        "channel_tile": C // 2,
        "partition_rows": 128,
        "rows_per_partition": groups,
        "row_block": block_rows,
        "partial_blocks": (N + block_rows - 1) // block_rows,
        "statistics_dtype": "float32",
        "reduction": "binary_group_tree_then_tensor_sum",
        "square_engine": "scalar",
        "affine_expansion": "doubling",
        "launches": 1,
        "global_barriers": 1,
    }
    return _batch_normalization[2](
        input, gamma, beta, N, C, eps, groups
    )


def get_last_config() -> dict:
    return dict(_last_config)
