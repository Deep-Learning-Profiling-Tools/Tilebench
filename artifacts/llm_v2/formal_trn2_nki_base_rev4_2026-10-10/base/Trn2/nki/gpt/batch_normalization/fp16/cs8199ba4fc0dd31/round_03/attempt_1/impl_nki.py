import torch
import torch_neuronx
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


def _tail_groups(rows, maximum):
    if rows <= 128 * 8:
        return 8
    if rows <= 128 * 16:
        return min(16, maximum)
    return maximum


def _load_grouped(x, r0, rows, c0, channels, groups, C):
    full = rows // groups
    remainder = rows % groups
    partitions = full + int(remainder != 0)

    tile = nl.ndarray(
        (partitions, groups, channels), dtype=x.dtype, buffer=nl.sbuf
    )
    if full > 0:
        nisa.dma_copy(
            dst=tile[0:full, :, :],
            src=x.ap(
                pattern=[[groups * C, full], [C, groups], [1, channels]],
                offset=r0 * C + c0,
            ),
        )
    if remainder != 0:
        tail = nl.ndarray(
            (1, groups, channels), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.memset(dst=tail, value=0.0)
        nisa.dma_copy(
            dst=tail[:, 0:remainder, :],
            src=x.ap(
                pattern=[[groups * C, 1], [C, remainder], [1, channels]],
                offset=(r0 + full * groups) * C + c0,
            ),
        )
        nisa.dma_copy(dst=tile[full:full + 1, :, :], src=tail)
    return tile


def _store_grouped(out, tile, r0, rows, c0, channels, groups, C):
    full = rows // groups
    remainder = rows % groups
    if full > 0:
        nisa.dma_copy(
            dst=out.ap(
                pattern=[[groups * C, full], [C, groups], [1, channels]],
                offset=r0 * C + c0,
            ),
            src=tile[0:full, :, :],
        )
    if remainder != 0:
        nisa.dma_copy(
            dst=out.ap(
                pattern=[[groups * C, 1], [C, remainder], [1, channels]],
                offset=(r0 + full * groups) * C + c0,
            ),
            src=tile[full:full + 1, 0:remainder, :],
        )


def _fold_moments(tile, groups, channels):
    partitions = tile.shape[0]

    # Both instructions promote their fp16 operands before arithmetic.
    squares = nl.ndarray(
        (partitions, groups, channels), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.activation(
        dst=squares,
        op=nl.square,
        data=tile,
        engine=nisa.engine.scalar,
    )

    width = groups // 2
    current = nl.ndarray(
        (partitions, 2, width, channels),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_tensor(
        dst=current[:, 0, :, :],
        data1=tile[:, 0:groups:2, :],
        data2=tile[:, 1:groups:2, :],
        op=nl.add,
    )
    nisa.tensor_tensor(
        dst=current[:, 1, :, :],
        data1=squares[:, 0:groups:2, :],
        data2=squares[:, 1:groups:2, :],
        op=nl.add,
    )

    # Fold sums and squared sums together after their first tree level.
    while width > 1:
        half = width // 2
        reduced = nl.ndarray(
            (partitions, 2, half, channels),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_tensor(
            dst=reduced,
            data1=current[:, :, 0:width:2, :],
            data2=current[:, :, 1:width:2, :],
            op=nl.add,
        )
        current = reduced
        width = half

    return current.reshape((partitions, 2, channels))


@nki.jit
def _batch_normalization(
    x, gamma, beta, N, C, eps, statistics_groups, apply_groups
):
    pid = nl.program_id(0)
    assert nl.num_programs() == 2

    P = 128
    F = C // 2
    c0 = pid * F
    statistics_rows = P * statistics_groups
    num_blocks = (N + statistics_rows - 1) // statistics_rows
    apply_rows = P * apply_groups
    num_apply_blocks = (N + apply_rows - 1) // apply_rows

    assert F <= 512
    assert num_blocks <= 128

    out = nl.ndarray((N, C), dtype=x.dtype, buffer=nl.shared_hbm)
    partials = nl.ndarray(
        (2, num_blocks, C), dtype=nl.float32, buffer=nl.shared_hbm
    )
    moments = nl.ndarray(
        (2, C), dtype=nl.float32, buffer=nl.shared_hbm
    )

    row_ones = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=row_ones, value=1.0)

    # First logical traversal: deterministic per-block fp32 moments.
    for b in range(num_blocks):
        r0 = b * statistics_rows
        rows = min(statistics_rows, N - r0)
        groups = _tail_groups(rows, statistics_groups)
        xt = _load_grouped(x, r0, rows, c0, F, groups, C)
        p = xt.shape[0]
        local = _fold_moments(xt, groups, F)

        sum_psum = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.psum)
        sumsq_psum = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=sum_psum,
            stationary=row_ones[0:p, :],
            moving=local[:, 0, :],
            accumulate=False,
        )
        nisa.nc_matmul(
            dst=sumsq_psum,
            stationary=row_ones[0:p, :],
            moving=local[:, 1, :],
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

    scale_rows = nl.broadcast_to(scale, (P, F))
    shift_rows = nl.broadcast_to(shift, (P, F))
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
        amount = min(filled, apply_groups - filled)
        nisa.tensor_copy(
            dst=scale_tile[:, filled:filled + amount, :],
            src=scale_tile[:, 0:amount, :],
        )
        nisa.tensor_copy(
            dst=shift_tile[:, filled:filled + amount, :],
            src=shift_tile[:, 0:amount, :],
        )
        filled = filled + amount

    # Second logical traversal. Multiplication is retained in fp32;
    # the final fp32 addition casts directly into the output staging tile.
    for b in range(num_apply_blocks):
        r0 = b * apply_rows
        rows = min(apply_rows, N - r0)
        groups = _tail_groups(rows, apply_groups)
        yt = _load_grouped(x, r0, rows, c0, F, groups, C)
        p = yt.shape[0]

        product = nl.ndarray(
            (p, groups, F), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=product,
            data1=yt,
            data2=scale_tile[0:p, 0:groups, :],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=yt,
            data1=product,
            data2=shift_tile[0:p, 0:groups, :],
            op=nl.add,
        )
        _store_grouped(out, yt, r0, rows, c0, F, groups, C)

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

    if N <= 128 * 8:
        statistics_groups = 8
        apply_groups = 8
    elif N <= 128 * 16:
        statistics_groups = 16
        apply_groups = 16
    else:
        statistics_groups = 32
        apply_groups = 24

    statistics_rows = 128 * statistics_groups
    apply_rows = 128 * apply_groups
    _last_config = {
        "lnc": 2,
        "channel_tile": C // 2,
        "partition_rows": 128,
        "statistics_rows_per_partition": statistics_groups,
        "apply_rows_per_partition": apply_groups,
        "statistics_row_block": statistics_rows,
        "apply_row_block": apply_rows,
        "partial_blocks": (N + statistics_rows - 1) // statistics_rows,
        "apply_blocks": (N + apply_rows - 1) // apply_rows,
        "statistics_dtype": "float32",
        "reduction": "joint_moment_tree_then_tensor_sum",
        "square_engine": "scalar",
        "input_staging_dtype": "float16",
        "output_cast": "final_fp32_add",
        "affine_expansion": "doubling",
        "tail": "zero_padded_adaptive_groups",
        "launches": 1,
        "global_barriers": 1,
    }
    return _batch_normalization[2](
        input, gamma, beta, N, C, eps, statistics_groups, apply_groups
    )


def get_last_config() -> dict:
    return dict(_last_config)
