import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _rmsnorm_kernel(x, rms_w, eps, row_tile, stat_chunk):
    batch, M, K = x.shape
    R = batch * M
    P = row_tile

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x_rows = x.reshape((R, K))
    out_rows = out.reshape((R, K))

    pid = nl.program_id(0)
    programs = nl.num_programs()

    # Cache the upcast weight for all row tiles owned by this core.
    weight = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
    weight_src = rms_w.reshape((1, K))
    nisa.dma_copy(
        dst=weight,
        src=weight_src.ap(pattern=[[0, P], [1, K]], offset=0),
    )

    # Reuse a bounded working set; retain each entire row until its output
    # has been computed, so there is only one HBM read of x.
    values = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
    squared = nl.ndarray((P, stat_chunk), dtype=nl.float32, buffer=nl.sbuf)
    sums = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    partial = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray((P, K), dtype=x.dtype, buffer=nl.sbuf)

    tiles = (R + P - 1) // P
    for tile in range(pid, tiles, programs):
        r0 = tile * P
        rows = min(P, R - r0)

        nisa.dma_copy(
            dst=values[0:rows, :],
            src=x_rows[r0:r0 + rows, :],
        )

        for k0 in range(0, K, stat_chunk):
            width = min(stat_chunk, K - k0)
            if k0 == 0:
                nisa.activation(
                    dst=squared[0:rows, 0:width],
                    op=nl.square,
                    data=values[0:rows, k0:k0 + width],
                    reduce_op=nl.add,
                    reduce_res=sums[0:rows, :],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
            else:
                nisa.activation(
                    dst=squared[0:rows, 0:width],
                    op=nl.square,
                    data=values[0:rows, k0:k0 + width],
                    reduce_op=nl.add,
                    reduce_res=partial[0:rows, :],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                nisa.tensor_tensor(
                    dst=sums[0:rows, :],
                    data1=sums[0:rows, :],
                    data2=partial[0:rows, :],
                    op=nl.add,
                )

        nisa.activation(
            dst=rstd[0:rows, :],
            op=nl.rsqrt,
            data=sums[0:rows, :],
            scale=1.0 / K,
            bias=eps,
        )

        nisa.tensor_scalar(
            dst=values[0:rows, :],
            data=values[0:rows, :],
            op0=nl.multiply,
            operand0=rstd[0:rows, :],
            engine=nisa.engine.vector,
        )
        nisa.tensor_tensor(
            dst=result[0:rows, :],
            data1=values[0:rows, :],
            data2=weight[0:rows, :],
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=out_rows[r0:r0 + rows, :],
            src=result[0:rows, :],
        )

    return out


def run(
    x: torch.Tensor,
    rms_w: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    global _last_config
    batch, M, K = x.shape
    row_tile = 128
    stat_chunk = min(2048, K)

    _last_config = {
        "lnc": 2,
        "rows_per_tile": row_tile,
        "statistic_chunk": stat_chunk,
        "normalization_width": K,
        "row_tiles": (batch * M + row_tile - 1) // row_tile,
        "retained_rows": True,
        "compute_dtype": "float32",
        "fused": True,
    }
    return _rmsnorm_kernel[2](x, rms_w, eps, row_tile, stat_chunk)


def get_last_config() -> dict:
    return dict(_last_config)
