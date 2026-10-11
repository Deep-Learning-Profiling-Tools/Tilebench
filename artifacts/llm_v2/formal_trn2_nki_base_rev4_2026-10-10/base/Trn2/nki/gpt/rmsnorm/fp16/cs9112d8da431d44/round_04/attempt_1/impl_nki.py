import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _rmsnorm_kernel(x, rms_w, eps, row_tile, chunk_width):
    batch, M, K = x.shape
    R = batch * M
    P = row_tile
    C = chunk_width
    chunks = (K + C - 1) // C

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x_rows = x.reshape((R, K))
    out_rows = out.reshape((R, K))

    pid = nl.program_id(0)
    programs = nl.num_programs()

    weight = nl.ndarray((P, K), dtype=rms_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=weight,
        src=rms_w.reshape((1, K)).ap(
            pattern=[[0, P], [1, K]], offset=0
        ),
    )

    tiles = (R + P - 1) // P
    for tile in range(pid, tiles, programs):
        r0 = tile * P
        rows = min(P, R - r0)

        # Keep the input on chip; the fp32 workspace is only one chunk wide.
        values = nl.ndarray((P, K), dtype=x.dtype, buffer=nl.sbuf)
        work = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        sums = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)

        if chunks == 1:
            partials = sums
        else:
            partials = nl.ndarray(
                (P, chunks), dtype=nl.float32, buffer=nl.sbuf
            )

        # Chunked loads allow DMA to overlap the preceding chunk's statistic.
        for chunk in range(chunks):
            k0 = chunk * C
            width = min(C, K - k0)

            nisa.dma_copy(
                dst=values[0:rows, k0:k0 + width],
                src=x_rows[r0:r0 + rows, k0:k0 + width],
            )
            nisa.activation(
                dst=work[0:rows, 0:width],
                op=nl.square,
                data=values[0:rows, k0:k0 + width],
                reduce_op=nl.add,
                reduce_res=partials[0:rows, chunk:chunk + 1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

        if chunks > 1:
            nisa.tensor_reduce(
                dst=sums[0:rows, :],
                op=nl.add,
                data=partials[0:rows, :],
                axis=1,
            )

        nisa.activation(
            dst=rstd[0:rows, :],
            op=nl.rsqrt,
            data=sums[0:rows, :],
            scale=1.0 / K,
            bias=eps,
        )

        # All arithmetic is fp32 internally. Only the final multiplication
        # downcasts, and chunked stores overlap subsequent vector work.
        for chunk in range(chunks):
            k0 = chunk * C
            width = min(C, K - k0)

            nisa.tensor_scalar(
                dst=work[0:rows, 0:width],
                data=values[0:rows, k0:k0 + width],
                op0=nl.multiply,
                operand0=rstd[0:rows, :],
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=values[0:rows, k0:k0 + width],
                data1=work[0:rows, 0:width],
                data2=weight[0:rows, k0:k0 + width],
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(
                dst=out_rows[r0:r0 + rows, k0:k0 + width],
                src=values[0:rows, k0:k0 + width],
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
    chunk_width = min(K, 2048)

    _last_config = {
        "lnc": 2,
        "rows_per_tile": row_tile,
        "row_tiles": (batch * M + row_tile - 1) // row_tile,
        "statistic_chunk": chunk_width,
        "normalization_width": chunk_width,
        "chunks_per_row": (K + chunk_width - 1) // chunk_width,
        "workspace_width": chunk_width,
        "retained_rows": True,
        "input_storage": "float16",
        "weight_storage": "float16",
        "compute_dtype": "float32",
        "fused": True,
    }
    return _rmsnorm_kernel[2](x, rms_w, eps, row_tile, chunk_width)


def get_last_config() -> dict:
    return dict(_last_config)
