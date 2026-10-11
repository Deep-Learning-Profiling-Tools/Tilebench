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

    programs = nl.num_programs()
    pid = nl.program_id(0)
    assert R % (P * programs) == 0
    tiles_per_program = R // (P * programs)

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x_rows = x.reshape((R, K))
    out_rows = out.reshape((R, K))

    weight = nl.ndarray((P, K), dtype=rms_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=weight,
        src=rms_w.reshape((1, K)).ap(
            pattern=[[0, P], [1, K]], offset=0
        ),
    )

    # Ping-pong row storage. Arithmetic upcasts fp16 operands internally.
    values = nl.ndarray((P, 2, K), dtype=x.dtype, buffer=nl.sbuf)

    # Separate free-axis regions keep scalar statistics and vector
    # normalization independent while they execute concurrently.
    workspace = nl.ndarray((P, 2, C), dtype=nl.float32, buffer=nl.sbuf)
    sums = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
    rstd = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)

    if chunks == 1:
        partials = sums.reshape((P, 2, 1))
    else:
        partials = nl.ndarray(
            (P, 2, chunks), dtype=nl.float32, buffer=nl.sbuf
        )

    # Each iteration gathers statistics for the current row tile while
    # normalizing the preceding tile. The last iteration drains the pipeline.
    for tile in range(tiles_per_program + 1):
        current_slot = tile % 2
        previous_slot = (tile - 1) % 2

        for chunk in range(chunks):
            k0 = chunk * C
            width = min(C, K - k0)

            if tile < tiles_per_program:
                r0 = (tile * programs + pid) * P
                nisa.dma_copy(
                    dst=values[:, current_slot, k0:k0 + width],
                    src=x_rows[r0:r0 + P, k0:k0 + width],
                )
                nisa.activation(
                    dst=workspace[:, 0, 0:width],
                    op=nl.square,
                    data=values[:, current_slot, k0:k0 + width],
                    reduce_op=nl.add,
                    reduce_res=partials[
                        :, current_slot, chunk:chunk + 1
                    ],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                    engine=nisa.engine.scalar,
                )

            if tile > 0:
                r0 = ((tile - 1) * programs + pid) * P
                nisa.tensor_scalar(
                    dst=workspace[:, 1, 0:width],
                    data=values[:, previous_slot, k0:k0 + width],
                    op0=nl.multiply,
                    operand0=rstd[:, previous_slot:previous_slot + 1],
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=values[:, previous_slot, k0:k0 + width],
                    data1=workspace[:, 1, 0:width],
                    data2=weight[:, k0:k0 + width],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
                nisa.dma_copy(
                    dst=out_rows[r0:r0 + P, k0:k0 + width],
                    src=values[:, previous_slot, k0:k0 + width],
                )

        if tile < tiles_per_program:
            if chunks > 1:
                # Keep the partial-sum reduction on ScalarE so it does not
                # block the preceding tile's VectorE normalization.
                nisa.activation(
                    dst=workspace[:, 0, 0:chunks],
                    op=nl.copy,
                    data=partials[:, current_slot, :],
                    reduce_op=nl.add,
                    reduce_res=sums[:, current_slot:current_slot + 1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                    engine=nisa.engine.scalar,
                )

            nisa.activation(
                dst=rstd[:, current_slot:current_slot + 1],
                op=nl.rsqrt,
                data=sums[:, current_slot:current_slot + 1],
                scale=1.0 / K,
                bias=eps,
                engine=nisa.engine.scalar,
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
        "row_tiles": batch * M // row_tile,
        "row_buffers": 2,
        "pipeline_lag": 1,
        "statistic_chunk": chunk_width,
        "normalization_width": chunk_width,
        "chunks_per_row": (K + chunk_width - 1) // chunk_width,
        "workspace_width": chunk_width,
        "workspace_regions": 2,
        "statistics_engine": "scalar",
        "normalization_engine": "vector",
        "retained_rows": True,
        "input_storage": "float16",
        "weight_storage": "float16",
        "compute_dtype": "float32",
        "fused": True,
    }
    return _rmsnorm_kernel[2](x, rms_w, eps, row_tile, chunk_width)


def get_last_config() -> dict:
    return dict(_last_config)
