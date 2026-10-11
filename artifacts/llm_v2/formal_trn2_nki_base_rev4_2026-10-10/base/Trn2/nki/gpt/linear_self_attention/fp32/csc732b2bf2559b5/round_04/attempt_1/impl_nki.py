import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _phi_fp32(x):
    negative = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    mapped = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)

    # phi(x) = exp(min(x, 0)) + max(x, 0).
    nisa.tensor_scalar(
        dst=negative, data=x, op0=nl.minimum, operand0=0.0
    )
    nisa.activation(dst=mapped, op=nl.exp, data=negative)
    nisa.tensor_scalar(
        dst=x, data=x, op0=nl.maximum, operand0=0.0
    )
    nisa.tensor_tensor(
        dst=mapped, data1=mapped, data2=x, op=nl.add
    )
    return mapped


def _transpose_fp32(x):
    transposed = nl.ndarray(
        (x.shape[1], x.shape[0]),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    nisa.nc_transpose(
        dst=transposed, data=x, engine=nisa.engine.tensor
    )
    result = nl.ndarray(
        (x.shape[1], x.shape[0]),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=result, src=transposed)
    return result


def _batch_factor(rows):
    if rows > 512:
        return 8
    elif rows > 256:
        return 4
    elif rows > 128:
        return 2
    else:
        return 1


@nki.jit
def _linear_attention(Q, K, V, eps, chunk):
    M, D = Q.shape
    pid = nl.program_id(0)
    peer = 1 - pid
    core_rows = M // 2
    core_start = pid * core_rows
    chunks = (core_rows + chunk - 1) // chunk
    d0 = min(D, 128)
    width = D + 1

    out = nl.ndarray(
        (M, D), dtype=nl.float32, buffer=nl.shared_hbm
    )

    # Augment the state with Z. All multiplies, including the final
    # column, use unrounded fp32 operands and fp32 accumulation.
    state0_acc = nl.ndarray(
        (d0, width), dtype=nl.float32, buffer=nl.psum
    )
    if D > 128:
        state1_acc = nl.ndarray(
            (D - 128, width), dtype=nl.float32, buffer=nl.psum
        )

    for t in range(chunks):
        rows = min(chunk, core_rows - t * chunk)
        start = core_start + t * chunk
        batch = _batch_factor(rows)
        partitions = rows // batch

        # Pack consecutive sequence rows into the free dimension.
        # This batches loads and feature maps without adding padding.
        k = nl.ndarray(
            (partitions, batch, D),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        augmented_v = nl.ndarray(
            (partitions, batch, width),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

        nisa.dma_copy(
            dst=k,
            src=K[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )
        nisa.dma_copy(
            dst=augmented_v[:, :, 0:D],
            src=V[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )
        phi_k = _phi_fp32(k)

        for b in range(batch):
            nisa.memset(
                dst=augmented_v[:, b, D:width], value=1.0
            )
            accumulate = (t > 0 or b > 0)

            nisa.nc_matmul(
                dst=state0_acc,
                stationary=phi_k[:, b, 0:d0],
                moving=augmented_v[:, b, :],
                accumulate=accumulate,
            )
            if D > 128:
                nisa.nc_matmul(
                    dst=state1_acc,
                    stationary=phi_k[:, b, 128:D],
                    moving=augmented_v[:, b, :],
                    accumulate=accumulate,
                )

    local_state0 = nl.ndarray(
        (d0, width), dtype=nl.float32, buffer=nl.sbuf
    )
    full_state0 = nl.ndarray(
        (d0, width), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=local_state0, src=state0_acc)
    nisa.sendrecv_compute(
        src=local_state0,
        dst=full_state0,
        send_to_rank=peer,
        recv_from_rank=peer,
        reduce_op=nl.add,
    )

    if D > 128:
        local_state1 = nl.ndarray(
            (D - 128, width), dtype=nl.float32, buffer=nl.sbuf
        )
        full_state1 = nl.ndarray(
            (D - 128, width), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=local_state1, src=state1_acc)
        nisa.sendrecv_compute(
            src=local_state1,
            dst=full_state1,
            send_to_rank=peer,
            recv_from_rank=peer,
            reduce_op=nl.add,
        )

    # Both cores now hold S and Z reduced over the entire sequence.
    for t in range(chunks):
        rows = min(chunk, core_rows - t * chunk)
        start = core_start + t * chunk
        batch = _batch_factor(rows)
        partitions = rows // batch

        q = nl.ndarray(
            (partitions, batch, D),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(
            dst=q,
            src=Q[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )
        phi_q = _phi_fp32(q)
        result_batch = nl.ndarray(
            (partitions, batch, D),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

        for b in range(batch):
            augmented_output = nl.ndarray(
                (partitions, width),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            qt0 = _transpose_fp32(phi_q[:, b, 0:d0])
            nisa.nc_matmul(
                dst=augmented_output,
                stationary=qt0,
                moving=full_state0,
                accumulate=False,
            )
            if D > 128:
                qt1 = _transpose_fp32(phi_q[:, b, 128:D])
                nisa.nc_matmul(
                    dst=augmented_output,
                    stationary=qt1,
                    moving=full_state1,
                    accumulate=True,
                )

            denominator = nl.ndarray(
                (partitions, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            inverse = nl.ndarray(
                (partitions, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.tensor_scalar(
                dst=denominator,
                data=augmented_output[:, D:width],
                op0=nl.add,
                operand0=eps,
            )
            nisa.reciprocal(dst=inverse, data=denominator)
            nisa.tensor_scalar(
                dst=result_batch[:, b, :],
                data=augmented_output[:, 0:D],
                op0=nl.multiply,
                operand0=inverse,
            )

        nisa.dma_copy(
            dst=out[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
            src=result_batch,
        )

    return out


def run(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
):
    global _LAST_CONFIG
    M, D = Q.shape
    chunk = 1024
    _LAST_CONFIG = {
        "lnc": 2,
        "launches": 1,
        "sequence_splits": 2,
        "rows_per_core": M // 2,
        "sequence_chunk": chunk,
        "contraction_tile_max": 128,
        "batch_factor": "power_of_two_up_to_8",
        "state_feature_tile": min(128, D),
        "augmented_state_width": D + 1,
        "feature_maps": "batched_fused_fp32",
        "state_multiply": "fp32",
        "numerator_multiply": "fp32",
        "normalizer_precision": "fp32",
        "normalizer_fusion": "augmented_state_column",
        "state_combine": "two_core_fp32_add",
        "padding": False,
    }
    return _linear_attention[2](Q, K, V, float(eps), chunk)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
