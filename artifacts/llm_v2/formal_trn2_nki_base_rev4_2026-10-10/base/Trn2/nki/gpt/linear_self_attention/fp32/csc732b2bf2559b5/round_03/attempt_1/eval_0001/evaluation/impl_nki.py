import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _phi_fp32(x):
    negative = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    mapped = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
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
    tp = nl.ndarray(
        (x.shape[1], x.shape[0]), dtype=nl.float32, buffer=nl.psum
    )
    nisa.nc_transpose(dst=tp, data=x, engine=nisa.engine.tensor)
    result = nl.ndarray(
        (x.shape[1], x.shape[0]), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=result, src=tp)
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

    out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.shared_hbm)

    # The final column holds Z, but its matmuls use unrounded fp32
    # operands independently of the TF32 matmuls forming S.
    state0_acc = nl.ndarray(
        (d0, width), dtype=nl.float32, buffer=nl.psum
    )
    if D > 128:
        state1_acc = nl.ndarray(
            (D - 128, width), dtype=nl.float32, buffer=nl.psum
        )

    ones = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    for t in range(chunks):
        rows = min(chunk, core_rows - t * chunk)
        start = core_start + t * chunk
        batch = _batch_factor(rows)
        partitions = rows // batch

        # Consecutive rows are packed into the free dimension. This
        # batches DMA and feature-map instructions without padding.
        k = nl.ndarray(
            (partitions, batch, D), dtype=nl.float32, buffer=nl.sbuf
        )
        v = nl.ndarray(
            (partitions, batch, D), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=k,
            src=K[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )
        nisa.dma_copy(
            dst=v,
            src=V[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )

        phi_k = _phi_fp32(k)
        kt = nl.ndarray(
            (partitions, batch, D), dtype=nl.tfloat32, buffer=nl.sbuf
        )
        vt = nl.ndarray(
            (partitions, batch, D), dtype=nl.tfloat32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=kt, src=phi_k)
        nisa.tensor_copy(dst=vt, src=v)

        for b in range(batch):
            accumulate = (t > 0 or b > 0)
            nisa.nc_matmul(
                dst=state0_acc[:, 0:D],
                stationary=kt[:, b, 0:d0],
                moving=vt[:, b, :],
                accumulate=accumulate,
            )
            nisa.nc_matmul(
                dst=state0_acc[:, D:width],
                stationary=phi_k[:, b, 0:d0],
                moving=ones[0:partitions, :],
                accumulate=accumulate,
            )
            if D > 128:
                nisa.nc_matmul(
                    dst=state1_acc[:, 0:D],
                    stationary=kt[:, b, 128:D],
                    moving=vt[:, b, :],
                    accumulate=accumulate,
                )
                nisa.nc_matmul(
                    dst=state1_acc[:, D:width],
                    stationary=phi_k[:, b, 128:D],
                    moving=ones[0:partitions, :],
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
    s0_tf = nl.ndarray(
        (d0, D), dtype=nl.tfloat32, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=s0_tf, src=full_state0[:, 0:D])

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
        s1_tf = nl.ndarray(
            (D - 128, D), dtype=nl.tfloat32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=s1_tf, src=full_state1[:, 0:D])

    # S and Z are now fully reduced over the complete sequence.
    for t in range(chunks):
        rows = min(chunk, core_rows - t * chunk)
        start = core_start + t * chunk
        batch = _batch_factor(rows)
        partitions = rows // batch

        q = nl.ndarray(
            (partitions, batch, D), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=q,
            src=Q[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )
        phi_q = _phi_fp32(q)
        result_batch = nl.ndarray(
            (partitions, batch, D), dtype=nl.float32, buffer=nl.sbuf
        )

        for b in range(batch):
            augmented_output = nl.ndarray(
                (partitions, width), dtype=nl.float32, buffer=nl.psum
            )

            qt0 = _transpose_fp32(phi_q[:, b, 0:d0])
            qt0_tf = nl.ndarray(
                (d0, partitions), dtype=nl.tfloat32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=qt0_tf, src=qt0)

            nisa.nc_matmul(
                dst=augmented_output[:, 0:D],
                stationary=qt0_tf,
                moving=s0_tf,
                accumulate=False,
            )
            nisa.nc_matmul(
                dst=augmented_output[:, D:width],
                stationary=qt0,
                moving=full_state0[:, D:width],
                accumulate=False,
            )

            if D > 128:
                qt1 = _transpose_fp32(phi_q[:, b, 128:D])
                qt1_tf = nl.ndarray(
                    (D - 128, partitions),
                    dtype=nl.tfloat32,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=qt1_tf, src=qt1)
                nisa.nc_matmul(
                    dst=augmented_output[:, 0:D],
                    stationary=qt1_tf,
                    moving=s1_tf,
                    accumulate=True,
                )
                nisa.nc_matmul(
                    dst=augmented_output[:, D:width],
                    stationary=qt1,
                    moving=full_state1[:, D:width],
                    accumulate=True,
                )

            denominator = nl.ndarray(
                (partitions, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            inverse = nl.ndarray(
                (partitions, 1), dtype=nl.float32, buffer=nl.sbuf
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
        "state_multiply": "tf32",
        "numerator_multiply": "tf32",
        "normalizer_precision": "fp32",
        "normalizer_fusion": "separate_fp32_augmented_column",
        "state_combine": "two_core_fp32_add",
        "padding": False,
    }
    return _linear_attention[2](Q, K, V, float(eps), chunk)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
