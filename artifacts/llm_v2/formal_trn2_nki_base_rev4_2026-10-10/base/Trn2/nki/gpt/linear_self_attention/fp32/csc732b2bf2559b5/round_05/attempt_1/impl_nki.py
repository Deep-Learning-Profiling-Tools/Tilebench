import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _phi_fp32(x):
    negative = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=negative, data=x, op0=nl.minimum, operand0=0.0
    )
    nisa.activation(dst=negative, op=nl.exp, data=negative)
    nisa.tensor_scalar(
        dst=x, data=x, op0=nl.maximum, operand0=0.0
    )
    nisa.tensor_tensor(
        dst=x, data1=x, data2=negative, op=nl.add
    )
    return x


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
def _linear_attention(Q, K, V, eps, chunk, use_tf32):
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

    state_width = width
    operand_dtype = nl.float32
    if use_tf32:
        state_width = D
        operand_dtype = nl.tfloat32

    state0_acc = nl.ndarray(
        (d0, state_width), dtype=nl.float32, buffer=nl.psum
    )
    if D > 128:
        state1_acc = nl.ndarray(
            (D - 128, state_width),
            dtype=nl.float32,
            buffer=nl.psum,
        )

    if use_tf32:
        z0_acc = nl.ndarray(
            (d0, 1), dtype=nl.float32, buffer=nl.psum
        )
        if D > 128:
            z1_acc = nl.ndarray(
                (D - 128, 1), dtype=nl.float32, buffer=nl.psum
            )
        ones = nl.ndarray(
            (128, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=ones, value=1.0)

    for t in range(chunks):
        rows = min(chunk, core_rows - t * chunk)
        start = core_start + t * chunk
        batch = _batch_factor(rows)
        partitions = rows // batch

        k = nl.ndarray(
            (partitions, batch, D),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        v = nl.ndarray(
            (partitions, batch, state_width),
            dtype=operand_dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(
            dst=k,
            src=K[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )
        nisa.dma_copy(
            dst=v[:, :, 0:D],
            src=V[start:start + rows, :].reshape(
                (partitions, batch, D)
            ),
        )
        phi_k = _phi_fp32(k)

        if use_tf32:
            phi_k_multiply = nl.ndarray(
                (partitions, batch, D),
                dtype=nl.tfloat32,
                buffer=nl.sbuf,
            )
            nisa.tensor_copy(dst=phi_k_multiply, src=phi_k)
        else:
            phi_k_multiply = phi_k

        for b in range(batch):
            accumulate = (t > 0 or b > 0)
            if not use_tf32:
                nisa.memset(dst=v[:, b, D:width], value=1.0)

            nisa.nc_matmul(
                dst=state0_acc,
                stationary=phi_k_multiply[:, b, 0:d0],
                moving=v[:, b, :],
                accumulate=accumulate,
            )
            if D > 128:
                nisa.nc_matmul(
                    dst=state1_acc,
                    stationary=phi_k_multiply[:, b, 128:D],
                    moving=v[:, b, :],
                    accumulate=accumulate,
                )

            if use_tf32:
                # Z uses the original fp32 feature map, not its
                # TF32-rounded matrix-multiplication operand.
                nisa.nc_matmul(
                    dst=z0_acc,
                    stationary=phi_k[:, b, 0:d0],
                    moving=ones[0:partitions, :],
                    accumulate=accumulate,
                )
                if D > 128:
                    nisa.nc_matmul(
                        dst=z1_acc,
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
    if use_tf32:
        nisa.tensor_copy(
            dst=local_state0[:, 0:D], src=state0_acc
        )
        nisa.tensor_copy(
            dst=local_state0[:, D:width], src=z0_acc
        )
    else:
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
        if use_tf32:
            nisa.tensor_copy(
                dst=local_state1[:, 0:D], src=state1_acc
            )
            nisa.tensor_copy(
                dst=local_state1[:, D:width], src=z1_acc
            )
        else:
            nisa.tensor_copy(dst=local_state1, src=state1_acc)

        nisa.sendrecv_compute(
            src=local_state1,
            dst=full_state1,
            send_to_rank=peer,
            recv_from_rank=peer,
            reduce_op=nl.add,
        )

    if use_tf32:
        multiply_state0 = nl.ndarray(
            (d0, D), dtype=nl.tfloat32, buffer=nl.sbuf
        )
        nisa.tensor_copy(
            dst=multiply_state0, src=full_state0[:, 0:D]
        )
        if D > 128:
            multiply_state1 = nl.ndarray(
                (D - 128, D),
                dtype=nl.tfloat32,
                buffer=nl.sbuf,
            )
            nisa.tensor_copy(
                dst=multiply_state1, src=full_state1[:, 0:D]
            )
    else:
        multiply_state0 = full_state0
        if D > 128:
            multiply_state1 = full_state1

    # The exchanged state is now completely reduced over all M rows.
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
            output_acc = nl.ndarray(
                (partitions, state_width),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            if use_tf32:
                denominator_acc = nl.ndarray(
                    (partitions, 1),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

            qt0 = _transpose_fp32(phi_q[:, b, 0:d0])
            if use_tf32:
                qt0_multiply = nl.ndarray(
                    (d0, partitions),
                    dtype=nl.tfloat32,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=qt0_multiply, src=qt0)
            else:
                qt0_multiply = qt0

            nisa.nc_matmul(
                dst=output_acc,
                stationary=qt0_multiply,
                moving=multiply_state0,
                accumulate=False,
            )
            if use_tf32:
                nisa.nc_matmul(
                    dst=denominator_acc,
                    stationary=qt0,
                    moving=full_state0[:, D:width],
                    accumulate=False,
                )

            if D > 128:
                qt1 = _transpose_fp32(phi_q[:, b, 128:D])
                if use_tf32:
                    qt1_multiply = nl.ndarray(
                        (D - 128, partitions),
                        dtype=nl.tfloat32,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(dst=qt1_multiply, src=qt1)
                else:
                    qt1_multiply = qt1

                nisa.nc_matmul(
                    dst=output_acc,
                    stationary=qt1_multiply,
                    moving=multiply_state1,
                    accumulate=True,
                )
                if use_tf32:
                    nisa.nc_matmul(
                        dst=denominator_acc,
                        stationary=qt1,
                        moving=full_state1[:, D:width],
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
            if use_tf32:
                denominator_source = denominator_acc
            else:
                denominator_source = output_acc[:, D:width]

            nisa.tensor_scalar(
                dst=denominator,
                data=denominator_source,
                op0=nl.add,
                operand0=eps,
            )
            nisa.reciprocal(dst=inverse, data=denominator)
            nisa.tensor_scalar(
                dst=result_batch[:, b, :],
                data=output_acc[:, 0:D],
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
    use_tf32 = D >= 128

    _LAST_CONFIG = {
        "lnc": 2,
        "launches": 1,
        "sequence_splits": 2,
        "rows_per_core": M // 2,
        "sequence_chunk": chunk,
        "contraction_tile_max": 128,
        "batch_factor": "power_of_two_up_to_8",
        "state_feature_tile": min(128, D),
        "exchanged_state_width": D + 1,
        "feature_maps": "batched_fused_fp32",
        "state_multiply": "tf32" if use_tf32 else "fp32",
        "numerator_multiply": "tf32" if use_tf32 else "fp32",
        "normalizer_precision": "fp32",
        "normalizer_fusion": (
            "separate_fp32_dots"
            if use_tf32
            else "augmented_state_column"
        ),
        "state_combine": "two_core_fp32_add",
        "padding": False,
    }
    return _linear_attention[2](
        Q, K, V, float(eps), chunk, use_tf32
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
