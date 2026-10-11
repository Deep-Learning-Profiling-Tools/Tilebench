import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _phi_fp32(x):
    negative = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    mapped = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)

    # phi(x) = max(x, 0) + exp(min(x, 0)).
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


@nki.jit
def _linear_attention(Q, K, V, eps, row_tile):
    M, D = Q.shape
    pid = nl.program_id(0)
    peer = 1 - pid
    rows_per_core = M // 2
    core_start = pid * rows_per_core
    tile_count = (rows_per_core + row_tile - 1) // row_tile
    d0 = min(D, 128)
    width = D + 1

    out = nl.ndarray(
        (M, D), dtype=nl.float32, buffer=nl.shared_hbm
    )

    # The final column is Z:
    # phi(K).T @ [V, 1] = [S, Z].
    # All operands and accumulations are fp32, including this column.
    state0_acc = nl.ndarray(
        (d0, width), dtype=nl.float32, buffer=nl.psum
    )
    if D > 128:
        state1_acc = nl.ndarray(
            (D - 128, width), dtype=nl.float32, buffer=nl.psum
        )

    for t in range(tile_count):
        local_start = t * row_tile
        rows = min(row_tile, rows_per_core - local_start)
        start = core_start + local_start

        k = nl.ndarray(
            (rows, D), dtype=nl.float32, buffer=nl.sbuf
        )
        augmented_v = nl.ndarray(
            (rows, width), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=k, src=K[start:start + rows, :]
        )
        nisa.dma_copy(
            dst=augmented_v[:, 0:D],
            src=V[start:start + rows, :],
        )
        nisa.memset(
            dst=augmented_v[:, D:width], value=1.0
        )

        phi_k = _phi_fp32(k)
        nisa.nc_matmul(
            dst=state0_acc,
            stationary=phi_k[:, 0:d0],
            moving=augmented_v,
            accumulate=(t > 0),
        )
        if D > 128:
            nisa.nc_matmul(
                dst=state1_acc,
                stationary=phi_k[:, 128:D],
                moving=augmented_v,
                accumulate=(t > 0),
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

    # Both S and Z are completely reduced before any output consumes them.
    for t in range(tile_count):
        local_start = t * row_tile
        rows = min(row_tile, rows_per_core - local_start)
        start = core_start + local_start

        q = nl.ndarray(
            (rows, D), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=q, src=Q[start:start + rows, :])
        phi_q = _phi_fp32(q)

        # Reduce numerator and denominator together:
        # phi(Q) @ [S, Z] = [numerator, denominator_without_eps].
        augmented_output = nl.ndarray(
            (rows, width), dtype=nl.float32, buffer=nl.psum
        )
        qt0 = _transpose_fp32(phi_q[:, 0:d0])
        nisa.nc_matmul(
            dst=augmented_output,
            stationary=qt0,
            moving=full_state0,
            accumulate=False,
        )
        if D > 128:
            qt1 = _transpose_fp32(phi_q[:, 128:D])
            nisa.nc_matmul(
                dst=augmented_output,
                stationary=qt1,
                moving=full_state1,
                accumulate=True,
            )

        denominator = nl.ndarray(
            (rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        inverse = nl.ndarray(
            (rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=denominator,
            data=augmented_output[:, D:width],
            op0=nl.add,
            operand0=eps,
        )
        nisa.reciprocal(dst=inverse, data=denominator)

        result = nl.ndarray(
            (rows, D), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=result,
            data=augmented_output[:, 0:D],
            op0=nl.multiply,
            operand0=inverse,
        )
        nisa.dma_copy(
            dst=out[start:start + rows, :], src=result
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
    _LAST_CONFIG = {
        "lnc": 2,
        "launches": 1,
        "sequence_tile": 128,
        "sequence_splits": 2,
        "rows_per_core": M // 2,
        "state_feature_tile": min(128, D),
        "augmented_state_width": D + 1,
        "feature_maps": "fused_fp32",
        "state_multiply": "fp32",
        "numerator_multiply": "fp32",
        "normalizer_precision": "fp32",
        "normalizer_fusion": "augmented_state_column",
        "state_combine": "two_core_fp32_add",
    }
    return _linear_attention[2](Q, K, V, float(eps), 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
