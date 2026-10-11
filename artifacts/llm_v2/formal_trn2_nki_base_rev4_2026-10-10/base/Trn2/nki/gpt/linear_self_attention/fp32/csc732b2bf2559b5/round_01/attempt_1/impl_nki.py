import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _phi_fp32(x):
    negative = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    mapped = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)

    # max(x, 0) + exp(min(x, 0)) is the prescribed feature map.
    nisa.tensor_scalar(
        dst=negative, data=x, op0=nl.minimum, operand0=0.0
    )
    nisa.activation(dst=mapped, op=nl.exp, data=negative)
    nisa.tensor_scalar(dst=x, data=x, op0=nl.maximum, operand0=0.0)
    nisa.tensor_tensor(dst=mapped, data1=mapped, data2=x, op=nl.add)
    return mapped


def _transpose_tf32(x):
    transposed = nl.ndarray(
        (x.shape[1], x.shape[0]), dtype=nl.float32, buffer=nl.psum
    )
    nisa.nc_transpose(
        dst=transposed, data=x, engine=nisa.engine.tensor
    )
    result = nl.ndarray(
        (x.shape[1], x.shape[0]), dtype=nl.tfloat32, buffer=nl.sbuf
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

    out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.shared_hbm)

    state0_acc = nl.ndarray((d0, D), dtype=nl.float32, buffer=nl.psum)
    if D > 128:
        state1_acc = nl.ndarray(
            (D - 128, D), dtype=nl.float32, buffer=nl.psum
        )
    z_acc = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.psum)

    ones = nl.ndarray((row_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    # Each core reduces a disjoint half of the sequence. Exact-size
    # remainder tiles keep feature-mapped padding out of both reductions.
    for t in range(tile_count):
        local_start = t * row_tile
        rows = min(row_tile, rows_per_core - local_start)
        start = core_start + local_start

        k = nl.ndarray((rows, D), dtype=nl.float32, buffer=nl.sbuf)
        v = nl.ndarray((rows, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k, src=K[start:start + rows, :])
        nisa.dma_copy(dst=v, src=V[start:start + rows, :])

        phi_k = _phi_fp32(k)

        # Z uses the unrounded fp32 feature map.
        nisa.nc_matmul(
            dst=z_acc,
            stationary=ones[0:rows, :],
            moving=phi_k,
            accumulate=(t > 0),
        )

        k_tf = nl.ndarray((rows, D), dtype=nl.tfloat32, buffer=nl.sbuf)
        v_tf = nl.ndarray((rows, D), dtype=nl.tfloat32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_tf, src=phi_k)
        nisa.tensor_copy(dst=v_tf, src=v)

        nisa.nc_matmul(
            dst=state0_acc,
            stationary=k_tf[:, 0:d0],
            moving=v_tf,
            accumulate=(t > 0),
        )
        if D > 128:
            nisa.nc_matmul(
                dst=state1_acc,
                stationary=k_tf[:, 128:D],
                moving=v_tf,
                accumulate=(t > 0),
            )

    local_state0 = nl.ndarray((d0, D), dtype=nl.float32, buffer=nl.sbuf)
    full_state0 = nl.ndarray((d0, D), dtype=nl.float32, buffer=nl.sbuf)
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
            (D - 128, D), dtype=nl.float32, buffer=nl.sbuf
        )
        full_state1 = nl.ndarray(
            (D - 128, D), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=local_state1, src=state1_acc)
        nisa.sendrecv_compute(
            src=local_state1,
            dst=full_state1,
            send_to_rank=peer,
            recv_from_rank=peer,
            reduce_op=nl.add,
        )

    local_z = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
    full_z = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=local_z, src=z_acc)
    nisa.sendrecv_compute(
        src=local_z,
        dst=full_z,
        send_to_rank=peer,
        recv_from_rank=peer,
        reduce_op=nl.add,
    )

    # Both cores now possess the completely reduced non-causal state.
    state0_tf = nl.ndarray((d0, D), dtype=nl.tfloat32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=state0_tf, src=full_state0)
    if D > 128:
        state1_tf = nl.ndarray(
            (D - 128, D), dtype=nl.tfloat32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=state1_tf, src=full_state1)

    z_broadcast = nl.broadcast_to(full_z, (row_tile, D))

    for t in range(tile_count):
        local_start = t * row_tile
        rows = min(row_tile, rows_per_core - local_start)
        start = core_start + local_start

        q = nl.ndarray((rows, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q, src=Q[start:start + rows, :])
        phi_q = _phi_fp32(q)

        # The denominator retains fp32 operands throughout.
        terms = nl.ndarray((rows, D), dtype=nl.float32, buffer=nl.sbuf)
        denominator = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        inverse = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=terms,
            data1=phi_q,
            data2=z_broadcast[0:rows, :],
            op=nl.multiply,
        )
        nisa.tensor_reduce(
            dst=denominator, op=nl.add, data=terms, axis=1
        )
        nisa.tensor_scalar(
            dst=denominator,
            data=denominator,
            op0=nl.add,
            operand0=eps,
        )
        nisa.reciprocal(dst=inverse, data=denominator)

        numerator = nl.ndarray((rows, D), dtype=nl.float32, buffer=nl.psum)
        qt0 = _transpose_tf32(phi_q[:, 0:d0])
        nisa.nc_matmul(
            dst=numerator,
            stationary=qt0,
            moving=state0_tf,
            accumulate=False,
        )
        if D > 128:
            qt1 = _transpose_tf32(phi_q[:, 128:D])
            nisa.nc_matmul(
                dst=numerator,
                stationary=qt1,
                moving=state1_tf,
                accumulate=True,
            )

        result = nl.ndarray((rows, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=result,
            data=numerator,
            op0=nl.multiply,
            operand0=inverse,
        )
        nisa.dma_copy(dst=out[start:start + rows, :], src=result)

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
        "state_feature_tile": min(128, D),
        "value_tile": D,
        "sequence_splits": 2,
        "rows_per_core": M // 2,
        "state_multiply": "tf32",
        "numerator_multiply": "tf32",
        "normalizer_precision": "fp32",
        "feature_maps": "fused_fp32",
        "state_combine": "two_core_fp32_add",
    }
    return _linear_attention[2](Q, K, V, float(eps), 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
