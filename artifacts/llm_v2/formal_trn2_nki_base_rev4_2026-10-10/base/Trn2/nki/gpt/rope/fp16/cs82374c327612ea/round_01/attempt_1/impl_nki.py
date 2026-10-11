import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _rope_kernel(q, cos, sin, tile_rows):
    B, S, H, D = q.shape
    K = D // 2
    P = tile_rows

    pid = nl.program_id(0)
    cores = nl.num_programs()
    rows_per_core = S // cores

    assert D % 2 == 0
    assert S % (cores * P) == 0

    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    qv = q.reshape((B, S, H, 2, K))
    ov = out.reshape((B, S, H, 2, K))

    for b in range(B):
        for t in range(rows_per_core // P):
            first = pid * rows_per_core + t * P
            last = first + P

            c_row = nl.ndarray((P, K), dtype=q.dtype, buffer=nl.sbuf)
            s_row = nl.ndarray((P, K), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=c_row, src=cos[first:last, :])
            nisa.dma_copy(dst=s_row, src=sin[first:last, :])

            c = nl.broadcast_to(c_row.reshape((P, 1, K)), (P, H, K))
            s = nl.broadcast_to(s_row.reshape((P, 1, K)), (P, H, K))

            q1 = nl.ndarray((P, H, K), dtype=q.dtype, buffer=nl.sbuf)
            q2 = nl.ndarray((P, H, K), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q1, src=qv[b, first:last, :, 0, :])
            nisa.dma_copy(dst=q2, src=qv[b, first:last, :, 1, :])

            out1 = nl.ndarray((P, H, K), dtype=q.dtype, buffer=nl.sbuf)
            out2 = nl.ndarray((P, H, K), dtype=q.dtype, buffer=nl.sbuf)

            # Each product is rounded to the input dtype, as in the reference.
            nisa.tensor_tensor(
                dst=out1, data1=q1, data2=c,
                op=nl.multiply, engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=out2, data1=q2, data2=c,
                op=nl.multiply, engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=q1, data1=q1, data2=s,
                op=nl.multiply, engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=q2, data1=q2, data2=s,
                op=nl.multiply, engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=out1, data1=out1, data2=q2,
                op=nl.subtract, engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=out2, data1=out2, data2=q1,
                op=nl.add, engine=nisa.engine.vector,
            )

            nisa.dma_copy(dst=ov[b, first:last, :, 0, :], src=out1)
            nisa.dma_copy(dst=ov[b, first:last, :, 1, :], src=out2)

    return out


def run(q, cos, sin):
    global _last_config
    B, S, H, D = q.shape
    tile_rows = 128
    _last_config = {
        "tile_rows": tile_rows,
        "heads_per_tile": int(H),
        "pairs_per_head": int(D // 2),
        "lnc": 2,
        "core_split": "sequence",
        "coefficient_mapping": "broadcast_over_heads",
        "arithmetic": "fp16_staged",
        "engine": "vector",
        "batch_size": int(B),
        "seq_len": int(S),
    }
    return _rope_kernel[2](q, cos, sin, tile_rows)


def get_last_config() -> dict:
    return _last_config.copy()
