import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _rope_kernel(q, cos, sin, partitions, rows_per_partition):
    B, S, H, D = q.shape
    K = D // 2
    P = partitions
    G = rows_per_partition
    tile_rows = P * G

    pid = nl.program_id(0)
    cores = nl.num_programs()
    tiles = S // tile_rows
    tiles_per_core = tiles // cores

    assert D % 2 == 0
    assert S % (tile_rows * cores) == 0

    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    qv = q.reshape((B, tiles, P, G, H, D))
    ov = out.reshape((B, tiles, P, G, H, D))
    cv = cos.reshape((tiles, P, G, K))
    sv = sin.reshape((tiles, P, G, K))

    for b in range(B):
        for t in range(tiles_per_core):
            tile = pid * tiles_per_core + t

            # [cos, sin, cos] permits both coefficient orderings using
            # positive-stride slices. Heads are broadcast through views.
            coefficient_rows = nl.ndarray(
                (P, G, 3, K), dtype=q.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=coefficient_rows[:, :, 0, :],
                src=cv[tile],
            )
            nisa.dma_copy(
                dst=coefficient_rows[:, :, 1, :],
                src=sv[tile],
            )
            nisa.tensor_copy(
                dst=coefficient_rows[:, :, 2, :],
                src=coefficient_rows[:, :, 0, :],
            )

            coefficients = coefficient_rows.expand_dim(2).broadcast(2, H)
            coefficients_c_s = coefficients[:, :, :, 0:2, :].reshape(
                (P, G, H, D)
            )
            coefficients_s_c = coefficients[:, :, :, 1:3, :].reshape(
                (P, G, H, D)
            )

            query = nl.ndarray(
                (P, G, H, D), dtype=q.dtype, buffer=nl.sbuf
            )
            products = nl.ndarray(
                (P, G, H, D), dtype=q.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=query, src=qv[b, tile])

            # Two full-width multiplies produce all four fp16 products.
            nisa.tensor_tensor(
                dst=products,
                data1=query,
                data2=coefficients_c_s,
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=query,
                data1=query,
                data2=coefficients_s_c,
                op=nl.multiply,
                engine=nisa.engine.vector,
            )

            query_halves = query.reshape((P, G * H, 2, K))
            product_halves = products.reshape((P, G * H, 2, K))

            # Preserve q1*sin until it has contributed to the second half.
            nisa.tensor_tensor(
                dst=query_halves[:, :, 1, :],
                data1=query_halves[:, :, 1, :],
                data2=query_halves[:, :, 0, :],
                op=nl.add,
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=query_halves[:, :, 0, :],
                data1=product_halves[:, :, 0, :],
                data2=product_halves[:, :, 1, :],
                op=nl.subtract,
                engine=nisa.engine.vector,
            )

            nisa.dma_copy(dst=ov[b, tile], src=query)

    return out


def run(q, cos, sin):
    global _last_config
    B, S, H, D = q.shape
    partitions = 128
    rows_per_partition = 4
    _last_config = {
        "partitions": partitions,
        "rows_per_partition": rows_per_partition,
        "tile_rows": partitions * rows_per_partition,
        "heads_per_row": int(H),
        "pairs_per_head": int(D // 2),
        "lnc": 2,
        "core_split": "sequence",
        "dma_layout": "contiguous_full_heads",
        "coefficient_mapping": "c_s_c_zero_copy_head_broadcast",
        "arithmetic": "fp16_staged",
        "engine": "vector",
        "vector_operations_per_tile": 4,
        "batch_size": int(B),
        "seq_len": int(S),
    }
    return _rope_kernel[2](q, cos, sin, partitions, rows_per_partition)


def get_last_config() -> dict:
    return _last_config.copy()
