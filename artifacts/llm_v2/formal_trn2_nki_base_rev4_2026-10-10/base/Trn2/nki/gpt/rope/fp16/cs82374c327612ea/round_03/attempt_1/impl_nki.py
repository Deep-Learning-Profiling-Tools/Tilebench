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
    qv = q.reshape((B, tiles, P, G * H, 2, K))
    ov = out.reshape((B, tiles, P, G * H, 2, K))
    cv = cos.reshape((tiles, P, G, K))
    sv = sin.reshape((tiles, P, G, K))

    for b in range(B):
        for t in range(tiles_per_core):
            tile = pid * tiles_per_core + t

            # Pack cosine and sine together, then replicate only over heads.
            # Unlike separate full cosine/sine tiles, each coefficient appears
            # only once per head in the expanded tile.
            coefficient_rows = nl.ndarray(
                (P, G, 2, K), dtype=q.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=coefficient_rows[:, :, 0, :], src=cv[tile]
            )
            nisa.dma_copy(
                dst=coefficient_rows[:, :, 1, :], src=sv[tile]
            )
            coefficients = nl.broadcast_to(
                coefficient_rows.reshape((P, G, 1, 2, K)),
                (P, G, H, 2, K),
            ).reshape((P, G * H, 2, K))

            query = nl.ndarray(
                (P, G * H, 2, K), dtype=q.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=query, src=qv[b, tile])

            # One full-width multiply forms [q1*cos, q2*sin].
            products = nl.ndarray(
                (P, G * H, 2, K), dtype=q.dtype, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=products,
                data1=query,
                data2=coefficients,
                op=nl.multiply,
                engine=nisa.engine.vector,
            )

            # Reuse the query tile for the other two products.
            nisa.tensor_tensor(
                dst=query[:, :, 0, :],
                data1=query[:, :, 0, :],
                data2=coefficients[:, :, 1, :],
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=query[:, :, 1, :],
                data1=query[:, :, 1, :],
                data2=coefficients[:, :, 0, :],
                op=nl.multiply,
                engine=nisa.engine.vector,
            )

            # Consume q1*sin before overwriting its slot with the first output.
            # Each product and final sum is rounded to the input dtype.
            nisa.tensor_tensor(
                dst=query[:, :, 1, :],
                data1=query[:, :, 1, :],
                data2=query[:, :, 0, :],
                op=nl.add,
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=query[:, :, 0, :],
                data1=products[:, :, 0, :],
                data2=products[:, :, 1, :],
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
        "coefficient_mapping": "packed_cos_sin_broadcast_over_heads",
        "arithmetic": "fp16_staged",
        "engine": "vector",
        "batch_size": int(B),
        "seq_len": int(S),
    }
    return _rope_kernel[2](q, cos, sin, partitions, rows_per_partition)


def get_last_config() -> dict:
    return _last_config.copy()
