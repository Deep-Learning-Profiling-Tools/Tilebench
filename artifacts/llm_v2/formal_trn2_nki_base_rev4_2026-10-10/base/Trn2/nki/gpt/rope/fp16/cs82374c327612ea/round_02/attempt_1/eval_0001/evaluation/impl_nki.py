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

    # Each partition contains G complete sequence rows. Both query halves
    # are transferred together, giving contiguous input and output DMAs.
    qv = q.reshape((B, tiles, P, G * H, 2, K))
    ov = out.reshape((B, tiles, P, G * H, 2, K))
    cv = cos.reshape((tiles, P, G, K))
    sv = sin.reshape((tiles, P, G, K))

    for b in range(B):
        for t in range(tiles_per_core):
            tile = pid * tiles_per_core + t

            query = nl.ndarray(
                (P, G * H, 2, K), dtype=q.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=query, src=qv[b, tile])

            sin_row = nl.ndarray((P, G, K), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sin_row, src=sv[tile])
            sin_full = nl.broadcast_to(
                sin_row.reshape((P, G, 1, K)),
                (P, G, H * 2, K),
            ).reshape((P, G * H, 2, K))

            sin_product = nl.ndarray(
                (P, G * H, 2, K), dtype=q.dtype, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=sin_product,
                data1=query,
                data2=sin_full,
                op=nl.multiply,
                engine=nisa.engine.vector,
            )

            cos_row = nl.ndarray((P, G, K), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=cos_row, src=cv[tile])
            cos_full = nl.broadcast_to(
                cos_row.reshape((P, G, 1, K)),
                (P, G, H * 2, K),
            ).reshape((P, G * H, 2, K))

            nisa.tensor_tensor(
                dst=query,
                data1=query,
                data2=cos_full,
                op=nl.multiply,
                engine=nisa.engine.vector,
            )

            # Products and final sums are rounded to fp16, matching the
            # reference. The sine products remain unchanged throughout.
            nisa.tensor_tensor(
                dst=query[:, :, 0, :],
                data1=query[:, :, 0, :],
                data2=sin_product[:, :, 1, :],
                op=nl.subtract,
                engine=nisa.engine.vector,
            )
            nisa.tensor_tensor(
                dst=query[:, :, 1, :],
                data1=query[:, :, 1, :],
                data2=sin_product[:, :, 0, :],
                op=nl.add,
                engine=nisa.engine.vector,
            )

            nisa.dma_copy(dst=ov[b, tile], src=query)

    return out


def run(q, cos, sin):
    global _last_config
    B, S, H, D = q.shape
    partitions = 128
    rows_per_partition = 2
    _last_config = {
        "partitions": partitions,
        "rows_per_partition": rows_per_partition,
        "tile_rows": partitions * rows_per_partition,
        "heads_per_row": int(H),
        "pairs_per_head": int(D // 2),
        "lnc": 2,
        "core_split": "sequence",
        "dma_layout": "contiguous_full_heads",
        "coefficient_mapping": "broadcast_over_heads_and_halves",
        "arithmetic": "fp16_staged",
        "engine": "vector",
        "batch_size": int(B),
        "seq_len": int(S),
    }
    return _rope_kernel[2](q, cos, sin, partitions, rows_per_partition)


def get_last_config() -> dict:
    return _last_config.copy()
