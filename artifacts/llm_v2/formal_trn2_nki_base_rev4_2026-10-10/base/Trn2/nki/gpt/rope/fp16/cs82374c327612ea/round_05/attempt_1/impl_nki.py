import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _rope_kernel(q, cos, sin, partitions, rows_per_partition, query_buffers):
    B, S, H, D = q.shape
    K = D // 2
    P = partitions
    G = rows_per_partition
    tile_rows = P * G

    pid = nl.program_id(0)
    cores = nl.num_programs()
    tiles = S // tile_rows
    tiles_per_core = tiles // cores
    core_begin = pid * tiles_per_core
    core_end = core_begin + tiles_per_core

    assert D % 2 == 0
    assert S % (tile_rows * cores) == 0
    assert query_buffers >= 1
    assert query_buffers <= 3
    assert query_buffers <= tiles_per_core

    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    qv = q.reshape((B, tiles, P, G, H, D))
    ov = out.reshape((B, tiles, P, G, H, D))
    cv = cos.reshape((tiles, P, G, K))
    sv = sin.reshape((tiles, P, G, K))

    # Fetch this core's coefficients once. The additional cosine copy
    # supplies both coefficient orderings through positive-stride views.
    coefficient_cache = nl.ndarray(
        (P, tiles_per_core, G, 3, K),
        dtype=q.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=coefficient_cache[:, :, :, 0, :],
        src=cv[core_begin:core_end, :, :, :].permute((1, 0, 2, 3)),
    )
    nisa.dma_copy(
        dst=coefficient_cache[:, :, :, 1, :],
        src=sv[core_begin:core_end, :, :, :].permute((1, 0, 2, 3)),
    )
    nisa.tensor_copy(
        dst=coefficient_cache[:, :, :, 2, :],
        src=coefficient_cache[:, :, :, 0, :],
    )

    # Independent allocations keep prefetched queries live while another
    # tile is being rotated or written back.
    query0 = nl.ndarray((P, G, H, D), dtype=q.dtype, buffer=nl.sbuf)
    query1 = nl.ndarray((P, G, H, D), dtype=q.dtype, buffer=nl.sbuf)
    query2 = nl.ndarray((P, G, H, D), dtype=q.dtype, buffer=nl.sbuf)
    products = nl.ndarray((P, G, H, D), dtype=q.dtype, buffer=nl.sbuf)

    for b in range(B):
        nisa.dma_copy(dst=query0, src=qv[b, core_begin])
        if query_buffers >= 2:
            nisa.dma_copy(dst=query1, src=qv[b, core_begin + 1])
        if query_buffers >= 3:
            nisa.dma_copy(dst=query2, src=qv[b, core_begin + 2])

        for t in range(tiles_per_core):
            slot = t % query_buffers
            if slot == 0:
                query = query0
            elif slot == 1:
                query = query1
            else:
                query = query2

            coefficient_rows = coefficient_cache[:, t, :, :, :]
            coefficients = coefficient_rows.expand_dim(2).broadcast(2, H)
            coefficients_c_s = coefficients[:, :, :, 0:2, :].reshape(
                (P, G, H, D)
            )
            coefficients_s_c = coefficients[:, :, :, 1:3, :].reshape(
                (P, G, H, D)
            )

            # Keep the reference's fp16 product rounding.
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

            nisa.dma_copy(dst=ov[b, core_begin + t], src=query)

            # Refill only after the current contents have been stored.
            # Other query buffers can compute independently of this DMA.
            if t + query_buffers < tiles_per_core:
                nisa.dma_copy(
                    dst=query,
                    src=qv[b, core_begin + t + query_buffers],
                )

    return out


def run(q, cos, sin):
    global _last_config
    B, S, H, D = q.shape
    partitions = 128
    rows_per_partition = 4
    tile_rows = partitions * rows_per_partition
    tiles_per_core = S // (2 * tile_rows)
    query_buffers = min(3, tiles_per_core)

    _last_config = {
        "partitions": partitions,
        "rows_per_partition": rows_per_partition,
        "tile_rows": tile_rows,
        "query_buffers": query_buffers,
        "coefficient_cache_tiles_per_core": tiles_per_core,
        "heads_per_row": int(H),
        "pairs_per_head": int(D // 2),
        "lnc": 2,
        "core_split": "sequence",
        "dma_layout": "contiguous_full_heads",
        "coefficient_mapping": "per_core_sbuf_c_s_c_head_broadcast",
        "pipeline": "prefetch_rotate_store_refill",
        "arithmetic": "fp16_staged",
        "engine": "vector",
        "vector_operations_per_tile": 4,
        "batch_size": int(B),
        "seq_len": int(S),
    }
    return _rope_kernel[2](
        q, cos, sin, partitions, rows_per_partition, query_buffers
    )


def get_last_config() -> dict:
    return _last_config.copy()
