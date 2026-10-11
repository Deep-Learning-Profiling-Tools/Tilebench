import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _gemm_fp16_bulk(a, b, block_m, block_n, a_chunk_k, b_chunk_k, b_buffers):
    M, K = a.shape
    _, N = b.shape

    tile_m = 128
    tile_n = 512
    tile_k = 128
    m_tiles = block_m // tile_m
    n_tiles = block_n // tile_n
    k_tiles = K // tile_k
    a_chunk_tiles = a_chunk_k // tile_k
    b_chunk_tiles = b_chunk_k // tile_k
    m_blocks = M // block_m
    n_blocks = N // block_n

    out = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    a_blocks = a.reshape((m_blocks, m_tiles, tile_m, K))
    b_blocks = b.reshape((k_tiles, tile_k, n_blocks, block_n))

    a_cache = nl.ndarray(
        (tile_k, k_tiles, m_tiles, tile_m),
        dtype=a.dtype,
        buffer=nl.sbuf,
    )

    pid = nl.program_id(0)
    programs = nl.num_programs()
    blocks_per_program = m_blocks // programs

    store_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(
        dst=store_offset,
        value=pid * blocks_per_program * block_m * N,
    )

    def rows_body(mb):
        a_current = a_blocks[mb]

        # Load contiguous K slabs before transposing their 128x128 tiles.
        a_raw = nl.ndarray(
            (tile_m, m_tiles, a_chunk_tiles, tile_k),
            dtype=a.dtype,
            buffer=nl.sbuf,
        )
        a_transposed = nl.ndarray(
            (tile_k, tile_m),
            dtype=a.dtype,
            buffer=nl.psum,
        )

        for ac in range(K // a_chunk_k):
            k0 = ac * a_chunk_k
            source = a_current[:, :, k0:k0 + a_chunk_k].permute((1, 0, 2))
            nisa.dma_copy(
                dst=a_raw.reshape((tile_m, m_tiles, a_chunk_k)),
                src=source,
            )

            for ak in range(a_chunk_tiles):
                ki = ac * a_chunk_tiles + ak
                for mi in range(m_tiles):
                    nisa.nc_transpose(
                        dst=a_transposed,
                        data=a_raw[:, mi, ak, :],
                        engine=nisa.engine.tensor,
                    )
                    nisa.tensor_copy(
                        dst=a_cache[:, ki, mi, :],
                        src=a_transposed,
                    )

        def columns_body(nb):
            acc = nl.ndarray(
                (tile_m, m_tiles, block_n),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            b_panels = nl.ndarray(
                (tile_k, b_buffers, b_chunk_tiles, block_n),
                dtype=b.dtype,
                buffer=nl.sbuf,
            )

            # Alternating bulk-load buffers permit DMA to overlap a whole
            # group of GEMM instructions, rather than issuing one DMA per K tile.
            for bc in range((k_tiles + b_chunk_tiles - 1) // b_chunk_tiles):
                slot = bc % b_buffers
                first_ki = bc * b_chunk_tiles
                count = min(b_chunk_tiles, k_tiles - first_ki)

                source = b_blocks[
                    first_ki:first_ki + count, :, nb, :
                ].permute((1, 0, 2))
                nisa.dma_copy(
                    dst=b_panels[:, slot, 0:count, :],
                    src=source,
                )

                for bk in range(count):
                    ki = first_ki + bk
                    for mi in range(m_tiles):
                        for ni in range(n_tiles):
                            n0 = ni * tile_n
                            nisa.nc_matmul(
                                dst=acc[:, mi, n0:n0 + tile_n],
                                stationary=a_cache[:, ki, mi, :],
                                moving=b_panels[
                                    :, slot, bk, n0:n0 + tile_n
                                ],
                                accumulate=(ki > 0),
                            )

            result = nl.ndarray(
                (tile_m, block_n),
                dtype=a.dtype,
                buffer=nl.sbuf,
            )
            for mi in range(m_tiles):
                nisa.tensor_copy(dst=result, src=acc[:, mi, :])
                nisa.dma_copy(
                    dst=out.ap(
                        pattern=[[N, tile_m], [1, block_n]],
                        offset=mi * tile_m * N,
                        scalar_offset=store_offset,
                        indirect_dim=1,
                    ),
                    src=result,
                )

            nisa.tensor_scalar(
                dst=store_offset,
                data=store_offset,
                op0=nl.add,
                operand0=block_n,
            )

        nl.fori_loop(0, n_blocks, columns_body)

        nisa.tensor_scalar(
            dst=store_offset,
            data=store_offset,
            op0=nl.add,
            operand0=(block_m - 1) * N,
        )

    nl.fori_loop(
        pid * blocks_per_program,
        (pid + 1) * blocks_per_program,
        rows_body,
    )
    return out


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG

    M, K = a.shape
    K_b, N = b.shape
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert K == K_b
    assert M == 4096 and N == 4096 and K % 1024 == 0

    cache_budget = 20 * 1024 * 1024
    block_m = 1024
    if block_m * K * 2 > cache_budget:
        block_m = 512

    block_n = 4096 // (block_m // 128)
    a_chunk_k = 1024
    b_buffers = 2

    a_cache_bytes = block_m * K * 2
    a_raw_bytes = block_m * a_chunk_k * 2
    result_bytes = 128 * block_n * 2

    # Choose the largest power-of-two slab within a conservative SBUF budget.
    sbuf_budget = 26 * 1024 * 1024
    reserve_bytes = 512 * 1024
    fixed_bytes = (
        a_cache_bytes + a_raw_bytes + result_bytes + reserve_bytes
    )
    b_chunk_k = 1024
    while (
        b_chunk_k * 2 <= min(K, 4096)
        and fixed_bytes + b_buffers * (b_chunk_k * 2) * block_n * 2
        <= sbuf_budget
    ):
        b_chunk_k *= 2

    assert a_cache_bytes <= cache_budget
    assert (
        fixed_bytes + b_buffers * b_chunk_k * block_n * 2
        <= sbuf_budget
    )

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_m": 128,
        "tile_n": 512,
        "tile_k": 128,
        "block_m": block_m,
        "block_n": block_n,
        "a_cache_k": K,
        "a_cache_bytes_per_core": a_cache_bytes,
        "a_cache_budget_bytes": cache_budget,
        "a_load_chunk_k": a_chunk_k,
        "a_load_buffers": 1,
        "b_load_chunk_k": b_chunk_k,
        "b_load_buffers": b_buffers,
        "b_load_bytes_per_core": b_buffers * b_chunk_k * block_n * 2,
        "sbuf_budget_bytes": sbuf_budget,
        "m_blocks_per_core": M // block_m // 2,
        "n_blocks": N // block_n,
        "k_tiles": K // 128,
        "accumulation_dtype": "float32",
        "k_loop": "static_bulk_load",
        "output_block_loops": "rolled",
        "output_address": "single_scalar_offset",
    }

    return _gemm_fp16_bulk[2](
        a, b, block_m, block_n, a_chunk_k, b_chunk_k, b_buffers
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
