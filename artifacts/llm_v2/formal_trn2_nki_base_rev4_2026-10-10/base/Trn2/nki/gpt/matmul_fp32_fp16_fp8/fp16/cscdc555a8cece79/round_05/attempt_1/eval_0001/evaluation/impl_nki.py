import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _gemm_fp16(
    a, b, block_m, block_n, cache_b, a_chunk_k, b_chunk_k, b_buffers
):
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
        a_groups = a_current.reshape(
            (block_m // 16, 16, k_tiles, tile_k)
        )

        # Batch K tiles into one hardware-DMA transpose. The middle
        # axes put K before the row groups, leaving each stationary
        # tile contiguous along its M dimension in SBUF.
        for ac in range(
            (k_tiles + a_chunk_tiles - 1) // a_chunk_tiles
        ):
            first_ki = ac * a_chunk_tiles
            count = min(a_chunk_tiles, k_tiles - first_ki)
            source = a_groups[
                :, :, first_ki:first_ki + count, :
            ].permute((1, 2, 0, 3))
            destination = a_cache[
                :, first_ki:first_ki + count, :, :
            ].reshape((tile_k, count, block_m // 16, 16))
            nisa.dma_transpose(
                dst=destination,
                src=source,
                axes=(3, 1, 2, 0),
                dge_mode=nisa.dge_mode.hwdge,
            )

        def columns_body(nb):
            if cache_b:
                b_cache = nl.ndarray(
                    (tile_k, k_tiles, block_n),
                    dtype=b.dtype,
                    buffer=nl.sbuf,
                )

                for bc in range(
                    (k_tiles + b_chunk_tiles - 1) // b_chunk_tiles
                ):
                    first_ki = bc * b_chunk_tiles
                    count = min(b_chunk_tiles, k_tiles - first_ki)
                    source = b_blocks[
                        first_ki:first_ki + count, :, nb, :
                    ].permute((1, 0, 2))
                    nisa.dma_copy(
                        dst=b_cache[:, first_ki:first_ki + count, :],
                        src=source,
                    )

                acc_m_tiles = 8 // n_tiles
                groups = m_tiles // acc_m_tiles

                for mg in range(groups):
                    acc = nl.ndarray(
                        (tile_m, acc_m_tiles, block_n),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )

                    for ki in range(k_tiles):
                        for mi in range(acc_m_tiles):
                            ai = mg * acc_m_tiles + mi
                            for ni in range(n_tiles):
                                n0 = ni * tile_n
                                nisa.nc_matmul(
                                    dst=acc[:, mi, n0:n0 + tile_n],
                                    stationary=a_cache[:, ki, ai, :],
                                    moving=b_cache[
                                        :, ki, n0:n0 + tile_n
                                    ],
                                    accumulate=(ki > 0),
                                )

                    result = nl.ndarray(
                        (tile_m, block_n),
                        dtype=a.dtype,
                        buffer=nl.sbuf,
                    )
                    for mi in range(acc_m_tiles):
                        ai = mg * acc_m_tiles + mi
                        nisa.tensor_copy(dst=result, src=acc[:, mi, :])
                        nisa.dma_copy(
                            dst=out.ap(
                                pattern=[[N, tile_m], [1, block_n]],
                                offset=ai * tile_m * N,
                                scalar_offset=store_offset,
                                indirect_dim=1,
                            ),
                            src=result,
                        )

            else:
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

                for bc in range(
                    (k_tiles + b_chunk_tiles - 1) // b_chunk_tiles
                ):
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
    assert M == 4096 and N == 4096
    assert 1024 <= K <= 20480 and K % 1024 == 0

    full_cache_budget = 24 * 1024 * 1024
    sbuf_budget = 26 * 1024 * 1024
    reserve_bytes = 512 * 1024
    a_chunk_k = min(K, 4096)

    # Full B caching amortizes its initial load over a large row panel.
    # Otherwise use a PSUM-resident 512x1024 output block and short,
    # circularly buffered B loads to reduce each block's startup delay.
    cache_b = (2048 + 1024) * K * 2 <= full_cache_budget

    if cache_b:
        block_m = 2048
        block_n = 1024
        while (
            block_n * 2 <= N
            and (block_m + block_n * 2) * K * 2
            <= full_cache_budget
        ):
            block_n *= 2

        b_chunk_k = min(K, 4096)
        b_buffers = 1
        b_load_bytes = K * block_n * 2
        acc_m_tiles = 8 // (block_n // 512)
    else:
        block_m = 512
        block_n = 1024
        b_chunk_k = 1024
        b_buffers = 3

        fixed_bytes = (
            block_m * K * 2
            + 128 * block_n * 2
            + reserve_bytes
        )
        if (
            fixed_bytes + b_buffers * b_chunk_k * block_n * 2
            > sbuf_budget
        ):
            b_buffers = 2

        b_load_bytes = b_buffers * b_chunk_k * block_n * 2
        acc_m_tiles = block_m // 128

    a_cache_bytes = block_m * K * 2
    result_bytes = 128 * block_n * 2
    assert (
        a_cache_bytes + b_load_bytes + result_bytes + reserve_bytes
        <= sbuf_budget
    )

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_m": 128,
        "tile_n": 512,
        "tile_k": 128,
        "block_m": block_m,
        "block_n": block_n,
        "accumulator_m_tiles": acc_m_tiles,
        "a_cache_k": K,
        "a_cache_bytes_per_core": a_cache_bytes,
        "a_load_kind": "hwdge_transpose_4d",
        "a_load_chunk_k": a_chunk_k,
        "a_transpose_first_axis": 16,
        "b_cache_all_k": cache_b,
        "b_load_chunk_k": b_chunk_k,
        "b_load_buffers": b_buffers,
        "b_load_bytes_per_core": b_load_bytes,
        "full_cache_budget_bytes": full_cache_budget,
        "sbuf_budget_bytes": sbuf_budget,
        "m_blocks_per_core": M // block_m // 2,
        "n_blocks": N // block_n,
        "k_tiles": K // 128,
        "accumulation_dtype": "float32",
        "k_loop": "static",
        "output_block_loops": "rolled",
        "output_address": "single_scalar_offset",
    }

    return _gemm_fp16[2](
        a,
        b,
        block_m,
        block_n,
        cache_b,
        a_chunk_k,
        b_chunk_k,
        b_buffers,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
