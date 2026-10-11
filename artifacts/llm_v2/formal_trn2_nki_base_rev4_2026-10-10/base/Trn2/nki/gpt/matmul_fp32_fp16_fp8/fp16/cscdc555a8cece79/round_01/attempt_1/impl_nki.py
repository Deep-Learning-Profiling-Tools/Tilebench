import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _gemm_fp16(a, b, block_m, block_n, tile_k, buffers):
    M, K = a.shape
    _, N = b.shape

    tile_m = 128
    tile_n = 512
    m_tiles = block_m // tile_m
    n_tiles = block_n // tile_n
    k_tiles = K // tile_k
    m_blocks = M // block_m
    n_blocks = N // block_n

    out = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    a_blocks = a.reshape((m_blocks, m_tiles, tile_m, K))
    b_blocks = b.reshape((k_tiles, tile_k, n_blocks, block_n))
    out_blocks = out.reshape(
        (m_blocks, m_tiles, tile_m, n_blocks, block_n)
    )

    # Cache one M block over the entire contraction dimension. Transposing
    # before computing allows all eight PSUM banks to hold GEMM accumulators.
    a_cache = nl.ndarray(
        (tile_k, k_tiles, m_tiles, tile_m),
        dtype=a.dtype,
        buffer=nl.sbuf,
    )

    pid = nl.program_id(0)
    programs = nl.num_programs()
    blocks_per_program = m_blocks // programs

    def rows_body(mb):
        a_current = a_blocks[mb]
        out_current = out_blocks[mb]

        a_raw = nl.ndarray(
            (tile_m, buffers, m_tiles, tile_k),
            dtype=a.dtype,
            buffer=nl.sbuf,
        )
        a_transposed = nl.ndarray(
            (tile_k, tile_m), dtype=a.dtype, buffer=nl.psum
        )

        for ki in range(k_tiles):
            slot = ki % buffers
            k0 = ki * tile_k
            source = a_current[:, :, k0:k0 + tile_k].permute((1, 0, 2))
            nisa.dma_copy(dst=a_raw[:, slot, :, :], src=source)

            for mi in range(m_tiles):
                nisa.nc_transpose(
                    dst=a_transposed,
                    data=a_raw[:, slot, mi, :],
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
                (tile_k, buffers, block_n),
                dtype=b.dtype,
                buffer=nl.sbuf,
            )

            # Static K indexing permits direct accesses to the resident
            # stationary tiles. Alternating B buffers overlap DMA with GEMM.
            for ki in range(k_tiles):
                slot = ki % buffers
                nisa.dma_copy(
                    dst=b_panels[:, slot, :],
                    src=b_blocks[ki, :, nb, :],
                )

                for mi in range(m_tiles):
                    for ni in range(n_tiles):
                        n0 = ni * tile_n
                        nisa.nc_matmul(
                            dst=acc[:, mi, n0:n0 + tile_n],
                            stationary=a_cache[:, ki, mi, :],
                            moving=b_panels[:, slot, n0:n0 + tile_n],
                            accumulate=(ki > 0),
                        )

            result = nl.ndarray(
                (tile_m, block_n), dtype=a.dtype, buffer=nl.sbuf
            )
            for mi in range(m_tiles):
                nisa.tensor_copy(dst=result, src=acc[:, mi, :])
                nisa.dma_copy(
                    dst=out_current[mi, :, nb, :],
                    src=result,
                )

        nl.fori_loop(0, n_blocks, columns_body)

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
    assert M % 1024 == 0 and N % 1024 == 0 and K % 128 == 0

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_m": 128,
        "tile_n": 512,
        "tile_k": 128,
        "block_m": 512,
        "block_n": 1024,
        "a_cache_k": K,
        "a_cache_bytes_per_core": 512 * K * 2,
        "a_load_buffers": 2,
        "b_load_buffers": 2,
        "m_blocks_per_core": M // 1024,
        "n_blocks": N // 1024,
        "k_tiles": K // 128,
        "accumulation_dtype": "float32",
        "k_loop": "static",
        "output_block_loops": "rolled",
    }

    return _gemm_fp16[2](a, b, 512, 1024, 128, 2)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
