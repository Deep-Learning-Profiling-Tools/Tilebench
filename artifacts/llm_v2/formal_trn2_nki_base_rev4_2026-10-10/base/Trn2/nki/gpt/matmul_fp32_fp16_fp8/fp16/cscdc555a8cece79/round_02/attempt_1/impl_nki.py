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

    a_cache = nl.ndarray(
        (tile_k, k_tiles, m_tiles, tile_m),
        dtype=a.dtype,
        buffer=nl.sbuf,
    )

    pid = nl.program_id(0)
    programs = nl.num_programs()
    blocks_per_program = m_blocks // programs

    # One runtime element offset handles both output coordinates without
    # combining two dynamic tensor selects.
    store_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(
        dst=store_offset,
        value=pid * blocks_per_program * block_m * N,
    )

    def rows_body(mb):
        a_current = a_blocks[mb]

        a_raw = nl.ndarray(
            (tile_m, buffers, m_tiles, tile_k),
            dtype=a.dtype,
            buffer=nl.sbuf,
        )
        a_transposed = nl.ndarray(
            (tile_k, tile_m),
            dtype=a.dtype,
            buffer=nl.psum,
        )

        # Finish all transposes before using the eight PSUM banks for GEMM.
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

        # The column loop advanced by N elements; advance to the next M block.
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
    assert M == 4096 and N == 4096 and K % 128 == 0

    # Keep the resident A block within a fixed 20 MiB SBUF budget.
    cache_budget = 20 * 1024 * 1024
    block_m = 1024
    if block_m * K * 2 > cache_budget:
        block_m = 512

    # Eight banks, each holding 512 fp32 values per partition.
    block_n = 4096 // (block_m // 128)
    assert block_m * K * 2 <= cache_budget

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_m": 128,
        "tile_n": 512,
        "tile_k": 128,
        "block_m": block_m,
        "block_n": block_n,
        "a_cache_k": K,
        "a_cache_bytes_per_core": block_m * K * 2,
        "a_cache_budget_bytes": cache_budget,
        "a_load_buffers": 2,
        "b_load_buffers": 2,
        "m_blocks_per_core": M // block_m // 2,
        "n_blocks": N // block_n,
        "k_tiles": K // 128,
        "accumulation_dtype": "float32",
        "k_loop": "static",
        "output_block_loops": "rolled",
        "output_address": "single_scalar_offset",
    }

    return _gemm_fp16[2](a, b, block_m, block_n, 128, 2)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
