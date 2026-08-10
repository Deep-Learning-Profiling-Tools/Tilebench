import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
    TILE = 128
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def phi_kernel(x):
        rows, cols = x.shape
        num_blocks = (rows + (PMAX - 1)) // PMAX
        result = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            free_index = nl.arange(cols)[None, :]
            mask_p = partition_index < (rows - offset)

            x_tile = nl.load(x[offset + partition_index, free_index], mask=mask_p, dtype=nl.float32)
            pos_val = nl.add(x_tile, 1.0)
            exp_val = nl.exp(x_tile)
            is_pos = nl.greater(x_tile, 0.0)
            result_tile = nl.where(is_pos, pos_val, exp_val)

            nl.store(result[offset + partition_index, free_index], value=result_tile, mask=mask_p)

        return result

    @nki.jit
    def stageA_kernel(phi_K, V_aug):
        M, D = phi_K.shape
        _, Dp1 = V_aug.shape
        num_m_tiles = (M + TILE - 1) // TILE
        num_d_tiles = (D + TILE - 1) // TILE

        S_aug = nl.ndarray((num_d_tiles * TILE, Dp1), dtype=nl.float32, buffer=nl.hbm)

        for di in range(num_d_tiles):
            d_offset = di * TILE
            d_valid = min(TILE, D - d_offset)
            psum = nl.zeros((TILE, Dp1), dtype=nl.float32, buffer=nl.psum)

            for mi in range(num_m_tiles):
                m_offset = mi * TILE
                m_partition = nl.arange(TILE)[:, None]
                d_free = nl.arange(TILE)[None, :]
                mask_m = m_partition < (M - m_offset)
                mask_d = d_free < d_valid

                stat_raw = nl.load(phi_K[m_offset + m_partition, d_offset + d_free], mask=(mask_m & mask_d), dtype=nl.float32)
                zero_stat = nl.zeros(stat_raw.shape, dtype=nl.float32, buffer=nl.sbuf)
                stat_tile = nl.where(mask_m & mask_d, stat_raw, zero_stat)

                mov_free = nl.arange(Dp1)[None, :]
                mov_tile = nl.load(V_aug[m_offset + m_partition, mov_free], mask=mask_m, dtype=nl.float32)
                zero_mov = nl.zeros(mov_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
                mov_safe = nl.where(mask_m, mov_tile, zero_mov)

                psum += nisa.nc_matmul(stat_tile, mov_safe)

            result_tile = nl.copy(psum, dtype=nl.float32)
            nl.store(S_aug[d_offset + nl.arange(TILE)[:, None], nl.arange(Dp1)[None, :]], value=result_tile)

        return S_aug

    @nki.jit
    def stageB_kernel(phi_Q, S_aug, D, eps):
        M, D_full = phi_Q.shape
        Dp1 = S_aug.shape[1]
        num_m_tiles = (M + TILE - 1) // TILE
        num_d_tiles = (D + TILE - 1) // TILE

        out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.hbm)

        for mi in range(num_m_tiles):
            m_offset = mi * TILE
            m_partition = nl.arange(TILE)[:, None]
            mask_m = m_partition < (M - m_offset)

            psum = nl.zeros((TILE, Dp1), dtype=nl.float32, buffer=nl.psum)
            for di in range(num_d_tiles):
                d_offset = di * TILE
                d_valid = min(TILE, D - d_offset)
                d_free = nl.arange(TILE)[None, :]
                mask_d = d_free < d_valid

                q_raw = nl.load(phi_Q[m_offset + m_partition, d_offset + d_free], mask=(mask_m & mask_d), dtype=nl.float32)
                zero_q = nl.zeros(q_raw.shape, dtype=nl.float32, buffer=nl.sbuf)
                q_tile = nl.where(mask_m & mask_d, q_raw, zero_q)
                q_t = nisa.nc_transpose(q_tile)

                s_tile = nl.load(S_aug[d_offset + nl.arange(TILE)[:, None], nl.arange(Dp1)[None, :]], dtype=nl.float32)

                psum += nisa.nc_matmul(q_t, s_tile)

            o_aug = nl.copy(psum, dtype=nl.float32)
            o_num = o_aug[:, 0:D]
            o_den = o_aug[:, D:D + 1]
            result_tile = nl.divide(o_num, nl.add(o_den, eps))

            nl.store(out[m_offset + m_partition, nl.arange(D)[None, :]], value=result_tile, mask=mask_m)

        return out


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    
    M, D = Q.shape
    ones_col = torch.ones(M, 1, dtype=V.dtype, device=V.device)
    V_aug = torch.cat([V, ones_col], dim=1)

    phi_Q = phi_kernel(Q)
    phi_K = phi_kernel(K)

    S_aug = stageA_kernel(phi_K, V_aug)
    out = stageB_kernel(phi_Q, S_aug, D, eps)
    return out


def get_last_config() -> dict | None:
    return None
