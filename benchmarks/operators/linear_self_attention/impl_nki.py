import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
    TILE = 128
except ImportError:
    nki = None


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def phi_kernel(x):
        """Linear-attention feature map ``phi(x) = x + 1 if x > 0 else exp(x)``.

        Args:
            x: [rows, cols] tensor in HBM.

        Returns:
            [rows, cols] fp32 tensor in HBM.

        Notes:
            ``nisa.select_reduce`` implements ``np.where`` but only accepts a scalar
            (or ``[P, 1]`` column) for ``on_false``, so the two branches cannot be
            selected in a single instruction. Instead each branch is selected against
            0.0 -- the ``x > 0`` branch directly, the ``exp`` branch with
            ``reverse_pred=True`` -- and the two disjoint halves are added.
        """
        rows, cols = x.shape
        kernel_assert(len(x.shape) == 2, "phi input must be 2D")
        kernel_assert(cols <= nl.tile_size.sbuf_fmax, "cols exceeds the SBUF free dimension")

        result = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.shared_hbm)

        num_row_tiles = div_ceil(rows, PMAX)

        for row_tile in range(num_row_tiles):
            # Boundary-clamped tile: the allocation itself is sized to the valid
            # extent, so no load/store masking is required.
            row_start = row_tile * PMAX
            row_size = min(PMAX, rows - row_start)

            # DMA cannot convert dtypes: load in the input dtype and let the
            # Vector/Scalar engines widen to fp32 on their way out.
            x_tile = nl.ndarray((row_size, cols), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[row_start:row_start + row_size, 0:cols])

            pos_val = nl.ndarray((row_size, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos_val, data=x_tile, op0=nl.add, operand0=1.0)

            exp_val = nl.ndarray((row_size, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_val, op=nl.exp, data=x_tile)

            # select_reduce requires an integer-typed predicate.
            is_pos = nl.ndarray((row_size, cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=is_pos, data=x_tile, op0=nl.greater, operand0=0.0)

            pos_part = nl.ndarray((row_size, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=pos_part, predicate=is_pos,
                               on_true=pos_val, on_false=0.0)

            exp_part = nl.ndarray((row_size, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=exp_part, predicate=is_pos,
                               on_true=exp_val, on_false=0.0, reverse_pred=True)

            result_tile = nl.ndarray((row_size, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=result_tile, data1=pos_part, data2=exp_part, op=nl.add)

            nisa.dma_copy(dst=result[row_start:row_start + row_size, 0:cols], src=result_tile)

        return result

    @nki.jit
    def stageA_kernel(phi_K, V_aug):
        """``S_aug = phi_K.T @ V_aug`` -- [D, D+1] linear-attention state.

        Args:
            phi_K: [M, D] fp32 tensor in HBM.
            V_aug: [M, D+1] fp32 tensor in HBM (``V`` with a column of ones appended).

        Returns:
            [D, D+1] fp32 tensor in HBM.

        Notes:
            ``nc_matmul`` computes ``dst[M, N] = stationary[K, M].T @ moving[K, N]``,
            so the contraction axis M (of the attention problem) is the partition axis
            of both operands. Each D-tile owns one PSUM bank that accumulates across
            all M-tiles via ``accumulate=True``.
        """
        M, D = phi_K.shape
        M_v, Dp1 = V_aug.shape
        kernel_assert(M == M_v, "phi_K and V_aug must have the same number of rows")
        kernel_assert(Dp1 <= nl.tile_size.psum_fmax, "D+1 exceeds the PSUM free dimension")

        num_m_tiles = div_ceil(M, TILE)
        num_d_tiles = div_ceil(D, TILE)

        S_aug = nl.ndarray((D, Dp1), dtype=nl.float32, buffer=nl.shared_hbm)

        for d_tile in range(num_d_tiles):
            d_start = d_tile * TILE
            d_size = min(TILE, D - d_start)

            psum = nl.ndarray((d_size, Dp1), dtype=nl.float32, buffer=nl.psum)

            for m_tile in range(num_m_tiles):
                m_start = m_tile * TILE
                m_size = min(TILE, M - m_start)

                # Both operands are clamped to the valid extent on the partition (M)
                # and free (D / D+1) axes, so there is no garbage to mask out.
                stat_tile = nl.ndarray((m_size, d_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=stat_tile,
                              src=phi_K[m_start:m_start + m_size, d_start:d_start + d_size])

                mov_raw = nl.ndarray((m_size, Dp1), dtype=V_aug.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mov_raw,
                              src=V_aug[m_start:m_start + m_size, 0:Dp1])

                # nc_matmul operands must share the (fp32) dtype of the phi tile.
                if V_aug.dtype == nl.float32:
                    mov_tile = mov_raw
                else:
                    mov_tile = nl.ndarray((m_size, Dp1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=mov_tile, src=mov_raw)

                nisa.nc_matmul(dst=psum, stationary=stat_tile, moving=mov_tile,
                               accumulate=(m_tile > 0))

            result_tile = nl.ndarray((d_size, Dp1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_tile, src=psum)
            nisa.dma_copy(dst=S_aug[d_start:d_start + d_size, 0:Dp1], src=result_tile)

        return S_aug

    @nki.jit
    def stageB_kernel(phi_Q, S_aug, eps):
        """``out = (phi_Q @ S_aug)[:, :D] / ((phi_Q @ S_aug)[:, D] + eps)``.

        Args:
            phi_Q: [M, D] fp32 tensor in HBM.
            S_aug: [D, D+1] fp32 tensor in HBM (output of ``stageA_kernel``).
            eps: denominator epsilon (compile-time constant).

        Returns:
            [M, D] fp32 tensor in HBM.

        Notes:
            ``nc_matmul`` contracts on the partition axis, so each ``phi_Q`` tile is
            transposed to ``[D_tile, M_tile]`` with ``nc_transpose`` (Tensor Engine,
            SBUF -> PSUM) before being used as the stationary operand. The division is
            expressed as ``nisa.reciprocal`` followed by a ``tensor_scalar`` multiply
            with the ``[P, 1]`` reciprocal column broadcast across the free axis --
            ``nl.divide`` is not a valid ISA operator.
        """
        M, D = phi_Q.shape
        D_s, Dp1 = S_aug.shape
        kernel_assert(D == D_s, "phi_Q columns must match S_aug rows")
        kernel_assert(Dp1 == D + 1, "S_aug must have D+1 columns")
        kernel_assert(Dp1 <= nl.tile_size.psum_fmax, "D+1 exceeds the PSUM free dimension")

        num_m_tiles = div_ceil(M, TILE)
        num_d_tiles = div_ceil(D, TILE)

        out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.shared_hbm)

        for m_tile in range(num_m_tiles):
            m_start = m_tile * TILE
            m_size = min(TILE, M - m_start)

            psum = nl.ndarray((m_size, Dp1), dtype=nl.float32, buffer=nl.psum)

            for d_tile in range(num_d_tiles):
                d_start = d_tile * TILE
                d_size = min(TILE, D - d_start)

                q_tile = nl.ndarray((m_size, d_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_tile,
                              src=phi_Q[m_start:m_start + m_size, d_start:d_start + d_size])

                # [m_size, d_size] -> [d_size, m_size] (partition/free axes swap).
                q_t_psum = nl.ndarray((d_size, m_size), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=q_t_psum, data=q_tile)

                # nc_matmul operands must live in SBUF.
                q_t = nl.ndarray((d_size, m_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=q_t, src=q_t_psum)

                s_tile = nl.ndarray((d_size, Dp1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=s_tile, src=S_aug[d_start:d_start + d_size, 0:Dp1])

                nisa.nc_matmul(dst=psum, stationary=q_t, moving=s_tile,
                               accumulate=(d_tile > 0))

            o_aug = nl.ndarray((m_size, Dp1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_aug, src=psum)

            # denom = o_aug[:, D] + eps, then multiply by its reciprocal.
            denom = nl.ndarray((m_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=denom, data=o_aug[0:m_size, D:Dp1],
                               op0=nl.add, operand0=eps)

            inv_denom = nl.ndarray((m_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_denom, data=denom)

            result_tile = nl.ndarray((m_size, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=result_tile, data=o_aug[0:m_size, 0:D],
                               op0=nl.multiply, operand0=inv_denom)

            nisa.dma_copy(dst=out[m_start:m_start + m_size, 0:D], src=result_tile)

        return out


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    M, D = Q.shape
    ones_col = torch.ones(M, 1, dtype=V.dtype, device=V.device)
    V_aug = torch.cat([V, ones_col], dim=1)

    phi_Q = phi_kernel(Q)
    phi_K = phi_kernel(K)

    S_aug = stageA_kernel(phi_K, V_aug)
    out = stageB_kernel(phi_Q, S_aug, float(eps))
    return out


def get_last_config() -> dict | None:
    return None
