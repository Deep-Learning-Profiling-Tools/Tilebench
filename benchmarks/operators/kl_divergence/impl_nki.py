import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# Hardware constant (NeuronCore-v2/v3): number of SBUF partitions.
PMAX = 128

# Rows whose full free extent fits in a single SBUF tile take the one-pass path;
# wider rows are streamed in ``FALLBACK_TILE``-wide column blocks.
FREE_CAP = 2048
FALLBACK_TILE = 2048


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    def _kl_row_terms(log_p_tile, q_tile, row_size, col_size, part_sum):
        """Row-sum of ``q * (log(q) - log_p)`` for one [row_size, col_size] block.

        ``log(0)`` is -inf, and ``0 * -inf`` is NaN, so the logarithm is masked to
        0.0 wherever ``q == 0`` before the multiply. ``nisa.select_reduce`` is the
        NKI 0.4.0 replacement for ``np.where``; its ``on_false`` operand must be a
        scalar (or a ``[P, 1]`` column), which the constant 0.0 satisfies directly.
        The predicate must be integer-typed, hence the ``uint8`` comparison result.
        """
        is_pos = nl.ndarray((row_size, col_size), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=is_pos, data=q_tile, op0=nl.greater, operand0=0.0)

        log_q = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=log_q, op=nl.log, data=q_tile)

        safe_log_q = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.select_reduce(dst=safe_log_q, predicate=is_pos, on_true=log_q, on_false=0.0)

        term = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=term, data1=safe_log_q, data2=log_p_tile, op=nl.subtract)
        nisa.tensor_tensor(dst=term, data1=q_tile, data2=term, op=nl.multiply)

        nisa.tensor_reduce(dst=part_sum, op=nl.add, data=term, axis=(1,))

    @nki.jit
    def kl_divergence_kernel(log_y_pred_input, y_true_input):
        """Row-wise KL divergence ``sum_j q[j] * (log(q[j]) - log_p[j])``.

        Mirrors ``(y_true * (torch.log(y_true) - log_y_pred)).sum(dim=-1)``.

        Args:
            log_y_pred_input: [rows, n_cols] log-probabilities in HBM.
            y_true_input: [rows, n_cols] probabilities in HBM.

        Returns:
            [rows, 1] fp32 tensor in HBM with the per-row divergence.

        Notes:
            * Row blocks (and, on the wide path, column blocks) are clamped to the
              tensor extents with ``min(...)``, so every tile is allocated at its
              real extent and no load/store masking is required.
            * ``n_cols <= FREE_CAP``: one pass, the whole row lives in SBUF and its
              row-sum is stored directly.
            * ``n_cols > FREE_CAP``: the row is streamed in ``FALLBACK_TILE``-wide
              column blocks whose partial row-sums are accumulated into an fp32
              ``[P, 1]`` accumulator.
        """
        kernel_assert(len(log_y_pred_input.shape) == 2, "input must be 2D [rows, n_cols]")
        kernel_assert(tuple(log_y_pred_input.shape) == tuple(y_true_input.shape),
                      "log_y_pred and y_true must have the same shape")

        n_rows, n_cols = log_y_pred_input.shape
        kernel_assert(min(n_cols, FALLBACK_TILE) <= nl.tile_size.sbuf_fmax,
                      "column block exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((n_rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        n_row_tiles = div_ceil(n_rows, PMAX)

        if n_cols <= FREE_CAP:
            for row_tile in range(n_row_tiles):
                row_start = row_tile * PMAX
                row_size = min(PMAX, n_rows - row_start)
                row_end = row_start + row_size

                # DMA cannot convert dtypes: load in the input dtype and let the
                # Vector/Scalar engines widen to fp32 on their way out.
                log_p_tile = nl.ndarray((row_size, n_cols), dtype=log_y_pred_input.dtype,
                                        buffer=nl.sbuf)
                q_tile = nl.ndarray((row_size, n_cols), dtype=y_true_input.dtype,
                                    buffer=nl.sbuf)
                nisa.dma_copy(dst=log_p_tile, src=log_y_pred_input[row_start:row_end, 0:n_cols])
                nisa.dma_copy(dst=q_tile, src=y_true_input[row_start:row_end, 0:n_cols])

                kl_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                _kl_row_terms(log_p_tile, q_tile, row_size, n_cols, kl_sum)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, 0:1], src=kl_sum)

            return out_hbm

        n_col_tiles = div_ceil(n_cols, FALLBACK_TILE)
        for row_tile in range(n_row_tiles):
            row_start = row_tile * PMAX
            row_size = min(PMAX, n_rows - row_start)
            row_end = row_start + row_size

            kl_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=kl_sum, value=0.0)

            for col_tile in range(n_col_tiles):
                col_start = col_tile * FALLBACK_TILE
                col_size = min(FALLBACK_TILE, n_cols - col_start)
                col_end = col_start + col_size

                log_p_tile = nl.ndarray((row_size, col_size), dtype=log_y_pred_input.dtype,
                                        buffer=nl.sbuf)
                q_tile = nl.ndarray((row_size, col_size), dtype=y_true_input.dtype,
                                    buffer=nl.sbuf)
                nisa.dma_copy(dst=log_p_tile,
                              src=log_y_pred_input[row_start:row_end, col_start:col_end])
                nisa.dma_copy(dst=q_tile,
                              src=y_true_input[row_start:row_end, col_start:col_end])

                part_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                _kl_row_terms(log_p_tile, q_tile, row_size, col_size, part_sum)

                nisa.tensor_tensor(dst=kl_sum, data1=kl_sum, data2=part_sum, op=nl.add)

            nisa.dma_copy(dst=out_hbm[row_start:row_end, 0:1], src=kl_sum)

        return out_hbm


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if log_y_pred.dtype == torch.int8 or y_true.dtype == torch.int8:
        raise NotImplementedError("kl_divergence NKI: int8 not supported")

    orig_rows_shape = log_y_pred.shape[:-1]
    cols = log_y_pred.shape[-1]
    log_p_2d = log_y_pred.reshape(-1, cols)
    q_2d = y_true.reshape(-1, cols)
    if not log_p_2d.is_contiguous():
        log_p_2d = log_p_2d.contiguous()
    if not q_2d.is_contiguous():
        q_2d = q_2d.contiguous()

    result = kl_divergence_kernel(log_p_2d, q_2d)
    return result.reshape(orig_rows_shape)


def get_last_config() -> dict | None:
    return None
