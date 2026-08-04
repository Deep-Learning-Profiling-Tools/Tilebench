import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
    FREE_CAP = 2048
    FALLBACK_TILE = 2048
except ImportError:
    nki = None


if nki is not None:
    @nki.jit
    def kl_divergence_kernel(log_y_pred_input, y_true_input):
        n_rows = log_y_pred_input.shape[0]
        n_cols = log_y_pred_input.shape[1]

        num_blocks = (n_rows + (PMAX - 1)) // PMAX
        hbm_result_tile = nl.ndarray((n_rows, 1), dtype=nl.float32, buffer=nl.hbm)

        if n_cols <= FREE_CAP:
            for i in range(num_blocks):
                offset = i * PMAX
                partition_index = nl.arange(PMAX)[:, None]
                free_index = nl.arange(n_cols)[None, :]
                mask_p = partition_index < (n_rows - offset)
                mask = mask_p & (free_index < n_cols)

                log_p_tile = nl.load(log_y_pred_input[offset + partition_index, free_index], mask=mask, dtype=nl.float32)
                q_tile = nl.load(y_true_input[offset + partition_index, free_index], mask=mask, dtype=nl.float32)

                zero_tile = nl.zeros(q_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
                safe_log_q = nl.where(nl.greater(q_tile, 0.0), nl.log(q_tile), zero_tile)
                term = nl.multiply(q_tile, nl.subtract(safe_log_q, log_p_tile))
                kl_sum = nl.sum(term, axis=1, keepdims=True)

                nl.store(hbm_result_tile[offset + partition_index, nl.arange(1)[None, :]],
                         value=kl_sum, mask=mask_p)
            return hbm_result_tile

        num_free_blocks = (n_cols + FALLBACK_TILE - 1) // FALLBACK_TILE
        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (n_rows - offset)

            kl_sum = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(num_free_blocks):
                free_offset = j * FALLBACK_TILE
                free_dim_index = nl.arange(FALLBACK_TILE)[None, :]
                mask_f = free_dim_index < (n_cols - free_offset)
                mask = mask_p & mask_f

                log_p_tile = nl.load(log_y_pred_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)
                q_tile = nl.load(y_true_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)

                zero_tile = nl.zeros(q_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
                safe_log_q = nl.where(nl.greater(q_tile, 0.0), nl.log(q_tile), zero_tile)
                term = nl.multiply(q_tile, nl.subtract(safe_log_q, log_p_tile))
                term_safe = nl.where(mask, term, zero_tile)
                kl_sum[...] = nl.add(kl_sum, nl.sum(term_safe, axis=1, keepdims=True))

            nl.store(hbm_result_tile[offset + partition_index, nl.arange(1)[None, :]],
                     value=kl_sum, mask=mask_p)

        return hbm_result_tile


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
