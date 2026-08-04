import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

FREE_CAP = 8192
FALLBACK_TILE = 2048

def _rstd_newton(val):
    half_val = nl.multiply(val, 0.5)
    y = nl.rsqrt(val)
    y = nl.multiply(y, nl.subtract(1.5, nl.multiply(half_val, nl.multiply(y, y))))
    y = nl.multiply(y, nl.subtract(1.5, nl.multiply(half_val, nl.multiply(y, y))))
    return y

if nki is not None:
    @nki.jit
    def rmsnorm_kernel(a_input, weight_input, eps):
        n_rows = a_input.shape[0]
        n_cols = a_input.shape[1]

        num_blocks = (n_rows + (PMAX - 1)) // PMAX
        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        _cap = 2048 if a_input.dtype == nl.float32 else FREE_CAP
        if n_cols <= _cap:
            for i in range(num_blocks):
                offset = i * PMAX
                partition_index = nl.arange(PMAX)[:, None]
                free_index = nl.arange(n_cols)[None, :]
                mask = (partition_index < (n_rows - offset)) & (free_index < n_cols)

                a_tile = nl.load(a_input[offset + partition_index, free_index], mask=mask, dtype=nl.float32)
                sq_tile = nl.multiply(a_tile, a_tile)
                sum_sq = nl.sum(sq_tile, axis=1, keepdims=True)

                mean_sq = nl.divide(sum_sq, float(n_cols))
                rstd = _rstd_newton(nl.add(mean_sq, eps, dtype=nl.float32))

                w_tile = nl.load(weight_input[nl.arange(1)[:, None], free_index], mask=(free_index < n_cols), dtype=nl.float32)
                w_broadcast = nl.broadcast_to(w_tile, shape=(PMAX, n_cols))

                normalized = nl.multiply(a_tile, rstd)
                result_tile = nl.multiply(normalized, w_broadcast, dtype=a_input.dtype)
                nl.store(hbm_result_tile[offset + partition_index, free_index], value=result_tile, mask=mask)
            return hbm_result_tile

        num_free_blocks = (n_cols + FALLBACK_TILE - 1) // FALLBACK_TILE
        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (n_rows - offset)

            sum_sq = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(num_free_blocks):
                free_offset = j * FALLBACK_TILE
                free_dim_index = nl.arange(FALLBACK_TILE)[None, :]
                mask_f = free_dim_index < (n_cols - free_offset)
                mask = mask_p & mask_f
                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)
                sq_tile = nl.multiply(a_tile, a_tile)
                zero_tile = nl.zeros(sq_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
                sq_tile_safe = nl.where(mask, sq_tile, zero_tile)
                sum_sq[...] = nl.add(sum_sq, nl.sum(sq_tile_safe, axis=1, keepdims=True))

            mean_sq = nl.divide(sum_sq, float(n_cols))
            rstd = _rstd_newton(nl.add(mean_sq, eps, dtype=nl.float32))

            for j in range(num_free_blocks):
                free_offset = j * FALLBACK_TILE
                free_dim_index = nl.arange(FALLBACK_TILE)[None, :]
                mask_f = free_dim_index < (n_cols - free_offset)
                mask = mask_p & mask_f
                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)
                w_tile = nl.load(weight_input[nl.arange(1)[:, None], free_offset + free_dim_index], mask=mask_f, dtype=nl.float32)
                w_broadcast = nl.broadcast_to(w_tile, shape=(PMAX, FALLBACK_TILE))
                normalized = nl.multiply(a_tile, rstd)
                result_tile = nl.multiply(normalized, w_broadcast, dtype=a_input.dtype)
                nl.store(hbm_result_tile[offset + partition_index, free_offset + free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile


def run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("rmsnorm NKI: int8 not supported")

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    weight_2d = weight.reshape(1, -1)
    out_2d = rmsnorm_kernel(x_2d, weight_2d, eps)
    return out_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    return None
