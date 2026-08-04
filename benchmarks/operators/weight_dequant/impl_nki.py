import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def weight_dequant_kernel(X, S, TILE_SIZE):
        M, N = X.shape

        num_row_blocks = (M + (PMAX - 1)) // PMAX
        num_col_blocks = (N + TILE_SIZE - 1) // TILE_SIZE

        hbm_result = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.hbm)

        for i in range(num_row_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (M - offset)
            s_row = offset // TILE_SIZE

            for j in range(num_col_blocks):
                free_offset = j * TILE_SIZE
                free_dim_index = nl.arange(TILE_SIZE)[None, :]
                mask_f = free_dim_index < (N - free_offset)
                mask = mask_p & mask_f
                s_col = free_offset // TILE_SIZE

                x_tile = nl.load(X[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)

                s_scalar = nl.load(S[s_row:s_row + 1, s_col:s_col + 1], dtype=nl.float32)
                s_broadcast = nl.broadcast_to(s_scalar, shape=(PMAX, TILE_SIZE))

                result_fp32 = nl.multiply(x_tile, s_broadcast)
                result_tile = nl.add(result_fp32, 0.0, dtype=X.dtype)

                nl.store(hbm_result[offset + partition_index, free_offset + free_dim_index], value=result_tile, mask=mask)

        return hbm_result


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if TILE_SIZE % PMAX != 0:
        raise NotImplementedError(
            f"weight_dequant NKI: TILE_SIZE ({TILE_SIZE}) must be a multiple of {PMAX}"
        )
    return weight_dequant_kernel(X, S, TILE_SIZE)


def get_last_config() -> dict | None:
    return None
