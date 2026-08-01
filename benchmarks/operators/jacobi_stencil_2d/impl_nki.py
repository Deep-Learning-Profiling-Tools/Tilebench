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
    def jacobi_kernel(padded_input, rows, cols):
        # padded_input: (rows+2, cols+2), padded_input[r+1, c+1] == input[r, c].
        # Row/col shifts read straight from HBM with the padded offset instead
        # of clamping indices, so no access ever goes negative; boundary
        # output cells select the (unshifted) center value, which equals the
        # original input exactly thanks to the padding.
        num_blocks = (rows + (PMAX - 1)) // PMAX

        hbm_result = nl.ndarray((rows, cols), dtype=padded_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            free_index = nl.arange(cols)[None, :]
            mask_p = partition_index < (rows - offset)

            col_index = nl.arange(cols + 2)[None, :]
            up_tile = nl.load(padded_input[offset + partition_index, col_index], mask=mask_p, dtype=nl.float32)
            center_tile = nl.load(padded_input[offset + partition_index + 1, col_index], mask=mask_p, dtype=nl.float32)
            down_tile = nl.load(padded_input[offset + partition_index + 2, col_index], mask=mask_p, dtype=nl.float32)

            up = up_tile[:, 1:cols + 1]
            down = down_tile[:, 1:cols + 1]
            center = center_tile[:, 1:cols + 1]
            left = center_tile[:, 0:cols]
            right = center_tile[:, 2:cols + 2]

            interior = nl.multiply(nl.add(nl.add(up, down), nl.add(left, right)), 0.25)

            is_row_boundary = ((offset + partition_index) == 0) | ((offset + partition_index) == (rows - 1))
            is_col_boundary_narrow = (free_index == 0) | (free_index == (cols - 1))
            is_col_boundary = nl.broadcast_to(is_col_boundary_narrow, shape=(PMAX, cols))
            is_boundary = is_row_boundary | is_col_boundary

            result_fp32 = nl.where(is_boundary, center, interior)
            result_tile = nl.add(result_fp32, 0.0, dtype=padded_input.dtype)

            nl.store(hbm_result[offset + partition_index, free_index], value=result_tile, mask=mask_p)

        return hbm_result


def run(input: torch.Tensor, rows: int, cols: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    padded = torch.nn.functional.pad(input, (1, 1, 1, 1))
    return jacobi_kernel(padded, rows, cols)


def get_last_config() -> dict | None:
    return None
