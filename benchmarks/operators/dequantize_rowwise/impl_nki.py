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
    def dequantize_rowwise_kernel(x_input, state_x_input):
        # x_input: (rows, cols) int8. state_x_input: (rows, 1) fp32.
        # out[r, c] = state_x[r] * x[r, c] / 127
        free_tile_size = 8192
        rows, cols = x_input.shape

        num_blocks = (rows + (PMAX - 1)) // PMAX
        num_free_blocks = (cols + free_tile_size - 1) // free_tile_size

        hbm_result_tile = nl.ndarray((rows, cols), dtype=nl.float16, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (rows - offset)

            state_tile = nl.load(state_x_input[offset + partition_index, nl.arange(1)[None, :]],
                                  dtype=nl.float32)
            scale_tile = nl.multiply(state_tile, 1.0 / 127.0)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
                mask_f = free_dim_index < (cols - free_offset)
                mask = mask_p & mask_f

                x_tile = nl.load(x_input[offset + partition_index, free_offset + free_dim_index],
                                  mask=mask, dtype=nl.float32)

                result_fp32 = nl.multiply(x_tile, scale_tile)
                result_tile = nl.add(result_fp32, 0.0, dtype=nl.float16)

                nl.store(hbm_result_tile[offset + partition_index, free_offset + free_dim_index],
                         value=result_tile, mask=mask)

        return hbm_result_tile


def run(x: torch.Tensor, state_x: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    rows = x.shape[0]
    state_x_2d = state_x.reshape(-1, 1)
    # state_x_2d is a tiny (rows, 1) allocation -- a tail block's masked
    # partition read can compute an address past the actual buffer end and
    # trip the compiler's static bounds check (unlike wide tensors, where the
    # same out-of-range read still lands inside the allocation). Pad rows to
    # a PMAX multiple so every access pattern the kernel emits stays in
    # bounds; the mask still discards the padding when producing the output.
    padded_rows = ((rows + PMAX - 1) // PMAX) * PMAX
    if padded_rows > rows:
        state_x_2d = torch.nn.functional.pad(state_x_2d, (0, 0, 0, padded_rows - rows))
    return dequantize_rowwise_kernel(x, state_x_2d)


def get_last_config() -> dict | None:
    return None
