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
    def gaussian_blur_kernel(padded_input, kernel, rows, cols, kernel_rows, kernel_cols):
        num_blocks = (rows + (PMAX - 1)) // PMAX

        hbm_result = nl.ndarray((rows, cols), dtype=padded_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            free_index = nl.arange(cols)[None, :]
            mask_p = partition_index < (rows - offset)

            acc = nl.zeros((PMAX, cols), dtype=nl.float32, buffer=nl.sbuf)
            col_index = nl.arange(cols + kernel_cols - 1)[None, :]
            for kh in range(kernel_rows):
                row_tile = nl.load(padded_input[offset + partition_index + kh, col_index],
                                    mask=mask_p, dtype=nl.float32)
                for kw in range(kernel_cols):
                    w_scalar = nl.load(kernel[kh:kh + 1, kw:kw + 1], dtype=nl.float32)
                    w_broadcast = nl.broadcast_to(w_scalar, shape=(PMAX, cols))
                    shifted = row_tile[:, kw:kw + cols]
                    acc[...] = nl.add(acc, nl.multiply(shifted, w_broadcast))

            result_tile = nl.add(acc, 0.0, dtype=padded_input.dtype)
            nl.store(hbm_result[offset + partition_index, free_index], value=result_tile, mask=mask_p)

        return hbm_result


def run(input: torch.Tensor, kernel: torch.Tensor, input_rows: int, input_cols: int,
        kernel_rows: int, kernel_cols: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    x = input.reshape(input_rows, input_cols)
    w = kernel.reshape(kernel_rows, kernel_cols)
    pr, pc = kernel_rows // 2, kernel_cols // 2
    padded = torch.nn.functional.pad(x, (pc, pc, pr, pr))
    result = gaussian_blur_kernel(padded, w, input_rows, input_cols, kernel_rows, kernel_cols)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None
