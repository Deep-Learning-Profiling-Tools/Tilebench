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
    def conv3d_kernel(input_3d, kernel_flat, out_depth, out_rows, out_cols,
                       kernel_depth, kernel_rows, kernel_cols):
        num_blocks = (out_rows + (PMAX - 1)) // PMAX
        w_idx1 = nl.arange(1)[None, :, None]
        w_idx2 = nl.arange(1)[None, None, :]

        hbm_result = nl.ndarray((out_rows, out_depth, out_cols), dtype=input_3d.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            grid = nl.mgrid[0:PMAX, 0:out_depth, 0:out_cols]
            mask_p = grid.p < (out_rows - offset)

            acc = nl.zeros((PMAX, out_depth, out_cols), dtype=nl.float32, buffer=nl.sbuf)
            for kd in range(kernel_depth):
                depth_idx = grid.x + kd
                for kh in range(kernel_rows):
                    row_idx = offset + grid.p + kh
                    for kw in range(kernel_cols):
                        col_idx = grid.y + kw
                        tile = nl.ndarray((PMAX, out_depth, out_cols), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.dma_copy(dst=tile, src=input_3d[depth_idx, row_idx, col_idx], mask=mask_p)

                        tap = kd * kernel_rows * kernel_cols + kh * kernel_cols + kw
                        w_scalar = nl.load(kernel_flat[tap + nl.arange(1)[:, None, None], w_idx1, w_idx2], dtype=nl.float32)
                        w_broadcast = nl.broadcast_to(w_scalar, shape=(PMAX, out_depth, out_cols))
                        acc[...] = nl.add(acc, nl.multiply(tile, w_broadcast))

            result_tile = nl.add(acc, 0.0, dtype=input_3d.dtype)
            nl.store(hbm_result[offset + grid.p, grid.x, grid.y], value=result_tile, mask=mask_p)

        return hbm_result


def run(input: torch.Tensor, kernel: torch.Tensor, input_depth: int, input_rows: int,
        input_cols: int, kernel_depth: int, kernel_rows: int, kernel_cols: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x = input.reshape(input_depth, input_rows, input_cols)
    kernel_flat = kernel.reshape(-1, 1, 1)

    out_depth = input_depth - kernel_depth + 1
    out_rows = input_rows - kernel_rows + 1
    out_cols = input_cols - kernel_cols + 1

    result = conv3d_kernel(x, kernel_flat, out_depth, out_rows, out_cols,
                            kernel_depth, kernel_rows, kernel_cols)
    return result.permute(1, 0, 2).reshape(-1)


def get_last_config() -> dict | None:
    return None
