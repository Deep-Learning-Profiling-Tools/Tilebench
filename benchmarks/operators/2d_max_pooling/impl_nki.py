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
    def max_pool2d_kernel(padded_input, H_out, W_out, kernel_size, stride):
        NC, Hp, Wp = padded_input.shape
        num_blocks = (NC + (PMAX - 1)) // PMAX
        NEG_INF = -3.0e38

        hbm_result = nl.ndarray((NC, H_out, W_out), dtype=padded_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            grid = nl.mgrid[0:PMAX, 0:H_out, 0:W_out]
            mask_p = grid.p < (NC - offset)

            running_max = nl.full((PMAX, H_out, W_out), NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    row_idx = grid.x * stride + kh
                    col_idx = grid.y * stride + kw
                    
                    tile = nl.ndarray((PMAX, H_out, W_out), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=tile, src=padded_input[offset + grid.p, row_idx, col_idx], mask=mask_p)
                    running_max[...] = nl.maximum(running_max, tile)

            result_tile = nl.add(running_max, 0.0, dtype=padded_input.dtype)
            nl.store(hbm_result[offset + grid.p, grid.x, grid.y], value=result_tile, mask=mask_p)

        return hbm_result


def run(input: torch.Tensor, N: int, C: int, H: int, W: int,
        kernel_size: int, stride: int, padding: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:

    x = input.view(N * C, H, W)
    
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    neg_inf = float("-inf")
    padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), value=neg_inf)

    result = max_pool2d_kernel(padded, H_out, W_out, kernel_size, stride)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None
