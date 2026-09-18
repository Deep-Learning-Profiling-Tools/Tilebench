import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


@nki.jit
def dequantize_rowwise_kernel(x_input, state_x_input):
    """Row-wise dequantization: ``out[r, c] = x[r, c] * state_x[r] / 127`` in fp16.

    Args:
        x_input: [rows, cols] quantized values in HBM (typically int8).
        state_x_input: [padded_rows, 1] per-row absmax scales in HBM (fp32).
            ``padded_rows`` is a multiple of ``PMAX`` (padded host-side in ``run``).

    Returns:
        [rows, cols] fp16 tensor in HBM. The output dtype is always fp16,
        independent of the input dtype (bitsandbytes rowwise-dequant semantics).

    Notes:
        * Tiled in two dimensions: ``PMAX`` rows by ``free_tile_size`` columns.
          Boundary tiles are clamped to the surviving extent rather than masked,
          so no out-of-range element is ever loaded, multiplied or stored.
        * The scale is naturally one value per partition, so the [P, 1] scale
          tile can be handed straight to ``nisa.tensor_scalar`` as ``operand0``,
          which broadcasts it along the free axis in hardware -- no cross-partition
          broadcast is needed.
        * ``scale_tile`` is computed once per row block and reused across every
          column block of that row block.
    """
    kernel_assert(len(x_input.shape) == 2, "x must be 2D [rows, cols]")
    kernel_assert(len(state_x_input.shape) == 2, "state_x must be 2D [rows, 1]")

    rows, cols = x_input.shape
    kernel_assert(state_x_input.shape[0] >= rows, "state_x has fewer rows than x")

    free_tile_size = 8192

    num_blocks = div_ceil(rows, PMAX)
    num_free_blocks = div_ceil(cols, free_tile_size)

    hbm_result_tile = nl.ndarray((rows, cols), dtype=nl.float16, buffer=nl.shared_hbm)

    for i in range(num_blocks):
        p_start = i * PMAX
        p_end = min(p_start + PMAX, rows)
        p_sz = p_end - p_start

        state_tile = nl.ndarray((p_sz, 1), dtype=state_x_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_tile, src=state_x_input[p_start:p_end, 0:1])

        # scale = state_x / 127 (fp32: tensor_scalar operand0 must be fp32)
        scale_tile = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scale_tile, data=state_tile, op0=nl.multiply,
                           operand0=1.0 / 127.0)

        for j in range(num_free_blocks):
            f_start = j * free_tile_size
            f_end = min(f_start + free_tile_size, cols)
            f_sz = f_end - f_start

            x_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x_input[p_start:p_end, f_start:f_end])

            # out = x * scale, broadcast across the free axis, cast to fp16
            result_tile = nl.ndarray((p_sz, f_sz), dtype=nl.float16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=result_tile, data=x_tile, op0=nl.multiply,
                               operand0=scale_tile)

            nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, f_start:f_end],
                          src=result_tile)

    return hbm_result_tile


def run(x: torch.Tensor, state_x: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    rows = x.shape[0]
    state_x_2d = state_x.reshape(-1, 1)
    padded_rows = div_ceil(rows, PMAX) * PMAX
    if padded_rows > rows:
        state_x_2d = torch.nn.functional.pad(state_x_2d, (0, 0, 0, padded_rows - rows))
    return dequantize_rowwise_kernel(x, state_x_2d)


def get_last_config() -> dict | None:
    return None
