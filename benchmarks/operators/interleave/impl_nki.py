import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

@nki.jit
def interleave_kernel(a_input, b_input):
    """Interleave two [P, F] tensors into a [P, 2*F] tensor.

    Output layout: out[p, 2*c] = a[p, c], out[p, 2*c + 1] = b[p, c].
    Flattened row-major this yields a[0], b[0], a[1], b[1], ... which matches
    ``out[0::2] = A; out[1::2] = B`` on the flattened inputs.

    The interleaving is done inside SBUF with strided destination slices
    (``out_tile[:, 0::2]`` / ``out_tile[:, 1::2]``) so that both the HBM loads
    and the HBM store stay fully contiguous DMAs.
    """
    P, F = a_input.shape

    num_blocks = (P + PMAX - 1) // PMAX

    # Per free block SBUF footprint is (a_tile + b_tile + out_tile) =
    # 4 * free_tile_size elements per partition, so keep this well under the
    # per-partition SBUF budget for fp32.
    free_tile_size = 4096
    num_free_blocks = (F + free_tile_size - 1) // free_tile_size

    hbm_result_tile = nl.ndarray((P, 2 * F), dtype=a_input.dtype, buffer=nl.shared_hbm)

    for i in range(num_blocks):
        p_start = i * PMAX
        p_end = min(p_start + PMAX, P)
        p_sz = p_end - p_start

        for j in range(num_free_blocks):
            f_start = j * free_tile_size
            f_end = min(f_start + free_tile_size, F)
            f_sz = f_end - f_start

            a_tile = nl.ndarray((p_sz, f_sz), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a_input[p_start:p_end, f_start:f_end])

            b_tile = nl.ndarray((p_sz, f_sz), dtype=b_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b_tile, src=b_input[p_start:p_end, f_start:f_end])

            out_tile = nl.ndarray((p_sz, 2 * f_sz), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile[0:p_sz, 0:2 * f_sz:2], src=a_tile)
            nisa.tensor_copy(dst=out_tile[0:p_sz, 1:2 * f_sz:2], src=b_tile)

            nisa.dma_copy(
                dst=hbm_result_tile[p_start:p_end, 2 * f_start:2 * f_end],
                src=out_tile,
            )

    return hbm_result_tile

def run(a: torch.Tensor, b: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        a = torch.nn.functional.pad(a, (0, padded_size - n))
        b = torch.nn.functional.pad(b, (0, padded_size - n))

    a_2d = a.reshape(PMAX, free_dim)
    b_2d = b.reshape(PMAX, free_dim)
    result = interleave_kernel(a_2d, b_2d)
    return result.reshape(-1)[:2 * n]

def get_last_config() -> dict | None:
    return None
