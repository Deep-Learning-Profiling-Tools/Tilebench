from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None


@nki.jit
def transpose_kernel(a_input, tile):
    """2D matrix transpose: out[n, m] = in[m, n].

    Tiles the input into (<=128, <=128) blocks, transposes each block with
    ``nisa.nc_transpose`` (Tensor Engine, SBUF -> PSUM), copies the result back to
    SBUF and DMAs it into the transposed location of the output tensor.

    Boundary tiles are handled by clamping the tile extents with ``min()`` (no
    masking): a partial input tile of shape (m_sz, n_sz) yields an output tile of
    shape (n_sz, m_sz) that is written to out[n_start:n_end, m_start:m_end].

    Notes:
        ``nc_transpose`` is lowered to ``nc_matmul`` against an identity matrix, which
        does not accept 1-byte integer operands. For int8/uint8 inputs the tile is
        widened to bfloat16 before the transpose and narrowed back afterwards. bfloat16
        has 8 mantissa bits, so every 8-bit integer is represented exactly and the
        round trip (including the fp32 PSUM accumulation of value * 1.0) is lossless.
    """
    m, n = a_input.shape

    hbm_result_tile = nl.ndarray((n, m), dtype=a_input.dtype, buffer=nl.shared_hbm)

    # dtypes the Tensor Engine cannot transpose natively -> transpose in bfloat16.
    needs_widening = a_input.dtype in (nl.int8, nl.uint8)
    transpose_dtype = nl.bfloat16 if needs_widening else a_input.dtype

    num_m_blocks = (m + tile - 1) // tile
    num_n_blocks = (n + tile - 1) // tile

    for i in range(num_m_blocks):
        m_start = i * tile
        m_end = min(m_start + tile, m)
        m_sz = m_end - m_start

        for j in range(num_n_blocks):
            n_start = j * tile
            n_end = min(n_start + tile, n)
            n_sz = n_end - n_start

            a_tile = nl.ndarray((m_sz, n_sz), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a_input[m_start:m_end, n_start:n_end])

            if needs_widening:
                wide_tile = nl.ndarray((m_sz, n_sz), dtype=transpose_dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=wide_tile, src=a_tile)
                source_tile = wide_tile
            else:
                source_tile = a_tile

            # Partition/free axes swap: (m_sz, n_sz) -> (n_sz, m_sz)
            psum_tile = nl.ndarray((n_sz, m_sz), dtype=transpose_dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=psum_tile, data=source_tile)

            # PSUM cannot be DMA'd to HBM directly; stage through SBUF (this copy also
            # narrows bfloat16 back to the 8-bit integer output dtype when widened).
            result_tile = nl.ndarray((n_sz, m_sz), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_tile, src=psum_tile)

            nisa.dma_copy(dst=hbm_result_tile[n_start:n_end, m_start:m_end], src=result_tile)

    return hbm_result_tile


_DEFAULT_CONFIG = SimpleNamespace(tile=128)
_SEARCH_SPACE = [SimpleNamespace(tile=t) for t in (32, 64, 128)]
_tuner = NkiAutotuner(transpose_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x, cfg.tile),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return transpose_kernel(x, cfg.tile)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
