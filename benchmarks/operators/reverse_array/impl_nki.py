from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

# Free-dimension tile size, in elements. Two SBUF buffers of this width are live
# at once, so 16384 costs 64KB/partition at fp16 and 128KB/partition at fp32,
# both within the 192KB per-partition SBUF budget.
FREE_TILE_SIZE = 16384


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


if nki is not None:
    @nki.jit
    def reverse_kernel(a_input, block_size):
        total_rows, total_cols = a_input.shape

        free_tile_size = min(block_size, total_cols)
        num_blocks = div_ceil(total_rows, PMAX)
        num_free_blocks = div_ceil(total_cols, free_tile_size)

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype,
                                     buffer=nl.shared_hbm)

        for i in range(num_blocks):
            p_offset = i * PMAX
            p_size = min(PMAX, total_rows - p_offset)

            for j in range(num_free_blocks):
                f_offset = j * free_tile_size
                f_size = min(free_tile_size, total_cols - f_offset)

                # Forward, fully contiguous load: one DMA for the whole tile.
                a_tile = nl.ndarray((p_size, f_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=a_tile,
                    src=a_input[p_offset:p_offset + p_size, f_offset:f_offset + f_size],
                )

                # Free-dim reversal on-chip: static step of -1 along the free
                # axis. The partition step stays equal to the free-dim element
                # count, as SBUF access patterns require.
                rev_tile = nl.ndarray((p_size, f_size), dtype=a_input.dtype, buffer=nl.sbuf)

                nisa.tensor_copy(
                    dst=rev_tile,
                    src=a_tile.ap(pattern=[[f_size, p_size], [-1, f_size]], offset=f_size - 1),
                )

                dst_c0 = total_cols - f_offset - f_size
                for p in range(p_size):
                    dst_row = total_rows - 1 - p_offset - p
                    nisa.dma_copy(
                        dst=hbm_result_tile[dst_row:dst_row + 1,
                                            dst_c0:dst_c0 + f_size],
                        src=rev_tile[p:p + 1, 0:f_size],
                    )

        return hbm_result_tile


_DEFAULT_CONFIG = SimpleNamespace(block_size=16384)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (2048, 4096, 8192, 16384)]
_tuner = NkiAutotuner(reverse_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    free_dim = div_ceil(n, PMAX)
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x_2d.shape), str(x_2d.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_2d, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = reverse_kernel(x_2d, cfg.block_size)

    pad_count = padded_size - n

    return result.reshape(-1)[pad_count:]


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
