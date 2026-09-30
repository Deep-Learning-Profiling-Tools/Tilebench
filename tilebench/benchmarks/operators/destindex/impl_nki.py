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

if nki is not None:
    @nki.jit
    def scatter_rows_kernel(out, src, dest, block_size):
        T, D = src.shape
        k_tile_size = block_size
        num_k_tiles = (T + k_tile_size - 1) // k_tile_size

        for kt in range(num_k_tiles):
            k_valid = min(k_tile_size, T - kt * k_tile_size)

            idx_tile = nl.ndarray((k_valid, 1), dtype=dest.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx_tile, src=dest.ap(pattern=[[1, k_valid], [1, 1]], offset=kt * k_tile_size))

            src_tile = nl.ndarray((k_valid, D), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=src_tile, src=src.ap(pattern=[[D, k_valid], [1, D]], offset=kt * k_tile_size * D))

            nisa.dma_copy(
                dst=out.ap(pattern=[[D, k_valid], [1, D]], offset=0, vector_offset=idx_tile, indirect_dim=0),
                src=src_tile,
            )

        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=128)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (32, 64, 128)]
_tuner = NkiAutotuner(scatter_rows_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(kv_nope: torch.Tensor, kv_rope: torch.Tensor, dest_loc: torch.Tensor,
        o_nope: torch.Tensor, o_rope: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs):
    T, H_nope, D_nope = kv_nope.shape
    _, H_rope, D_rope = kv_rope.shape

    kv_nope_2d = kv_nope.reshape(T, H_nope * D_nope)
    kv_rope_2d = kv_rope.reshape(T, H_rope * D_rope)
    out_nope_2d = o_nope.reshape(T, H_nope * D_nope).clone()
    out_rope_2d = o_rope.reshape(T, H_rope * D_rope).clone()
    dest = dest_loc.to(torch.int32)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(kv_nope_2d.shape), str(kv_nope_2d.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (out_nope_2d, kv_nope_2d, dest, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    out_nope_2d = scatter_rows_kernel(out_nope_2d, kv_nope_2d, dest, cfg.block_size)
    out_rope_2d = scatter_rows_kernel(out_rope_2d, kv_rope_2d, dest, cfg.block_size)

    return out_nope_2d.reshape(T, H_nope, D_nope), out_rope_2d.reshape(T, H_rope, D_rope)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
