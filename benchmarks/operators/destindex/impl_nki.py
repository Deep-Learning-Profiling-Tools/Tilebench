import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def scatter_rows_kernel(out, src, dest):
        T, D = src.shape
        k_tile_size = PMAX
        num_k_tiles = (T + k_tile_size - 1) // k_tile_size

        for kt in nl.affine_range(num_k_tiles):
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

    out_nope_2d = scatter_rows_kernel(out_nope_2d, kv_nope_2d, dest)
    out_rope_2d = scatter_rows_kernel(out_rope_2d, kv_rope_2d, dest)

    return out_nope_2d.reshape(T, H_nope, D_nope), out_rope_2d.reshape(T, H_rope, D_rope)


def get_last_config() -> dict | None:
    return None
