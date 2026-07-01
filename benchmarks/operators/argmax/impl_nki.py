import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
    FREE_TILE = 16384
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def argmax_kernel(a_input):
        m, n = a_input.shape
        num_blocks = (m + PMAX - 1) // PMAX
        num_free_blocks = (n + FREE_TILE - 1) // FREE_TILE

        hbm_result = nl.ndarray((m, 1), dtype=nl.uint32, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            p_mask = partition_index < (m - offset)

            best_val = nl.full((PMAX, 8), fill_value=float('-inf'), dtype=nl.float32, buffer=nl.sbuf)
            best_idx = nl.zeros((PMAX, 1), dtype=nl.uint32, buffer=nl.sbuf)

            for j in range(num_free_blocks):
                free_offset = j * FREE_TILE
                free_dim_index = nl.arange(FREE_TILE)[None, :]
                f_mask = free_dim_index < (n - free_offset)
                mask = p_mask & f_mask

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index],
                                mask=mask)

                tile_top8_vals = nisa.max8(src=a_tile, mask=mask)
                tile_top8_idx = nisa.nc_find_index8(data=a_tile, vals=tile_top8_vals, mask=mask)

                # cast to uint32 and add global offset
                tile_max_val = tile_top8_vals[:, 0:1]
                tile_max_idx = nl.static_cast(tile_top8_idx[:, 0:1], nl.uint32)
                tile_max_idx = nl.add(tile_max_idx, free_offset)

                better = nl.greater(tile_max_val, best_val[:, 0:1])
                best_val[...] = nl.where(better, tile_top8_vals, best_val)
                best_idx[...] = nl.where(better, nl.static_cast(tile_max_idx, nl.uint32), best_idx)

            nl.store(hbm_result[offset + partition_index, nl.arange(1)[None, :]],
                    value=best_idx, mask=p_mask)

        return hbm_result

def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if dim != 1:
        raise NotImplementedError("NKI argmax only supports dim=1")
    if x.dtype == torch.int8:
        raise NotImplementedError("NKI argmax: int8 not supported")
    result = argmax_kernel(x)
    return result.reshape(-1).to(torch.int64)

def get_last_config() -> dict | None:
    return None