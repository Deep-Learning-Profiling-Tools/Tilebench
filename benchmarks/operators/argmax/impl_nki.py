import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
    free_tile_size = 16384
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def argmax_kernel(a_input):
        num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX
        num_free_blocks = (a_input.shape[1] + free_tile_size - 1) // free_tile_size

        hbm_result_tile = nl.ndarray((a_input.shape[0], 1), dtype=nl.uint32, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (a_input.shape[0] - offset)

            global_max = nl.full((PMAX, 1), fill_value=float('-inf'), dtype=nl.float32, buffer=nl.sbuf)
            global_idx = nl.zeros((PMAX, 1), dtype=nl.uint32, buffer=nl.sbuf)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
                mask_f = free_dim_index < (a_input.shape[1] - free_offset)
                mask = mask_p & mask_f

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask)

                top8 = nisa.max8(src=a_tile, mask=mask)
                idx8 = nisa.nc_find_index8(data=a_tile, vals=top8, mask=mask)

                chunk_max = nl.static_cast(top8[:, 0:1], nl.float32)
                chunk_idx = nl.add(nl.static_cast(idx8[:, 0:1], nl.uint32), free_offset, dtype=nl.uint32)

                better = nl.greater(chunk_max, global_max)
                new_idx = nl.where(better, chunk_idx, global_idx)
                new_max = nl.where(better, chunk_max, global_max)
                global_idx[...] = new_idx
                global_max[...] = new_max

            nl.store(hbm_result_tile[offset + partition_index, nl.arange(1)[None, :]], value=global_idx, mask=mask_p)

        return hbm_result_tile


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("argmax NKI: int8 not supported")

    result = argmax_kernel(x)
    return result.reshape(-1).to(torch.int64)

def get_last_config() -> dict | None:
    return None