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
    def transpose_kernel(a_input):
        m, n = a_input.shape
        hbm_result_tile = nl.ndarray((n, m), dtype=a_input.dtype, buffer=nl.hbm)

        num_m_blocks = (m + (PMAX - 1)) // PMAX
        num_n_blocks = (n + (PMAX - 1)) // PMAX

        for i in range(num_m_blocks):
            m_offset = i * PMAX

            for j in range(num_n_blocks):
                n_offset = j * PMAX

                partition_index = nl.arange(PMAX)[:, None]
                free_dim_index = nl.arange(PMAX)[None, :]

                load_mask = (partition_index < (m - m_offset)) & (free_dim_index < (n - n_offset))

                a_tile = nl.load(a_input[m_offset + partition_index, n_offset + free_dim_index], mask=load_mask)

                result_tile = nl.transpose(a_tile)

                store_mask = (partition_index < (n - n_offset)) & (free_dim_index < (m - m_offset))

                nl.store(hbm_result_tile[n_offset + partition_index, m_offset + free_dim_index], value=result_tile, mask=store_mask)

        return hbm_result_tile

def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    m, n = x.shape
    m_padded = ((m + PMAX - 1) // PMAX) * PMAX
    n_padded = ((n + PMAX - 1) // PMAX) * PMAX

    if m_padded > m or n_padded > n:
        x = torch.nn.functional.pad(x, (0, n_padded - n, 0, m_padded - m))

    result = transpose_kernel(x)
    return result[:n, :m]

def get_last_config() -> dict | None:
    return None