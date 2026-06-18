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
    def fused_activation_kernel(x_input, gate_input, bias_input):
        num_blocks = (x_input.shape[0] + (PMAX - 1)) // PMAX

        hbm_result_tile = nl.ndarray(x_input.shape, dtype=x_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(x_input.shape[1])[None, :]
            
            mask = partition_index < (x_input.shape[0] - offset)
            
            x_tile = nl.load(x_input[offset + partition_index, free_dim_index], mask=mask)
            gate_tile = nl.load(gate_input[offset + partition_index, free_dim_index], mask=mask)
            bias_tile = nl.load(bias_input[offset + partition_index, free_dim_index], mask=mask)

            x_gate = nl.multiply(x_tile, gate_tile, mask=mask)

            x_gate_bias = nl.add(x_gate, bias_tile, mask=mask)

            result_tile = nl.silu(x_gate_bias, mask=mask)
            
            nl.store(hbm_result_tile[offset + partition_index, free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("fused_activation NKI: int8 not supported")
    
    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))
        gate = torch.nn.functional.pad(gate, (0, padded_size - n))
        bias = torch.nn.functional.pad(bias, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    x_gate_2d = gate.reshape(PMAX, free_dim)
    bias_2d = bias.reshape(PMAX, free_dim)
    result = fused_activation_kernel(x_2d, x_gate_2d, bias_2d)
    return result.reshape(-1)[:n]

def get_last_config() -> dict | None:
    return None