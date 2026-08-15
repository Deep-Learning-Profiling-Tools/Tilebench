import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

@nki.jit
def fused_activation_kernel(x_input, gate_input, bias_input):
    """Fused activation: out = silu(x * gate + bias), tiled over partition and free dims."""
    P, F = x_input.shape

    num_blocks = (P + PMAX - 1) // PMAX

    free_tile_size = 16384
    num_free_blocks = (F + free_tile_size - 1) // free_tile_size

    hbm_result_tile = nl.ndarray(x_input.shape, dtype=x_input.dtype, buffer=nl.shared_hbm)

    for i in range(num_blocks):
        p_start = i * PMAX
        p_end = min(p_start + PMAX, P)
        p_sz = p_end - p_start

        for j in range(num_free_blocks):
            f_start = j * free_tile_size
            f_end = min(f_start + free_tile_size, F)
            f_sz = f_end - f_start

            x_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x_input[p_start:p_end, f_start:f_end])

            gate_tile = nl.ndarray((p_sz, f_sz), dtype=gate_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gate_tile, src=gate_input[p_start:p_end, f_start:f_end])

            bias_tile = nl.ndarray((p_sz, f_sz), dtype=bias_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=bias_tile, src=bias_input[p_start:p_end, f_start:f_end])

            # x_gate = x * gate
            x_gate_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=x_gate_tile, data1=x_tile, data2=gate_tile,
                               op=nl.multiply)

            # x_gate_bias = x_gate + bias
            x_gate_bias_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=x_gate_bias_tile, data1=x_gate_tile, data2=bias_tile,
                               op=nl.add)

            # out = silu(x_gate_bias)
            result_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.activation(dst=result_tile, data=x_gate_bias_tile, op=nl.silu)

            nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, f_start:f_end], src=result_tile)

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
    gate_2d = gate.reshape(PMAX, free_dim)
    bias_2d = bias.reshape(PMAX, free_dim)
    result = fused_activation_kernel(x_2d, gate_2d, bias_2d)

    return result.reshape(-1)[:n]

def get_last_config() -> dict | None:
    return None
