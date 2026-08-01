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
    def rope_kernel(q, cos, sin):
        # q: (seq_len, head_dim) for one (batch, head) slice. cos/sin:
        # (seq_len, head_dim // 2), same shape as each half of q -- a plain
        # elementwise op, no broadcast needed.
        seq_len, head_dim = q.shape
        half = head_dim // 2
        num_blocks = (seq_len + (PMAX - 1)) // PMAX

        hbm_result = nl.ndarray((seq_len, head_dim), dtype=q.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            half_index = nl.arange(half)[None, :]
            mask_p = partition_index < (seq_len - offset)

            q1 = nl.load(q[offset + partition_index, half_index], mask=mask_p, dtype=nl.float32)
            q2 = nl.load(q[offset + partition_index, half + half_index], mask=mask_p, dtype=nl.float32)
            cos_tile = nl.load(cos[offset + partition_index, half_index], mask=mask_p, dtype=nl.float32)
            sin_tile = nl.load(sin[offset + partition_index, half_index], mask=mask_p, dtype=nl.float32)

            q1_out = nl.subtract(nl.multiply(q1, cos_tile), nl.multiply(q2, sin_tile))
            q2_out = nl.add(nl.multiply(q2, cos_tile), nl.multiply(q1, sin_tile))

            q1_out_c = nl.add(q1_out, 0.0, dtype=q.dtype)
            q2_out_c = nl.add(q2_out, 0.0, dtype=q.dtype)

            nl.store(hbm_result[offset + partition_index, half_index], value=q1_out_c, mask=mask_p)
            nl.store(hbm_result[offset + partition_index, half + half_index], value=q2_out_c, mask=mask_p)

        return hbm_result


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    batch, seq_len, n_heads, head_dim = q.shape
    out = torch.empty_like(q)
    for b in range(batch):
        for h in range(n_heads):
            out[b, :, h, :] = rope_kernel(q[b, :, h, :].contiguous(), cos, sin)
    return out


def get_last_config() -> dict | None:
    return None
