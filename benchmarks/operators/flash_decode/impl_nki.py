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
    def flash_decode_kernel(mid_o_t, mid_o_lse, valid_blocks):
        # mid_o_t: (head_dim, num_blocks) for one (batch, head) slice --
        # already transposed on the host so head_dim (<=128) sits on
        # partitions. mid_o_lse: (1, num_blocks). valid_blocks: (1, 1)
        # scalar (this batch's ceil(seqlen / block_seq), computed on host).
        head_dim, num_blocks = mid_o_t.shape
        NEG_INF = -3.0e38

        hbm_result = nl.ndarray((head_dim, 1), dtype=mid_o_t.dtype, buffer=nl.hbm)

        p_index = nl.arange(head_dim)[:, None]
        f_index = nl.arange(num_blocks)[None, :]

        grid = nl.mgrid[0:1, 0:num_blocks]
        block_iota = nl.add(nisa.iota(expr=grid.x, dtype=nl.int32), 0)
        block_iota_f = nl.add(block_iota, 0.0, dtype=nl.float32)

        valid_f = nl.load(valid_blocks[nl.arange(1)[:, None], nl.arange(1)[None, :]], dtype=nl.float32)
        valid_broadcast = nl.broadcast_to(valid_f, shape=(1, num_blocks))
        mask = nl.less(block_iota_f, valid_broadcast)

        lse_tile = nl.load(mid_o_lse[nl.arange(1)[:, None], f_index], dtype=nl.float32)
        zero_row = nl.zeros((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        neg_row = nl.full((1, num_blocks), NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
        masked_lse = nl.where(mask, lse_tile, neg_row)

        global_max = nl.max(masked_lse, axis=1, keepdims=True)
        global_max_b = nl.broadcast_to(global_max, shape=(1, num_blocks))
        weights = nl.exp(nl.subtract(masked_lse, global_max_b))
        weights = nl.where(mask, weights, zero_row)

        denom = nl.sum(weights, axis=1, keepdims=True)

        weights_bh = nl.broadcast_to(weights, shape=(head_dim, num_blocks))
        denom_bh = nl.broadcast_to(denom, shape=(head_dim, 1))

        mo_tile = nl.load(mid_o_t[p_index, f_index], dtype=nl.float32)
        numerator = nl.sum(nl.multiply(mo_tile, weights_bh), axis=1, keepdims=True)

        out_val = nl.divide(numerator, nl.add(denom_bh, 1e-10))
        out_cast = nl.add(out_val, 0.0, dtype=mid_o_t.dtype)

        nl.store(hbm_result[p_index, nl.arange(1)[None, :]], value=out_cast)
        return hbm_result


def run(mid_o: torch.Tensor, mid_o_lse: torch.Tensor, b_seqlen: torch.Tensor,
        block_seq, block_size: int = None, **kwargs) -> torch.Tensor:
    if isinstance(block_seq, torch.Tensor):
        block_seq = block_seq.item()

    batch, heads, num_blocks, head_dim = mid_o.shape
    valid_blocks_count = ((b_seqlen + block_seq - 1) // block_seq).to(torch.float32).reshape(batch, 1, 1)

    mid_o_t = mid_o.permute(0, 1, 3, 2).contiguous()  # (B, H, head_dim, num_blocks)
    lse_2d = mid_o_lse.reshape(batch, heads, 1, num_blocks)

    out = torch.empty(batch, heads, head_dim, dtype=mid_o.dtype, device=mid_o.device)
    for b in range(batch):
        for h in range(heads):
            res = flash_decode_kernel(mid_o_t[b, h], lse_2d[b, h], valid_blocks_count[b])
            out[b, h, :] = res.reshape(-1)
    return out


def get_last_config() -> dict | None:
    return None
