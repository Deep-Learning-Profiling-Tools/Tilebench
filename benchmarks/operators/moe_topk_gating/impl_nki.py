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
    def moe_topk_kernel(logits):
        M, E = logits.shape
        num_blocks = (M + (PMAX - 1)) // PMAX

        hbm_weights = nl.ndarray((M, 2), dtype=logits.dtype, buffer=nl.hbm)
        hbm_idx = nl.ndarray((M, 2), dtype=nl.int32, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            free_index = nl.arange(E)[None, :]
            mask_p = partition_index < (M - offset)

            a_tile = nl.load(logits[offset + partition_index, free_index], mask=mask_p, dtype=nl.float32)

            top8 = nisa.max8(src=a_tile, mask=mask_p)
            idx8 = nisa.nc_find_index8(data=a_tile, vals=top8, mask=mask_p)

            max1 = top8[:, 0:1]
            max2 = top8[:, 1:2]
            idx1 = nl.static_cast(idx8[:, 0:1], nl.int32)
            idx2 = nl.static_cast(idx8[:, 1:2], nl.int32)

            e2nd = nl.exp(nl.subtract(max2, max1))
            e1st = nl.exp(nl.subtract(max1, max1))
            denom = nl.add(e2nd, e1st)
            w0 = nl.divide(e2nd, denom)
            w1 = nl.divide(e1st, denom)

            w_tile = nl.ndarray((PMAX, 2), dtype=logits.dtype, buffer=nl.sbuf)
            w_tile[:, 0:1] = nl.add(w1, 0.0, dtype=logits.dtype)
            w_tile[:, 1:2] = nl.add(w0, 0.0, dtype=logits.dtype)

            idx_tile = nl.ndarray((PMAX, 2), dtype=nl.int32, buffer=nl.sbuf)
            idx_tile[:, 0:1] = idx1
            idx_tile[:, 1:2] = idx2

            out_free = nl.arange(2)[None, :]
            nl.store(hbm_weights[offset + partition_index, out_free], value=w_tile, mask=mask_p)
            nl.store(hbm_idx[offset + partition_index, out_free], value=idx_tile, mask=mask_p)

        return hbm_weights, hbm_idx


def run(logits: torch.Tensor, M: int, E: int, k: int, block_size: int = 1024,
        autotune: bool = False, **kwargs):
    if k != 2:
        raise NotImplementedError("moe_topk_gating NKI: only k=2 is supported")
    weights, idx = moe_topk_kernel(logits)
    return weights, idx.to(torch.int32)


def get_last_config() -> dict | None:
    return None
