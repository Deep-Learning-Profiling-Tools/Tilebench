from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def rope_kernel(q, cos, sin, block_size):
        """Rotary position embedding for one (seq_len, head_dim) slice of q.

        cos/sin are (seq_len, head_dim // 2) -- the same shape as each half of q,
        so this is a plain elementwise op, no broadcast needed.

        Per PMAX-row block:
            q1_out = q1 * cos - q2 * sin
            q2_out = q2 * cos + q1 * sin
        """
        seq_len, head_dim = q.shape
        half = head_dim // 2
        num_blocks = (seq_len + (block_size - 1)) // block_size

        hbm_result = nl.ndarray((seq_len, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

        for i in range(num_blocks):
            p_start = i * block_size
            p_end = min(p_start + block_size, seq_len)
            p_sz = p_end - p_start

            # Tiles are sized to the clamped extent, so no masking is needed.
            q1_tile = nl.ndarray((p_sz, half), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q1_tile, src=q[p_start:p_end, 0:half])

            q2_tile = nl.ndarray((p_sz, half), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q2_tile, src=q[p_start:p_end, half:head_dim])

            cos_tile = nl.ndarray((p_sz, half), dtype=cos.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=cos_tile, src=cos[p_start:p_end, 0:half])

            sin_tile = nl.ndarray((p_sz, half), dtype=sin.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sin_tile, src=sin[p_start:p_end, 0:half])

            # Products are accumulated in fp32 regardless of the input dtype.
            q1_cos = nl.ndarray((p_sz, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q1_cos, data1=q1_tile, data2=cos_tile, op=nl.multiply)

            q2_sin = nl.ndarray((p_sz, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q2_sin, data1=q2_tile, data2=sin_tile, op=nl.multiply)

            q2_cos = nl.ndarray((p_sz, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q2_cos, data1=q2_tile, data2=cos_tile, op=nl.multiply)

            q1_sin = nl.ndarray((p_sz, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q1_sin, data1=q1_tile, data2=sin_tile, op=nl.multiply)

            q1_out = nl.ndarray((p_sz, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q1_out, data1=q1_cos, data2=q2_sin, op=nl.subtract)

            q2_out = nl.ndarray((p_sz, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q2_out, data1=q2_cos, data2=q1_sin, op=nl.add)

            # tensor_copy performs the cast back to q's dtype.
            q1_out_c = nl.ndarray((p_sz, half), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q1_out_c, src=q1_out)

            q2_out_c = nl.ndarray((p_sz, half), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q2_out_c, src=q2_out)

            nisa.dma_copy(dst=hbm_result[p_start:p_end, 0:half], src=q1_out_c)
            nisa.dma_copy(dst=hbm_result[p_start:p_end, half:head_dim], src=q2_out_c)

        return hbm_result


_DEFAULT_CONFIG = SimpleNamespace(block_size=128)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (32, 64, 128)]
_tuner = NkiAutotuner(rope_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    batch, seq_len, n_heads, head_dim = q.shape
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(q.shape), str(q.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (q[0, :, 0, :].contiguous(), cos, sin, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    out = torch.empty_like(q)
    for b in range(batch):
        for h in range(n_heads):
            out[b, :, h, :] = rope_kernel(q[b, :, h, :].contiguous(), cos, sin, cfg.block_size)
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
