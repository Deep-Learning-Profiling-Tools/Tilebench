import functools
import os
import re
import subprocess
from types import SimpleNamespace

import torch

from tilebench.core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

MAX8_MIN_ELEMS = 8
MAX8_MAX_ELEMS = 16384

@functools.lru_cache(maxsize=1)
def _lnc_degree() -> int:
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=10).stdout
        lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", out)
        if lnc:
            return int(lnc.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return 1


if nki is not None:
    @nki.jit
    def moe_topk_kernel(logits, block_size):
        M, E = logits.shape

        num_par_blocks = (M + block_size - 1) // block_size

        tile_sz = max(E, MAX8_MIN_ELEMS)

        hbm_weights = nl.ndarray((M, 2), dtype=logits.dtype, buffer=nl.shared_hbm)
        hbm_idx = nl.ndarray((M, 2), dtype=nl.int32, buffer=nl.shared_hbm)

        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (num_par_blocks + n_prog - 1) // n_prog
        b_lo = min(num_par_blocks, pid * per_core)
        b_hi = min(num_par_blocks, b_lo + per_core)

        for i in range(b_lo, b_hi):
            p_start = i * block_size
            p_end = min(p_start + block_size, M)
            p_sz = p_end - p_start

            a_tile = nl.ndarray((p_sz, tile_sz), dtype=logits.dtype, buffer=nl.sbuf)
            if tile_sz != E:
                nisa.memset(dst=a_tile, value=float('-inf'))
            nisa.dma_copy(dst=a_tile[0:p_sz, 0:E], src=logits[p_start:p_end, 0:E])

            top8 = nl.ndarray((p_sz, 8), dtype=logits.dtype, buffer=nl.sbuf)
            nisa.max8(dst=top8, src=a_tile)

            idx8 = nl.ndarray((p_sz, 8), dtype=nl.uint16, buffer=nl.sbuf)
            nisa.nc_find_index8(dst=idx8, data=a_tile, vals=top8)

            max1 = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=max1, src=top8[0:p_sz, 0:1])

            max2 = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=max2, src=top8[0:p_sz, 1:2])

            d2nd = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=d2nd, data1=max2, data2=max1, op=nl.subtract)
            e2nd = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e2nd, op=nl.exp, data=d2nd)

            d1st = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=d1st, data1=max1, data2=max1, op=nl.subtract)
            e1st = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e1st, op=nl.exp, data=d1st)

            denom = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=denom, data1=e2nd, data2=e1st, op=nl.add)

            inv_denom = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_denom, data=denom)

            w0 = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=w0, data1=e2nd, data2=inv_denom, op=nl.multiply)

            w1 = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=w1, data1=e1st, data2=inv_denom, op=nl.multiply)

            w_tile = nl.ndarray((p_sz, 2), dtype=logits.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_tile[0:p_sz, 0:1], src=w1)
            nisa.tensor_copy(dst=w_tile[0:p_sz, 1:2], src=w0)

            idx_tile = nl.ndarray((p_sz, 2), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=idx_tile[0:p_sz, 0:1], src=idx8[0:p_sz, 0:1])
            nisa.tensor_copy(dst=idx_tile[0:p_sz, 1:2], src=idx8[0:p_sz, 1:2])

            nisa.dma_copy(dst=hbm_weights[p_start:p_end, 0:2], src=w_tile)
            nisa.dma_copy(dst=hbm_idx[p_start:p_end, 0:2], src=idx_tile)

        return hbm_weights, hbm_idx


_DEFAULT_CONFIG = SimpleNamespace(block_size=128)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (32, 64, 128)]
_LNC = _lnc_degree()
_kernel = moe_topk_kernel[_LNC] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(logits: torch.Tensor, M: int, E: int, k: int, block_size: int = 1024,
        autotune: bool = False, **kwargs):
    if k != 2:
        raise NotImplementedError("moe_topk_gating NKI: only k=2 is supported")
    if E > MAX8_MAX_ELEMS:
        raise NotImplementedError(
            f"moe_topk_gating NKI: E={E} exceeds the max8 limit of {MAX8_MAX_ELEMS}")
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(logits.shape), str(logits.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (logits, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    weights, idx = _kernel(logits, cfg.block_size)
    return weights, idx


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
