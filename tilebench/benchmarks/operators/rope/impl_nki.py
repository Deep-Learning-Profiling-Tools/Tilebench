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
    PMAX = 128


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
    def rope_kernel(q, cos, sin, group_size):
        B, S, H, D = q.shape
        half = D // 2
        T = B * S
        row = H * D
        out = nl.ndarray((B, S, H, D), dtype=q.dtype, buffer=nl.shared_hbm)

        n_tiles = (T + PMAX - 1) // PMAX
        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (n_tiles + n_prog - 1) // n_prog
        t_lo = min(n_tiles, pid * per_core)
        t_hi = min(n_tiles, t_lo + per_core)

        for t in range(t_lo, t_hi):
            r0 = t * PMAX
            rs = min(PMAX, T - r0)
            s0 = r0 % S
            assert s0 + rs <= S, "a token tile must not cross a batch boundary"

            cos_t = nl.ndarray((rs, half), dtype=cos.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=cos_t, src=cos.ap(pattern=[[half, rs], [1, half]], offset=s0 * half))
            sin_t = nl.ndarray((rs, half), dtype=sin.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sin_t, src=sin.ap(pattern=[[half, rs], [1, half]], offset=s0 * half))

            for h0 in range(0, H, group_size):
                gs = min(group_size, H - h0)
                w = gs * D
                q_t = nl.ndarray((rs, w), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_t, src=q.ap(pattern=[[row, rs], [1, w]], offset=r0 * row + h0 * D))

                q1 = q_t.ap(pattern=[[w, rs], [D, gs], [1, half]], offset=0)
                q2 = q_t.ap(pattern=[[w, rs], [D, gs], [1, half]], offset=half)
                cos_b = cos_t.ap(pattern=[[half, rs], [0, gs], [1, half]], offset=0)
                sin_b = sin_t.ap(pattern=[[half, rs], [0, gs], [1, half]], offset=0)

                o_t = nl.ndarray((rs, w), dtype=q.dtype, buffer=nl.sbuf)
                o1 = o_t.ap(pattern=[[w, rs], [D, gs], [1, half]], offset=0)
                o2 = o_t.ap(pattern=[[w, rs], [D, gs], [1, half]], offset=half)

                a = nl.ndarray((rs, gs, half), dtype=nl.float32, buffer=nl.sbuf)
                b = nl.ndarray((rs, gs, half), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a, data1=q1, data2=cos_b, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=q2, data2=sin_b, op=nl.multiply)
                nisa.tensor_tensor(dst=o1, data1=a, data2=b, op=nl.subtract)
                nisa.tensor_tensor(dst=a, data1=q2, data2=cos_b, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=q1, data2=sin_b, op=nl.multiply)
                nisa.tensor_tensor(dst=o2, data1=a, data2=b, op=nl.add)

                nisa.dma_copy(dst=out.ap(pattern=[[row, rs], [1, w]], offset=r0 * row + h0 * D), src=o_t)

        return out


_DEFAULT_CONFIG = SimpleNamespace(group_size=16)
_SEARCH_SPACE = [SimpleNamespace(group_size=g) for g in (1, 2, 4, 8, 16, 32)]
_LNC = _lnc_degree()
_kernel = rope_kernel[_LNC] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    B, S, H, D = q.shape
    if S % PMAX and B > 1:
        raise NotImplementedError("rope NKI: seq_len must be a multiple of 128 when batch > 1")
    if autotune:
        space = [c for c in _SEARCH_SPACE if c.group_size <= H]
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(q.shape), str(q.dtype)),
            search_space=space,
            args_fn=lambda cfg: (q, cos, sin, cfg.group_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(q, cos, sin, min(cfg.group_size, H))


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
