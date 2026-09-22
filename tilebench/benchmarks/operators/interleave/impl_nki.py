import functools
import os
import re
import subprocess
from types import SimpleNamespace

import torch

from tilebench.core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
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
    def interleave_kernel(a_input, b_input, block_size):
        n = a_input.shape[0]
        out = nl.ndarray((2 * n,), dtype=a_input.dtype, buffer=nl.shared_hbm)

        num_programs = nl.num_programs()
        per_core = (n + num_programs - 1) // num_programs
        lo = nl.program_id(0) * per_core
        hi = min(n, lo + per_core)

        chunk = PMAX * block_size
        for j in range(max(0, (hi - lo + chunk - 1) // chunk)):
            start = lo + j * chunk
            count = min(chunk, hi - start)
            q = count // PMAX
            r = count - q * PMAX
            if q > 0:
                _tile_body(PMAX, q, start, a_input, b_input, out)
            if r > 0:
                _tile_body(1, r, start + q * PMAX, a_input, b_input, out)
        return out

    def _tile_body(p, f, start, a_input, b_input, out):
        a_tile = nl.ndarray((p, f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a_input.ap(pattern=[[f, p], [1, f]], offset=start))
        b_tile = nl.ndarray((p, f), dtype=b_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b_input.ap(pattern=[[f, p], [1, f]], offset=start))
        out_tile = nl.ndarray((p, 2 * f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_tile[0:p, 0:2 * f:2], src=a_tile)
        nisa.tensor_copy(dst=out_tile[0:p, 1:2 * f:2], src=b_tile)
        nisa.dma_copy(dst=out.ap(pattern=[[2 * f, p], [1, 2 * f]], offset=2 * start), src=out_tile)


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048, 4096, 8192)]
_kernel = interleave_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(A: torch.Tensor, B: torch.Tensor, N: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    a_flat, b_flat = A.reshape(-1), B.reshape(-1)
    n = a_flat.numel()
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, str(a_flat.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (a_flat, b_flat, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(a_flat, b_flat, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
