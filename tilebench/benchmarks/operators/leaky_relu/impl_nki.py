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
    def leaky_relu_kernel(a_input, block_size):
        n = a_input.shape[0]
        out = nl.ndarray((n,), dtype=a_input.dtype, buffer=nl.shared_hbm)

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
                _tile_body(PMAX, q, start, a_input, out)
            if r > 0:
                _tile_body(1, r, start + q * PMAX, a_input, out)
        return out

    def _tile_body(p, f, start, a_input, out):
        tile = nl.ndarray((p, f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=a_input.ap(pattern=[[f, p], [1, f]], offset=start))
        nisa.scalar_tensor_tensor(dst=tile, data=tile, op0=nl.multiply, operand0=0.01,
                                  op1=nl.maximum, operand1=tile)
        nisa.dma_copy(dst=out.ap(pattern=[[f, p], [1, f]], offset=start), src=tile)


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048, 4096, 8192)]
_kernel = leaky_relu_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, N: int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x_flat = input.reshape(-1)
    n = x_flat.numel()
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, str(x_flat.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_flat, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = _kernel(x_flat, cfg.block_size)
    return result if input.dim() == 1 else result.reshape(input.shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
