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
    def dropout_kernel(x_input, keep_input, keep_scale, block_size):
        n = x_input.shape[0]
        out = nl.ndarray((n,), dtype=x_input.dtype, buffer=nl.shared_hbm)

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
                _tile_body(PMAX, q, start, x_input, keep_input, keep_scale, out)
            if r > 0:
                _tile_body(1, r, start + q * PMAX, x_input, keep_input, keep_scale, out)
        return out

    def _tile_body(p, f, start, x_input, keep_input, keep_scale, out):
        x_tile = nl.ndarray((p, f), dtype=x_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x_input.ap(pattern=[[f, p], [1, f]], offset=start))
        keep_tile = nl.ndarray((p, f), dtype=keep_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=keep_tile, src=keep_input.ap(pattern=[[f, p], [1, f]], offset=start))
        nisa.scalar_tensor_tensor(dst=x_tile, data=x_tile, op0=nl.multiply, operand0=keep_scale,
                                  op1=nl.multiply, operand1=keep_tile)
        nisa.dma_copy(dst=out.ap(pattern=[[f, p], [1, f]], offset=start), src=x_tile)


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048)]
_kernel = dropout_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, x_keep: torch.Tensor, p: float, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x_flat, keep_flat = x.reshape(-1), x_keep.reshape(-1)
    n = x_flat.numel()
    keep_scale = 1.0 / (1.0 - p)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, str(x_flat.dtype), keep_scale),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_flat, keep_flat, keep_scale, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = _kernel(x_flat, keep_flat, keep_scale, cfg.block_size)
    return result if x.dim() == 1 else result.reshape(x.shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
