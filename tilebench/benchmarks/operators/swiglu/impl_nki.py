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
    def swiglu_kernel(x_input, y_input, block_size):
        P, F = x_input.shape
        num_row_blocks = (P + PMAX - 1) // PMAX
        num_col_blocks = (F + block_size - 1) // block_size

        out = nl.ndarray(x_input.shape, dtype=x_input.dtype, buffer=nl.shared_hbm)

        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (num_row_blocks + n_prog - 1) // n_prog
        r_lo = min(num_row_blocks, pid * per_core)
        r_hi = min(num_row_blocks, r_lo + per_core)

        for i in range(r_lo, r_hi):
            p_start = i * PMAX
            p_sz = min(PMAX, P - p_start)
            for j in range(num_col_blocks):
                f_start = j * block_size
                f_sz = min(block_size, F - f_start)

                x_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=x_input[p_start:p_start + p_sz, f_start:f_start + f_sz])
                y_tile = nl.ndarray((p_sz, f_sz), dtype=y_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=y_tile, src=y_input[p_start:p_start + p_sz, f_start:f_start + f_sz])

                sig_tile = nl.ndarray((p_sz, f_sz), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sig_tile, data=x_tile, op=nl.sigmoid)
                silu_tile = nl.ndarray((p_sz, f_sz), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=silu_tile, data1=sig_tile, data2=x_tile, op=nl.multiply)

                res_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=res_tile, data1=silu_tile, data2=y_tile, op=nl.multiply)

                nisa.dma_copy(dst=out[p_start:p_start + p_sz, f_start:f_start + f_sz], src=res_tile)

        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048, 4096, 8192, 16384)]
_SBUF_BUDGET = 96 * 1024
_LNC = _lnc_degree()
_kernel = swiglu_kernel[_LNC] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False,
        **kwargs) -> torch.Tensor:
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype)),
            search_space=[c for c in _SEARCH_SPACE
                          if c.block_size * (3 * x.element_size() + 8) <= _SBUF_BUDGET],
            args_fn=lambda cfg: (x, y, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(x, y, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
