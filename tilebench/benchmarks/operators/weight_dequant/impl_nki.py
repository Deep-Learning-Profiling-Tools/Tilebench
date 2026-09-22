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
    def weight_dequant_kernel(X, S, TILE_SIZE, block_size):
        M, N = X.shape
        out = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)
        s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
        s_stride = S.shape[1]
        n_blocks = (N + block_size - 1) // block_size

        n_tiles = (M + PMAX - 1) // PMAX
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = ti * PMAX
            rs = min(PMAX, M - r0)
            sr = r0 // TILE_SIZE
            s_raw = nl.ndarray((rs, s_cols), dtype=S.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s_raw, src=S.ap(pattern=[[0, rs], [1, s_cols]], offset=sr * s_stride))
            s_f32 = nl.ndarray((rs, s_cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s_f32, src=s_raw)
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, N - c0)
                x_tile = nl.ndarray((rs, cs), dtype=X.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=X[r0:r0 + rs, c0:c0 + cs])
                y_tile = nl.ndarray((rs, cs), dtype=X.dtype, buffer=nl.sbuf)
                if c0 % TILE_SIZE == 0 and cs % TILE_SIZE == 0:
                    n_sub = cs // TILE_SIZE
                    sc0 = c0 // TILE_SIZE
                    nisa.tensor_tensor(dst=y_tile.ap(pattern=[[cs, rs], [TILE_SIZE, n_sub], [1, TILE_SIZE]]),
                                       data1=x_tile.ap(pattern=[[cs, rs], [TILE_SIZE, n_sub], [1, TILE_SIZE]]),
                                       data2=s_f32.ap(pattern=[[s_cols, rs], [1, n_sub], [0, TILE_SIZE]], offset=sc0),
                                       op=nl.multiply)
                else:
                    for sc in range(c0 // TILE_SIZE, (c0 + cs + TILE_SIZE - 1) // TILE_SIZE):
                        a = max(sc * TILE_SIZE, c0) - c0
                        b = min((sc + 1) * TILE_SIZE, c0 + cs) - c0
                        nisa.tensor_scalar(dst=y_tile[0:rs, a:b], data=x_tile[0:rs, a:b],
                                           op0=nl.multiply, operand0=s_f32[0:rs, sc:sc + 1])
                nisa.dma_copy(dst=out[r0:r0 + rs, c0:c0 + cs], src=y_tile)
        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (256, 512, 1024, 2048, 4096)]
_kernel = weight_dequant_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if TILE_SIZE % PMAX != 0:
        raise NotImplementedError(
            f"weight_dequant NKI: TILE_SIZE ({TILE_SIZE}) must be a multiple of {PMAX}"
        )
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(X.shape), str(X.dtype), TILE_SIZE),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (X, S, TILE_SIZE, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(X, S, TILE_SIZE, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
