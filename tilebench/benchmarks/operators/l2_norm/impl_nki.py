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


def _itemsize(dtype) -> int:
    return 4 if dtype == nl.float32 else 2


if nki is not None:
    @nki.jit
    def l2_norm_kernel(a_input, eps, block_size):
        n_cols = a_input.shape[-1]
        n_rows = 1
        for d in a_input.shape[:-1]:
            n_rows *= d
        out = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
        n_blocks = (n_cols + block_size - 1) // block_size
        keep = n_cols * _itemsize(a_input.dtype) <= 64 * 1024

        n_tiles = (n_rows + PMAX - 1) // PMAX
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = ti * PMAX
            rs = min(PMAX, n_rows - r0)
            parts = nl.ndarray((rs, n_blocks), dtype=nl.float32, buffer=nl.sbuf)
            x_tiles = []
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, n_cols - c0)
                x_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input.ap(pattern=[[n_cols, rs], [1, cs]], offset=r0 * n_cols + c0))
                if keep:
                    x_tiles.append(x_tile)
                sq_tile = nl.ndarray((rs, cs), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq_tile, op=nl.square, data=x_tile,
                                       reduce_op=nl.add, reduce_res=parts[0:rs, cb:cb + 1])
            sum_sq = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            if n_blocks == 1:
                nisa.tensor_scalar(dst=sum_sq, data=parts, op0=nl.add, operand0=eps)
            else:
                nisa.tensor_reduce(dst=sum_sq, op=nl.add, data=parts, axis=(1,))
                nisa.tensor_scalar(dst=sum_sq, data=sum_sq, op0=nl.add, operand0=eps)
            rstd = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=sum_sq, op0=nl.rsqrt, operand0=None)
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, n_cols - c0)
                if keep:
                    x_tile = x_tiles[cb]
                else:
                    x_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=x_tile, src=a_input.ap(pattern=[[n_cols, rs], [1, cs]], offset=r0 * n_cols + c0))
                y_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=rstd)
                nisa.dma_copy(dst=out.ap(pattern=[[n_cols, rs], [1, cs]], offset=r0 * n_cols + c0), src=y_tile)
        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048)]
_kernel = l2_norm_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, eps: float = 1e-6, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("l2_norm NKI: int8 not supported")
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype), float(eps)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x, float(eps), cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(x, float(eps), cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
