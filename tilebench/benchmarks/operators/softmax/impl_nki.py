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
except ImportError:
    nki = None

PMAX = 128
NEG_INF = -3.0e38
RESIDENT_CAP = {torch.float32: 2048, torch.float16: 8192, torch.bfloat16: 8192}


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


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def softmax_kernel(a_input, resident, block_size):
        n_rows, n_cols = a_input.shape
        out_hbm = nl.ndarray((n_rows, n_cols), dtype=a_input.dtype, buffer=nl.shared_hbm)
        n_row_tiles = _div_ceil(n_rows, PMAX)

        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = _div_ceil(n_row_tiles, n_prog)
        t_lo = min(n_row_tiles, pid * per_core)
        t_hi = min(n_row_tiles, t_lo + per_core)

        for row_tile in range(t_lo, t_hi):
            r0 = row_tile * PMAX
            rs = min(PMAX, n_rows - r0)

            if resident:
                x_tile = nl.ndarray((rs, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[r0:r0 + rs, 0:n_cols])
                row_max = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=x_tile, axis=(1,))
                work = nl.ndarray((rs, n_cols), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=work, data=x_tile, op0=nl.subtract, operand0=row_max)
                nisa.activation(dst=work, op=nl.exp, data=work)
                row_sum = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=row_sum, op=nl.add, data=work, axis=(1,))
                inv_sum = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_sum, data=row_sum)
                y_tile = nl.ndarray((rs, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=work, op0=nl.multiply, operand0=inv_sum)
                nisa.dma_copy(dst=out_hbm[r0:r0 + rs, 0:n_cols], src=y_tile)
                continue

            n_col_tiles = _div_ceil(n_cols, block_size)
            row_max = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=row_max, value=NEG_INF)
            row_sum = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=row_sum, value=0.0)

            for c in range(n_col_tiles):
                c0 = c * block_size
                cs = min(block_size, n_cols - c0)
                x_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[r0:r0 + rs, c0:c0 + cs])

                blk_max = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=blk_max, op=nl.maximum, data=x_tile, axis=(1,))
                new_max = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=new_max, data1=row_max, data2=blk_max, op=nl.maximum)

                corr = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=corr, data1=row_max, data2=new_max, op=nl.subtract)
                nisa.activation(dst=corr, op=nl.exp, data=corr)
                nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=corr, op=nl.multiply)

                work = nl.ndarray((rs, cs), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=work, data=x_tile, op0=nl.subtract, operand0=new_max)
                nisa.activation(dst=work, op=nl.exp, data=work)
                part = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=part, op=nl.add, data=work, axis=(1,))
                nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=part, op=nl.add)
                nisa.tensor_copy(dst=row_max, src=new_max)

            inv_sum = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_sum, data=row_sum)

            for c in range(n_col_tiles):
                c0 = c * block_size
                cs = min(block_size, n_cols - c0)
                x_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[r0:r0 + rs, c0:c0 + cs])
                work = nl.ndarray((rs, cs), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=work, data=x_tile, op0=nl.subtract, operand0=row_max)
                nisa.activation(dst=work, op=nl.exp, data=work)
                y_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=work, op0=nl.multiply, operand0=inv_sum)
                nisa.dma_copy(dst=out_hbm[r0:r0 + rs, c0:c0 + cs], src=y_tile)

        return out_hbm


_DEFAULT_CONFIG = SimpleNamespace(resident=False, block_size=1024)
_LNC = _lnc_degree()
_kernel = softmax_kernel[_LNC] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def _search_space(n_cols: int, dtype) -> list:
    space = [SimpleNamespace(resident=False, block_size=b) for b in (512, 1024, 2048)]
    if n_cols <= RESIDENT_CAP.get(dtype, 2048):
        space.append(SimpleNamespace(resident=True, block_size=n_cols))
    return space


def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype)),
            search_space=_search_space(x.shape[-1], x.dtype),
            args_fn=lambda cfg: (x, cfg.resident, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(x, cfg.resident, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
