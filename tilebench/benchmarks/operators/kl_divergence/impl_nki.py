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

TINY = 1.0e-38


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
    def kl_divergence_kernel(log_p_input, q_input, block_size):
        n_rows, n_cols = log_p_input.shape
        out = nl.ndarray((n_rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        n_blocks = (n_cols + block_size - 1) // block_size

        n_tiles = (n_rows + PMAX - 1) // PMAX
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = ti * PMAX
            rs = min(PMAX, n_rows - r0)
            parts = nl.ndarray((rs, n_blocks), dtype=nl.float32, buffer=nl.sbuf)
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, n_cols - c0)
                log_p = nl.ndarray((rs, cs), dtype=log_p_input.dtype, buffer=nl.sbuf)
                q = nl.ndarray((rs, cs), dtype=q_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=log_p, src=log_p_input[r0:r0 + rs, c0:c0 + cs])
                nisa.dma_copy(dst=q, src=q_input[r0:r0 + rs, c0:c0 + cs])
                t = nl.ndarray((rs, cs), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=t, data=q, op0=nl.maximum, operand0=TINY)
                nisa.activation(dst=t, op=nl.log, data=t)
                nisa.scalar_tensor_tensor(dst=t, data=log_p, op0=nl.multiply, operand0=-1.0,
                                          op1=nl.add, operand1=t)
                nisa.tensor_tensor(dst=t, data1=t, data2=q, op=nl.multiply)
                nisa.activation_reduce(dst=t, op=nl.copy, data=t,
                                       reduce_op=nl.add, reduce_res=parts[0:rs, cb:cb + 1])
            if n_blocks == 1:
                nisa.dma_copy(dst=out[r0:r0 + rs, 0:1], src=parts)
            else:
                total = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=total, op=nl.add, data=parts, axis=(1,))
                nisa.dma_copy(dst=out[r0:r0 + rs, 0:1], src=total)
        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048, 4096)]
_kernel = kl_divergence_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if log_y_pred.dtype == torch.int8 or y_true.dtype == torch.int8:
        raise NotImplementedError("kl_divergence NKI: int8 not supported")
    lead_shape = log_y_pred.shape[:-1]
    cols = log_y_pred.shape[-1]
    log_p_2d = log_y_pred.reshape(-1, cols)
    q_2d = y_true.reshape(-1, cols)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(log_p_2d.shape), str(log_p_2d.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (log_p_2d, q_2d, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(log_p_2d, q_2d, cfg.block_size).reshape(lead_shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
