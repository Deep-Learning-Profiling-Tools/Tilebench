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

NEG_BIG = -3.0e38


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
    def cross_entropy_kernel(logits, targets, block_size):
        B, C = logits.shape
        out = nl.ndarray((B, 1), dtype=logits.dtype, buffer=nl.shared_hbm)
        rows_per_tile = min(block_size, PMAX)

        class_iota = nl.ndarray((rows_per_tile, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=class_iota, pattern=[[1, C]], offset=0, channel_multiplier=0)

        n_tiles = (B + rows_per_tile - 1) // rows_per_tile
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = ti * rows_per_tile
            rs = min(rows_per_tile, B - r0)
            x_tile = nl.ndarray((rs, C), dtype=logits.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=logits[r0:r0 + rs, 0:C])
            tgt_i32 = nl.ndarray((rs, 1), dtype=targets.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tgt_i32, src=targets[r0:r0 + rs, 0:1])
            tgt_f32 = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tgt_f32, src=tgt_i32)

            row_max = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=x_tile, axis=(1,))
            neg_max = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-1.0)
            exp_tile = nl.ndarray((rs, C), dtype=nl.float32, buffer=nl.sbuf)
            row_sum = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max,
                                   reduce_op=nl.add, reduce_res=row_sum)
            log_sum = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=log_sum, op=nl.log, data=row_sum)

            mask = nl.ndarray((rs, C), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=class_iota[0:rs, 0:C], op0=nl.equal, operand0=tgt_f32)
            selected = nl.ndarray((rs, C), dtype=logits.dtype, buffer=nl.sbuf)
            tgt_logit = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=selected, predicate=mask, on_true=x_tile, on_false=NEG_BIG,
                               reduce_res=tgt_logit, reduce_cmd=nisa.reduce_cmd.reset_reduce,
                               reduce_op=nl.maximum)

            loss = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=loss, data1=log_sum, data2=row_max, op=nl.add)
            nisa.tensor_tensor(dst=loss, data1=loss, data2=tgt_logit, op=nl.subtract)
            loss_out = nl.ndarray((rs, 1), dtype=logits.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=loss_out, src=loss)
            nisa.dma_copy(dst=out[r0:r0 + rs, 0:1], src=loss_out)
        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=128)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (32, 64, 128)]
_kernel = cross_entropy_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(logits: torch.Tensor, targets: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    targets_2d = targets.reshape(-1, 1).to(torch.int32)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(logits.shape), str(logits.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (logits, targets_2d, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(logits, targets_2d, cfg.block_size).reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
