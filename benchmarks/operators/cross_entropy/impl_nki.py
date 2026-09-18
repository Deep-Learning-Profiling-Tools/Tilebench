"""NKI (AWS Trainium) implementation of cross_entropy: ``loss[r] = logsumexp(x[r]) - x[r, t[r]]``.

Design (mirrors the Triton / cuTile kernels):

* Triton runs one program per row over the whole class axis
  (``BLOCK_CLASSES = next_pow2(C)``): row max, ``exp(x - max)``, sum, the target
  logit, ``loss = -(x[t] - max - log(sum))``. NKI puts ``block_size`` rows
  (<= 128, the SBUF partition count) on the partitions with the whole class axis
  in the free dimension and does the same steps per row tile:
  - ``max`` on the Vector engine;
  - fused ``exp(x - max)`` + free-axis sum on the Scalar engine
    (``activation_reduce``), then ``log`` of the sum;
  - the target logit is a data-dependent gather, done with the one-hot trick:
    a class-id iota compared against each row's target gives a mask, and a
    masked ``select_reduce(max)`` returns ``x[t]`` in one Vector-engine
    instruction;
  - ``loss = log_sum + max - x[t]`` on ``[rows, 1]`` columns.
  Triton autotunes only ``num_warps`` here; the NKI tunable is the rows per tile
  (default 128 = all partitions; search 32/64/128).
* Row tiles are split across the NeuronCores of the logical core (LNC2 on trn2)
  by ``nl.program_id(0)``. Partial row tiles are clamped with ``min()``.
"""
import functools
import os
import re
import subprocess
from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128

# Finite stand-in for -inf: the masked-out classes must never win the max-reduce.
NEG_BIG = -3.0e38


@functools.lru_cache(maxsize=1)
def _lnc_degree() -> int:
    """Logical-NeuronCore degree the kernel is launched with (``kernel[lnc]``).

    Must match the LNC the XLA module is compiled for: launching a ``kernel[2]``
    into an ``--lnc 1`` module silently computes only core 0's half.
    """
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
        """Per-row cross-entropy of ``[B, C]`` logits against ``[B, 1]`` int32 targets.

        Args:
            logits: ``[B, C]`` tensor in HBM (fp16 / bf16 / fp32).
            targets: ``[B, 1]`` int32 tensor in HBM with the target class per row.
            block_size: rows per tile (<= 128, compile-time constant).
        Returns:
            ``[B, 1]`` tensor in HBM with the dtype of ``logits``.
        """
        B, C = logits.shape
        out = nl.ndarray((B, 1), dtype=logits.dtype, buffer=nl.shared_hbm)
        rows_per_tile = min(block_size, PMAX)

        # class_iota[r, c] = c on every partition (built once per launch).
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
            # tensor_scalar's [P, 1] operand must be fp32 (class ids < 2^24: exact).
            tgt_f32 = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tgt_f32, src=tgt_i32)

            # ---- row max, exp(x - max) + sum (fused), log(sum) ----
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

            # ---- x[r, t[r]] via one-hot mask + masked max-reduce ----
            mask = nl.ndarray((rs, C), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=class_iota[0:rs, 0:C], op0=nl.equal, operand0=tgt_f32)
            # select_reduce's dst must share on_true's dtype; the reduce result is fp32.
            selected = nl.ndarray((rs, C), dtype=logits.dtype, buffer=nl.sbuf)
            tgt_logit = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=selected, predicate=mask, on_true=x_tile, on_false=NEG_BIG,
                               reduce_res=tgt_logit, reduce_cmd=nisa.reduce_cmd.reset_reduce,
                               reduce_op=nl.maximum)

            # ---- loss = log_sum + max - x[t] ----
            loss = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=loss, data1=log_sum, data2=row_max, op=nl.add)
            nisa.tensor_tensor(dst=loss, data1=loss, data2=tgt_logit, op=nl.subtract)
            loss_out = nl.ndarray((rs, 1), dtype=logits.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=loss_out, src=loss)
            nisa.dma_copy(dst=out[r0:r0 + rs, 0:1], src=loss_out)
        return out


# Triton autotunes num_warps only (whole class axis per program); the NKI knob is rows per tile.
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
