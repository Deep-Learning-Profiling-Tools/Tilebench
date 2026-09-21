"""NKI (AWS Trainium) implementation of kl_divergence: ``loss[r] = sum_c q * (log(q) - log_p)``.

Design (mirrors the Triton / cuTile kernels):

* Triton runs one program per row and walks the row in ``BLOCK_SIZE``-column
  blocks, accumulating ``q * (where(q > 0, log(q), 0) - log_p)`` in fp32. NKI
  puts 128 rows on the 128 SBUF partitions and walks the same column blocks;
  per block: two ``[128, BLOCK]`` DMAs, ``log(max(q, tiny))`` on the Scalar engine
  (the clamp makes the ``q == 0`` term exactly 0, as Triton's ``where`` does),
  ``(log_q - log_p) * q`` on the Vector engine and the free-axis sum on the Scalar
  engine (identity activation + reduce, fp32). The per-block partials are reduced
  once at the end.
  ``BLOCK_SIZE`` defaults / search space are those of ``impl_triton.py`` /
  ``impl_cutile.py`` (default 1024; search 512/1024/2048/4096).
* Row tiles are split across the NeuronCores of the logical core (LNC2 on trn2)
  by ``nl.program_id(0)``.
* Partial row tiles / column blocks are clamped with ``min()``; nothing is
  padded or sliced on the host.
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

TINY = 1.0e-38   # smallest normal fp32 (log(TINY) is finite)


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
    def kl_divergence_kernel(log_p_input, q_input, block_size):
        """Row-wise KL divergence of ``[n_rows, n_cols]`` HBM tensors -> ``[n_rows, 1]`` fp32.

        Args:
            log_p_input: ``[n_rows, n_cols]`` log-probabilities in HBM (fp32).
            q_input: ``[n_rows, n_cols]`` probabilities in HBM (fp32).
            block_size: columns per block (``BLOCK_SIZE``, compile-time constant).
        """
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
                # log(q) on the Scalar engine. log(0) = -inf would give 0 * -inf = NaN, so q is
                # clamped to the smallest positive fp32 first: where q == 0 the term becomes
                # (log(tiny) - log_p) * 0 = 0, the same value Triton's tl.where(q > 0, ...) yields.
                # One fp32 work tile per block, updated in place (keeps the SBUF footprint at
                # three [128, BLOCK] tiles per block so several blocks stay in flight).
                t = nl.ndarray((rs, cs), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=t, data=q, op0=nl.maximum, operand0=TINY)      # max(q, tiny)
                nisa.activation(dst=t, op=nl.log, data=t)                               # log(q)
                # term = (log_q - log_p) * q on the Vector engine, free-axis sum on the Scalar
                # engine (identity activation + reduce, fp32 accumulate) to balance the engines.
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


# Same column block sizes as impl_triton.py (BLOCK_SIZE default 1024; search 512/1024/2048/4096).
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
