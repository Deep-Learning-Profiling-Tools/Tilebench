"""NKI (AWS Trainium) implementation of dequantize_rowwise: ``out[r, c] = fp16(x[r, c] * state[r] / 127)``.

Design (mirrors the Triton / cuTile kernels):

* Triton launches one program per (row, ``CHUNK``-column block) and does
  ``x * scale * (1/127)`` in fp32 before storing fp16. NKI walks the same
  ``CHUNK``-column blocks, with the rows of each program's share laid out as
  ``rows_per_partition`` consecutive rows per SBUF partition (partition ``p``
  owns rows ``[p * rpp, (p + 1) * rpp)``): one ``[128, rpp, CHUNK]`` int8 DMA
  brings ``rpp`` row segments per partition in a single transfer, one
  Vector-engine ``tensor_tensor`` multiplies by the ``[128, rpp]`` fp32 scale
  column (``state / 127``, broadcast along the columns) with the fp16 cast on
  the way out, and one DMA stores the block. Keeping several rows per partition
  makes every DMA large enough to run at HBM bandwidth on these small matrices.
  ``CHUNK`` defaults / search space are those of ``impl_triton.py`` /
  ``impl_cutile.py`` (default 512; search 256/512/1024).
* Rows are split across the NeuronCores of the logical core (LNC2 on trn2) by
  ``nl.program_id(0)``. If a program's share is not a multiple of 128 rows it
  falls back to plain 128-row tiles (``[128, CHUNK]`` blocks, per-partition
  scalar scale).
* ``state_x`` is read as a ``[rows, 1]`` view -- nothing is padded on the host.
"""
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
    def dequantize_rowwise_kernel(x_input, state_input, block_size):
        """``out[r, c] = x[r, c] * state[r] / 127`` in fp16 over a ``[rows, cols]`` HBM tensor.

        Args:
            x_input: ``[rows, cols]`` quantized values in HBM (int8).
            state_input: ``[rows, 1]`` per-row absmax scales in HBM (fp32).
            block_size: columns per block (``CHUNK``, compile-time constant).
        """
        rows, cols = x_input.shape
        out = nl.ndarray((rows, cols), dtype=nl.float16, buffer=nl.shared_hbm)
        n_blocks = (cols + block_size - 1) // block_size

        num_programs = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (rows + num_programs - 1) // num_programs
        r_lo = pid * per_core
        r_hi = min(rows, r_lo + per_core)
        my_rows = max(0, r_hi - r_lo)
        if my_rows > 0 and my_rows % PMAX == 0:
            _rows_per_partition(x_input, state_input, out, r_lo, my_rows // PMAX, cols, block_size, n_blocks)
        else:
            _row_tiles(x_input, state_input, out, r_lo, r_hi, cols, block_size, n_blocks)
        return out

    def _rows_per_partition(x_input, state_input, out, r_lo, rpp, cols, block_size, n_blocks):
        """Partition ``p`` owns rows ``r_lo + p*rpp .. +rpp``; blocks are ``[128, rb, cs]`` tiles."""
        # scale[p, i] = state[r_lo + p*rpp + i] / 127 (fp32)
        state_tile = nl.ndarray((PMAX, rpp), dtype=state_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_tile, src=state_input.ap(pattern=[[rpp, PMAX], [1, rpp]], offset=r_lo))
        scale = nl.ndarray((PMAX, rpp), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scale, data=state_tile, op0=nl.multiply, operand0=1.0 / 127.0)
        rb = max(1, min(rpp, 16384 // block_size))       # rows per partition per block (SBUF budget)
        for rb0 in range(0, rpp, rb):
            rbn = min(rb, rpp - rb0)
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, cols - c0)
                base = (r_lo + rb0) * cols + c0
                x_tile = nl.ndarray((PMAX, rbn, cs), dtype=x_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=x_input.ap(pattern=[[rpp * cols, PMAX], [cols, rbn], [1, cs]], offset=base))
                y_tile = nl.ndarray((PMAX, rbn, cs), dtype=nl.float16, buffer=nl.sbuf)
                # int8 * per-row fp32 scale (broadcast along the columns) -> fp16
                nisa.tensor_tensor(dst=y_tile, data1=x_tile,
                                   data2=scale.ap(pattern=[[rpp, PMAX], [1, rbn], [0, cs]], offset=rb0),
                                   op=nl.multiply)
                nisa.dma_copy(dst=out.ap(pattern=[[rpp * cols, PMAX], [cols, rbn], [1, cs]], offset=base), src=y_tile)

    def _row_tiles(x_input, state_input, out, r_lo, r_hi, cols, block_size, n_blocks):
        """Fallback: plain 128-row tiles with a per-partition scalar scale."""
        for r0 in range(r_lo, r_hi, PMAX):
            rs = min(PMAX, r_hi - r0)
            state_tile = nl.ndarray((rs, 1), dtype=state_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=state_tile, src=state_input[r0:r0 + rs, 0:1])
            scale = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=scale, data=state_tile, op0=nl.multiply, operand0=1.0 / 127.0)
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, cols - c0)
                x_tile = nl.ndarray((rs, cs), dtype=x_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=x_input[r0:r0 + rs, c0:c0 + cs])
                y_tile = nl.ndarray((rs, cs), dtype=nl.float16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=scale)
                nisa.dma_copy(dst=out[r0:r0 + rs, c0:c0 + cs], src=y_tile)


# Same column block sizes as impl_triton.py / impl_cutile.py (CHUNK default 512; search 256/512/1024).
_DEFAULT_CONFIG = SimpleNamespace(block_size=512)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (256, 512, 1024)]
_kernel = dequantize_rowwise_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, state_x: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    state_2d = state_x.reshape(-1, 1)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x, state_2d, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(x, state_2d, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
