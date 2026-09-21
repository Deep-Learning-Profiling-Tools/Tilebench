"""NKI (AWS Trainium) implementation of mean_reduction: ``y[r] = mean(x[r, :])`` (fp32 output).

Design (mirrors the Triton / cuTile kernels):

* Triton runs one program per row and walks the row in ``BLOCK_N``-column blocks,
  accumulating in fp32. NKI puts 128 rows on the 128 SBUF partitions and walks
  the same ``BLOCK_N``-column blocks: each block is one ``[128, BLOCK_N]`` DMA and
  one Vector-engine free-axis reduction (fp32 accumulation) into a per-block
  partial column; the partials are then reduced once more and scaled by ``1/N``.
  ``BLOCK_N`` defaults / search space are those of ``impl_triton.py`` /
  ``impl_cutile.py`` (default 1024; search 512/1024/2048).
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
    def mean_rowwise_kernel(a_input, block_size):
        """Row-wise mean of a ``[n_rows, n_cols]`` HBM tensor -> ``[n_rows, 1]`` fp32.

        Args:
            a_input: ``[n_rows, n_cols]`` tensor in HBM (fp16 / bf16 / fp32).
            block_size: columns per block (``BLOCK_N``, compile-time constant).
        """
        n_rows, n_cols = a_input.shape
        out = nl.ndarray((n_rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        n_blocks = (n_cols + block_size - 1) // block_size
        inv_n = 1.0 / n_cols

        n_tiles = (n_rows + PMAX - 1) // PMAX
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = ti * PMAX
            rs = min(PMAX, n_rows - r0)
            # One fp32 partial sum per column block, reduced once at the end.
            parts = nl.ndarray((rs, n_blocks), dtype=nl.float32, buffer=nl.sbuf)
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, n_cols - c0)
                x_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[r0:r0 + rs, c0:c0 + cs])
                # Vector engine reduces along the free axis with an fp32 accumulator.
                nisa.tensor_reduce(dst=parts[0:rs, cb:cb + 1], op=nl.add, data=x_tile, axis=(1,))
            mean = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            if n_blocks == 1:
                nisa.tensor_scalar(dst=mean, data=parts, op0=nl.multiply, operand0=inv_n)
            else:
                total = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=total, op=nl.add, data=parts, axis=(1,))
                nisa.tensor_scalar(dst=mean, data=total, op0=nl.multiply, operand0=inv_n)
            nisa.dma_copy(dst=out[r0:r0 + rs, 0:1], src=mean)
        return out


# Same column block sizes as impl_triton.py (BLOCK_N default 1024; search 512/1024/2048).
_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048)]
_kernel = mean_rowwise_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("mean_reduction NKI: int8 not supported")
    if x.dim() != 2 or dim != 1:
        raise NotImplementedError("mean_reduction NKI: expects a 2D input reduced over dim=1")
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(x, cfg.block_size).reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
