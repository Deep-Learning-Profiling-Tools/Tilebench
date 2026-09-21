"""NKI (AWS Trainium) implementation of reverse_array: ``out[i] = x[n - 1 - i]``.

Design (mirrors the Triton / cuTile kernels):

* Triton handles ``BLOCK_SIZE`` contiguous output elements per program by
  loading the mirrored input range. NKI processes ``[128, BLOCK_SIZE]`` tiles:
  a contiguous ``[128, BLOCK_SIZE]`` load (partition ``p`` holds input elements
  ``[start + p*B, start + (p+1)*B)``), an on-chip free-axis reversal
  (``tensor_copy`` from a stride ``-1`` view), and a store in which partition
  ``p`` lands at output offset ``n - start - (p+1)*B``. Partition order in the
  output is the mirror image of the input's, and the DMA engines only accept
  positive partition strides, so the store is one contiguous DMA per partition
  row (``BLOCK_SIZE`` elements each) -- a large ``BLOCK_SIZE`` keeps those DMAs
  efficient. ``BLOCK_SIZE`` search space is that of ``impl_triton.py`` /
  ``impl_cutile.py`` (1024/2048/4096/8192); the default is 8192 (largest
  per-row DMA).
* Work is split across the NeuronCores of the logical core (LNC2 on trn2):
  each program instance (``nl.program_id(0)``) owns a contiguous half of the
  input range.
* Tails are handled inside the kernel (``[128, q]`` + ``[1, r]`` tiles); no
  host-side padding / reshape / slicing.
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
    def reverse_kernel(a_input, block_size):
        """``out[i] = a_input[n - 1 - i]`` over a flat ``(n,)`` HBM tensor.

        Args:
            a_input: flat ``(n,)`` tensor in HBM (fp16 / bf16 / fp32 / int8).
            block_size: elements per partition per tile (compile-time constant).
        """
        n = a_input.shape[0]
        out = nl.ndarray((n,), dtype=a_input.dtype, buffer=nl.shared_hbm)

        num_programs = nl.num_programs()
        per_core = (n + num_programs - 1) // num_programs
        lo = nl.program_id(0) * per_core
        hi = min(n, lo + per_core)

        chunk = PMAX * block_size
        for j in range(max(0, (hi - lo + chunk - 1) // chunk)):
            start = lo + j * chunk
            count = min(chunk, hi - start)
            q = count // PMAX
            r = count - q * PMAX
            if q > 0:
                _tile_body(PMAX, q, start, n, a_input, out)
            if r > 0:
                _tile_body(1, r, start + q * PMAX, n, a_input, out)
        return out

    def _tile_body(p, f, start, n, a_input, out):
        """Reverse the ``[p, f]`` tile at flat input offset ``start``."""
        tile = nl.ndarray((p, f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=a_input.ap(pattern=[[f, p], [1, f]], offset=start))
        # Free-axis reversal on-chip: stride -1 along the free axis of the source view.
        rev = nl.ndarray((p, f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rev, src=tile.ap(pattern=[[f, p], [-1, f]], offset=f - 1))
        # Partition p holds input [start + p*f, start + (p+1)*f) -> output [n - start - (p+1)*f, ...).
        for pp in range(p):
            dst0 = n - start - (pp + 1) * f
            nisa.dma_copy(dst=out.ap(pattern=[[f, 1], [1, f]], offset=dst0), src=rev[pp:pp + 1, 0:f])


# Same block sizes as impl_triton.py / impl_cutile.py (search 1024/2048/4096/8192); default 8192.
_DEFAULT_CONFIG = SimpleNamespace(block_size=8192)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048, 4096, 8192)]
_kernel = reverse_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, N: int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x_flat = input.reshape(-1)
    n = x_flat.numel()
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, str(x_flat.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_flat, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = _kernel(x_flat, cfg.block_size)
    return result if input.dim() == 1 else result.reshape(input.shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
