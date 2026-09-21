"""NKI (AWS Trainium) implementation of interleave: ``out[0::2] = A; out[1::2] = B``.

Design (mirrors the Triton / cuTile kernels):

* The flat inputs are processed in fixed-size blocks. A Triton program handles
  ``BLOCK_SIZE`` contiguous elements of ``A`` and ``B`` and writes ``2 * BLOCK_SIZE``
  outputs; the NKI analogue is one ``[128, BLOCK_SIZE]`` SBUF tile per input --
  128 partitions (the 128 parallel lanes of the Vector engine) times
  ``BLOCK_SIZE`` contiguous elements per partition. The default and the autotune
  search space use the same ``BLOCK_SIZE`` values as ``impl_triton.py`` /
  ``impl_cutile.py`` (default 1024; search 1024/2048/4096/8192).
* The interleaving itself is done on-chip with strided SBUF destination slices
  (``out_tile[:, 0::2] = a``, ``out_tile[:, 1::2] = b``), so every HBM load and
  the HBM store stay fully contiguous DMAs.
* Work is split across the NeuronCores of the logical core (LNC2 on trn2):
  each program instance (``nl.program_id(0)``) owns a contiguous half of the
  element range, so both physical cores' DMA engines are used.
* Tails are handled inside the kernel: the last block of a core may cover fewer
  than ``128 * BLOCK_SIZE`` elements, in which case it is processed as one
  ``[128, q]`` tile plus one ``[1, r]`` tile (``r < 128``). No host-side
  padding / reshape / slicing is needed -- those torch ops would be fused into
  the same NEFF as the kernel and dominate its runtime.
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
    def interleave_kernel(a_input, b_input, block_size):
        """``out[2i] = a[i]; out[2i+1] = b[i]`` over flat ``(n,)`` HBM tensors.

        Args:
            a_input, b_input: flat ``(n,)`` tensors in HBM (same dtype).
            block_size: elements per partition per tile (compile-time constant).
        """
        n = a_input.shape[0]
        out = nl.ndarray((2 * n,), dtype=a_input.dtype, buffer=nl.shared_hbm)

        # Contiguous element range owned by this program instance (NeuronCore).
        num_programs = nl.num_programs()
        per_core = (n + num_programs - 1) // num_programs
        lo = nl.program_id(0) * per_core
        hi = min(n, lo + per_core)

        chunk = PMAX * block_size
        for j in range(max(0, (hi - lo + chunk - 1) // chunk)):
            start = lo + j * chunk
            count = min(chunk, hi - start)
            q = count // PMAX            # full partitions: [PMAX, q] tiles
            r = count - q * PMAX         # remaining < PMAX elements: [1, r] tiles
            if q > 0:
                _tile_body(PMAX, q, start, a_input, b_input, out)
            if r > 0:
                _tile_body(1, r, start + q * PMAX, a_input, b_input, out)
        return out

    def _tile_body(p, f, start, a_input, b_input, out):
        """Interleave the ``[p, f]`` tiles of a and b at flat input offset ``start``."""
        a_tile = nl.ndarray((p, f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a_input.ap(pattern=[[f, p], [1, f]], offset=start))
        b_tile = nl.ndarray((p, f), dtype=b_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b_input.ap(pattern=[[f, p], [1, f]], offset=start))
        # On-chip interleave into the even / odd free-dim slots of one [p, 2f] tile.
        out_tile = nl.ndarray((p, 2 * f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_tile[0:p, 0:2 * f:2], src=a_tile)
        nisa.tensor_copy(dst=out_tile[0:p, 1:2 * f:2], src=b_tile)
        # Partition p of the tile holds inputs [start + p*f, start + (p+1)*f), whose
        # outputs are the contiguous range [2*(start + p*f), 2*(start + (p+1)*f)).
        nisa.dma_copy(dst=out.ap(pattern=[[2 * f, p], [1, 2 * f]], offset=2 * start), src=out_tile)


# Same block sizes as impl_triton.py / impl_cutile.py (default 1024; search 1024/2048/4096/8192).
_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048, 4096, 8192)]
_kernel = interleave_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(A: torch.Tensor, B: torch.Tensor, N: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    a_flat, b_flat = A.reshape(-1), B.reshape(-1)
    n = a_flat.numel()
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, str(a_flat.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (a_flat, b_flat, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(a_flat, b_flat, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
