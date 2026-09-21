"""NKI (AWS Trainium) implementation of quantize_global: ``y = fp16(x)``.

Design (mirrors the Triton / cuTile kernels):

* The flat input is processed in fixed-size blocks. A Triton program handles
  ``BLOCK_SIZE`` contiguous elements; the NKI analogue is one ``[128, BLOCK_SIZE]``
  SBUF tile -- 128 partitions (the 128 parallel lanes of the Vector/Scalar
  engines) times ``BLOCK_SIZE`` contiguous elements per partition. The default
  and the autotune search space use the same ``BLOCK_SIZE`` values as
  ``impl_triton.py`` / ``impl_cutile.py`` (default 2048; search 2048/4096/8192/16384).
* Work is split across the NeuronCores of the logical core (LNC2 on trn2):
  each program instance (``nl.program_id(0)``) owns a contiguous half of the
  element range, so both physical cores' DMA engines are used.
* Tails are handled inside the kernel: the last block of a core may cover fewer
  than ``128 * BLOCK_SIZE`` elements, in which case it is processed as one
  ``[128, q]`` tile plus one ``[1, r]`` tile (``r < 128``). No host-side
  padding / reshape / slicing is needed -- those torch ops would be fused into
  the same NEFF as the kernel and dominate its runtime.
* The kernel reads and writes the flat ``(n,)`` HBM tensors directly through
  ``.ap()`` access patterns, so the output is returned to the caller as-is.
* The Triton search space lists 8192 twice; NkiAutotuner rejects duplicate
  candidates, so the NKI search space is the de-duplicated set.
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
    def quantize_global_kernel(a_input, block_size):
        """``out[i] = float16(a_input[i])`` over a flat ``(n,)`` fp32 HBM tensor.

        Args:
            a_input: flat ``(n,)`` fp32 tensor in HBM.
            block_size: elements per partition per tile (compile-time constant).
        """
        n = a_input.shape[0]
        out = nl.ndarray((n,), dtype=nl.float16, buffer=nl.shared_hbm)

        # Contiguous element range owned by this program instance (NeuronCore).
        num_programs = nl.num_programs()
        per_core = (n + num_programs - 1) // num_programs
        lo = nl.program_id(0) * per_core
        hi = min(n, lo + per_core)

        chunk = PMAX * block_size
        for j in range(max(0, (hi - lo + chunk - 1) // chunk)):
            start = lo + j * chunk
            count = min(chunk, hi - start)
            q = count // PMAX            # full partitions: [PMAX, q] tile
            r = count - q * PMAX         # remaining < PMAX elements: [1, r] tile
            if q > 0:
                _tile_body(PMAX, q, start, a_input, out)
            if r > 0:
                _tile_body(1, r, start + q * PMAX, a_input, out)
        return out

    def _tile_body(p, f, start, a_input, out):
        """y = fp16(x) on the ``[p, f]`` tile at flat offset ``start``."""
        tile = nl.ndarray((p, f), dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=a_input.ap(pattern=[[f, p], [1, f]], offset=start))
        # The dtype cast happens in the on-chip copy (DMA cannot convert dtypes).
        q_tile = nl.ndarray((p, f), dtype=nl.float16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_tile, src=tile)
        nisa.dma_copy(dst=out.ap(pattern=[[f, p], [1, f]], offset=start), src=q_tile)



# Same block sizes as impl_triton.py / impl_cutile.py (default 2048; search 2048/4096/8192/16384).
_DEFAULT_CONFIG = SimpleNamespace(block_size=2048)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (2048, 4096, 8192, 16384)]
_kernel = quantize_global_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x_flat = x.reshape(-1)
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
    return result if x.dim() == 1 else result.reshape(x.shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
