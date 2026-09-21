"""NKI (AWS Trainium) implementation of fused_activation: ``y = silu(x * gate + bias)``.

Design (mirrors the Triton / cuTile kernels):

* The flat input is processed in fixed-size blocks. A Triton program handles
  ``BLOCK_SIZE`` contiguous elements; the NKI analogue is one ``[128, BLOCK_SIZE]``
  SBUF tile -- 128 partitions (the 128 parallel lanes of the Vector/Scalar
  engines) times ``BLOCK_SIZE`` contiguous elements per partition. The default
  and the autotune search space use the same ``BLOCK_SIZE`` values as
  ``impl_triton.py`` / ``impl_cutile.py`` (default 1024; search 512/1024/2048).
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
    def fused_activation_kernel(x_input, gate_input, bias_input, block_size):
        """``out[i] = silu(x[i] * gate[i] + bias[i])`` over flat ``(n,)`` HBM tensors.

        Args:
            x_input, gate_input, bias_input: flat ``(n,)`` tensors in HBM (same dtype).
            block_size: elements per partition per tile (compile-time constant).
        """
        n = x_input.shape[0]
        out = nl.ndarray((n,), dtype=x_input.dtype, buffer=nl.shared_hbm)

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
                _tile_body(PMAX, q, start, x_input, gate_input, bias_input, out)
            if r > 0:
                _tile_body(1, r, start + q * PMAX, x_input, gate_input, bias_input, out)
        return out

    def _tile_body(p, f, start, x_input, gate_input, bias_input, out):
        """y = silu(x * gate + bias) on the ``[p, f]`` tile at flat offset ``start``."""
        x_tile = nl.ndarray((p, f), dtype=x_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x_input.ap(pattern=[[f, p], [1, f]], offset=start))
        gate_tile = nl.ndarray((p, f), dtype=gate_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=gate_tile, src=gate_input.ap(pattern=[[f, p], [1, f]], offset=start))
        bias_tile = nl.ndarray((p, f), dtype=bias_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_tile, src=bias_input.ap(pattern=[[f, p], [1, f]], offset=start))
        nisa.tensor_tensor(dst=x_tile, data1=x_tile, data2=gate_tile, op=nl.multiply)
        nisa.tensor_tensor(dst=x_tile, data1=x_tile, data2=bias_tile, op=nl.add)
        nisa.activation(dst=x_tile, op=nl.silu, data=x_tile)
        nisa.dma_copy(dst=out.ap(pattern=[[f, p], [1, f]], offset=start), src=x_tile)



# Same block sizes as impl_triton.py / impl_cutile.py (default 1024; search 512/1024/2048).
_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048)]
_kernel = fused_activation_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x_flat, gate_flat, bias_flat = x.reshape(-1), gate.reshape(-1), bias.reshape(-1)
    n = x_flat.numel()
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, str(x_flat.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_flat, gate_flat, bias_flat, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = _kernel(x_flat, gate_flat, bias_flat, cfg.block_size)
    return result if x.dim() == 1 else result.reshape(x.shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
