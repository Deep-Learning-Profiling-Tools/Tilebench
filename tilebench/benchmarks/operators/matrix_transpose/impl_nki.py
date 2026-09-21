"""NKI (AWS Trainium) implementation of matrix_transpose: ``out[n, m] = in[m, n]``.

Design (mirrors the Triton / cuTile kernels' tiled load / store):

* Triton reads ``BLOCK_TILE x BLOCK_TILE`` blocks and writes them transposed.
  NKI tiles the input into ``[BLOCK_SIZE rows, 128 cols]`` blocks: on Trainium2
  the DMA engines can transpose 2-byte / 4-byte elements on the fly
  (``nisa.dma_transpose``), so each block is one HBM -> SBUF transposing DMA
  into a ``[128, BLOCK_SIZE]`` tile (128 output rows on the partitions, a
  ``BLOCK_SIZE``-element contiguous run per row) followed by one contiguous
  HBM store. No compute engine is involved, exactly like the Triton / cuTile
  load-store kernels. ``BLOCK_SIZE`` default 2048; search 1024/2048/4096
  (Triton's square 32/64/128 tiles map to the 128-partition x ``BLOCK_SIZE``
  DMA tiles here).
* 1-byte dtypes (int8) cannot be transposed by the DMA engines; they take the
  Tensor-engine path: a ``[128, BLOCK_SIZE]`` input band is widened to bf16
  (lossless for 8-bit integers), transposed 128x128 block by block with
  ``nisa.nc_transpose`` into ``[128, BLOCK_SIZE]`` output bands, narrowed back
  and stored with the same contiguous DMAs.
* Column blocks of the input (= row blocks of the output) are split across the
  NeuronCores of the logical core (LNC2 on trn2) by ``nl.program_id(0)``.
  Boundary blocks are clamped with ``min()``; nothing is padded on the host.
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
    def transpose_kernel(a_input, block_size):
        """``out[n, m] = a_input[m, n]`` for a ``[m, n]`` HBM tensor.

        Args:
            a_input: ``[m, n]`` tensor in HBM (fp16 / bf16 / fp32 / int8).
            block_size: input rows per block (compile-time constant).
        """
        m, n = a_input.shape
        out = nl.ndarray((n, m), dtype=a_input.dtype, buffer=nl.shared_hbm)
        n_col_blocks = (n + PMAX - 1) // PMAX          # 128 input cols = 128 output rows
        n_row_blocks = (m + block_size - 1) // block_size
        dma_path = a_input.dtype not in (nl.int8, nl.uint8)

        num_programs = nl.num_programs()
        per_core = (n_col_blocks + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for cb in range(pid * per_core, min(n_col_blocks, (pid + 1) * per_core)):
            c0 = cb * PMAX
            cs = min(PMAX, n - c0)
            for rb in range(n_row_blocks):
                r0 = rb * block_size
                rs = min(block_size, m - r0)
                if dma_path:
                    # HBM [rs, cs] block -> transposed SBUF tile [cs, rs] in one DMA.
                    t_tile = nl.ndarray((cs, rs), dtype=a_input.dtype, buffer=nl.sbuf)
                    nisa.dma_transpose(dst=t_tile, src=a_input[r0:r0 + rs, c0:c0 + cs], axes=(1, 0))
                    nisa.dma_copy(dst=out[c0:c0 + cs, r0:r0 + rs], src=t_tile)
                else:
                    _transpose_block_int8(a_input, out, r0, rs, c0, cs)
        return out

    def _transpose_block_int8(a_input, out, r0, rs, c0, cs):
        """Tensor-engine transpose of the int8 block ``a_input[r0:r0+rs, c0:c0+cs]``.

        The block is loaded as up-to-128-row bands (contiguous DMAs), widened to
        bf16 (exact for 8-bit integers), transposed 128x128 sub-block by
        sub-block on the Tensor engine and narrowed back into one ``[cs, rs]``
        output tile that is stored with a single contiguous DMA.
        """
        t_tile = nl.ndarray((cs, rs), dtype=a_input.dtype, buffer=nl.sbuf)
        for bb in range((rs + PMAX - 1) // PMAX):
            b0 = bb * PMAX
            bs = min(PMAX, rs - b0)
            band = nl.ndarray((bs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=band, src=a_input[r0 + b0:r0 + b0 + bs, c0:c0 + cs])
            wide = nl.ndarray((bs, cs), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wide, src=band)
            ps = nl.ndarray((cs, bs), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=ps, data=wide)
            nisa.tensor_copy(dst=t_tile[0:cs, b0:b0 + bs], src=ps)
        nisa.dma_copy(dst=out[c0:c0 + cs, r0:r0 + rs], src=t_tile)


# Rows per DMA block (contiguous run per output row): default 2048; search 1024/2048/4096.
_DEFAULT_CONFIG = SimpleNamespace(block_size=2048)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048, 4096)]
_kernel = transpose_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
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
    return _kernel(x, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
