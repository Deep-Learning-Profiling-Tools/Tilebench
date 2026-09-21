"""NKI (AWS Trainium) implementation of jacobi_stencil_2d: 5-point average with pass-through borders.

Design (mirrors the Triton / cuTile kernels):

* Triton launches one program per ``[BLOCK_SIZE_R, BLOCK_SIZE_C]`` output block,
  loads the four shifted neighbours + the centre and writes
  ``where(interior, 0.25 * (up + down + left + right), centre)``. NKI puts 128
  output rows on the 128 SBUF partitions and walks the same
  ``BLOCK_SIZE_C``-column blocks: the centre row window of the block (plus one
  halo column on each side) is one HBM DMA; the up / down windows are the same
  rows shifted by one partition, built from two SBUF -> SBUF DMAs plus one
  halo row each from HBM, so every input element is read from HBM once. The
  four neighbours are shifted SBUF views of those windows (no extra loads), the sum
  and ``* 0.25`` run on the Vector engine, and the two border columns are
  copied through before one DMA store. Border rows are pure HBM -> HBM copies.
  ``BLOCK_SIZE_C`` defaults / search space are those of ``impl_triton.py`` /
  ``impl_cutile.py`` (default 1024; search 256/512/1024/2048); rows per tile is
  the 128-partition hardware width (Triton's ``BLOCK_SIZE_R``).
* Interior row tiles are split across the NeuronCores of the logical core
  (LNC2 on trn2) by ``nl.program_id(0)``.
* The input is read in place with clamped windows; no host-side padding.
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
    def jacobi_kernel(x, block_size):
        """5-point Jacobi step on a ``[rows, cols]`` HBM grid (borders pass through).

        Args:
            x: ``[rows, cols]`` tensor in HBM (fp16 / bf16 / fp32).
            block_size: output columns per block (``BLOCK_SIZE_C``, compile-time constant).
        """
        rows, cols = x.shape
        dtype = x.dtype
        out = nl.ndarray((rows, cols), dtype=dtype, buffer=nl.shared_hbm)
        pid = nl.program_id(0)
        num_programs = nl.num_programs()

        # ---- border rows: HBM -> HBM copies (program 0 only) ----
        if pid == 0:
            nisa.dma_copy(dst=out[0:1, 0:cols], src=x[0:1, 0:cols])
            if rows > 1:
                nisa.dma_copy(dst=out[rows - 1:rows, 0:cols], src=x[rows - 1:rows, 0:cols])

        # ---- interior rows 1 .. rows-2, 128 per tile, tiles split across programs ----
        n_interior = max(0, rows - 2)
        n_tiles = (n_interior + PMAX - 1) // PMAX
        per_core = (n_tiles + num_programs - 1) // num_programs
        n_blocks = (cols + block_size - 1) // block_size
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = 1 + ti * PMAX                 # first output row of the tile
            rs = min(PMAX, rows - 1 - r0)      # rows in the tile (all interior)
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, cols - c0)
                # Window columns [wl, wr): the block plus one halo column each side (clamped).
                wl = max(c0 - 1, 0)
                wr = min(c0 + cs + 1, cols)
                ws = wr - wl
                ctr = nl.ndarray((rs, ws), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=ctr, src=x[r0:r0 + rs, wl:wr])
                # up[p] = row r0 - 1 + p and dn[p] = row r0 + 1 + p are the centre rows shifted by
                # one partition: only the two halo rows come from HBM, the rest are SBUF -> SBUF
                # DMA copies (compute engines cannot read a tile at a partition offset, DMAs can).
                up = nl.ndarray((rs, ws), dtype=dtype, buffer=nl.sbuf)
                dn = nl.ndarray((rs, ws), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=up[0:1, 0:ws], src=x[r0 - 1:r0, wl:wr])
                if rs > 1:
                    nisa.dma_copy(dst=up[1:rs, 0:ws], src=ctr[0:rs - 1, 0:ws])
                    nisa.dma_copy(dst=dn[0:rs - 1, 0:ws], src=ctr[1:rs, 0:ws])
                nisa.dma_copy(dst=dn[rs - 1:rs, 0:ws], src=x[r0 + rs:r0 + rs + 1, wl:wr])
                result = nl.ndarray((rs, cs), dtype=dtype, buffer=nl.sbuf)
                # Interior columns of this block: global [ci0, ci1).
                ci0 = max(c0, 1)
                ci1 = min(c0 + cs, cols - 1)
                if ci1 > ci0:
                    n_int = ci1 - ci0
                    a = ci0 - wl                       # window offset of column ci0
                    o = ci0 - c0                       # result offset of column ci0
                    acc = nl.ndarray((rs, n_int), dtype=nl.float32, buffer=nl.sbuf)
                    # up + down + left + right, in the reference's evaluation order
                    nisa.tensor_tensor(dst=acc, data1=up[0:rs, a:a + n_int],
                                       data2=dn[0:rs, a:a + n_int], op=nl.add)
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=ctr[0:rs, a - 1:a - 1 + n_int], op=nl.add)
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=ctr[0:rs, a + 1:a + 1 + n_int], op=nl.add)
                    nisa.tensor_scalar(dst=result[0:rs, o:o + n_int], data=acc, op0=nl.multiply, operand0=0.25)
                # Border columns pass through unchanged.
                if c0 == 0:
                    nisa.tensor_copy(dst=result[0:rs, 0:1], src=ctr[0:rs, 0:1])
                if c0 + cs == cols and cols > 1:
                    nisa.tensor_copy(dst=result[0:rs, cs - 1:cs], src=ctr[0:rs, ws - 1:ws])
                nisa.dma_copy(dst=out[r0:r0 + rs, c0:c0 + cs], src=result)
        return out


# Same column block sizes as impl_triton.py / impl_cutile.py (BLOCK_SIZE_C default 1024; search 256..2048).
_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (256, 512, 1024, 2048)]
_kernel = jacobi_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, rows: int, cols: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if rows < 1 or cols < 1 or tuple(input.shape) != (rows, cols):
        raise ValueError("jacobi_stencil_2d NKI: input must be a (rows, cols) grid with rows, cols >= 1")
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=((rows, cols), str(input.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (input, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(input, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
