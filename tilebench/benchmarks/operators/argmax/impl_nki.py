"""NKI (AWS Trainium) implementation of argmax: ``out[r] = argmax(x[r, :])`` (first occurrence).

Design (mirrors the Triton / cuTile kernels):

* Triton runs one program per row and walks the row in ``BLOCK_N``-column
  chunks, keeping a running ``(max, idx)`` pair that a chunk only replaces when
  its maximum is *strictly* greater (first occurrence wins). NKI puts 128 rows on
  the 128 SBUF partitions and walks the same ``BLOCK_N`` chunks:
  - per chunk: one ``[128, BLOCK_N]`` DMA, ``nisa.max8`` (top-8 values) and
    ``nisa.nc_find_index8`` (first-occurrence index of each value) -- only the
    top-1 slot of each is used;
  - the per-chunk maxima / indices are then merged across chunks in one go:
    ``gmax = max(chunk_max)``, and among the chunks whose maximum equals ``gmax``
    the smallest global index is picked with a masked ``select_reduce(max)`` on
    the negated indices, which reproduces Triton's strict-greater tie-breaking.
  ``BLOCK_N`` defaults / search space are those of ``impl_triton.py`` /
  ``impl_cutile.py`` (default 256; search 256/512/1024/2048).
* Row tiles are split across the NeuronCores of the logical core (LNC2 on trn2)
  by ``nl.program_id(0)``. Partial row tiles / chunks are clamped with ``min()``.
* Index arithmetic is done in fp32 (exact below 2^24) and cast to int32 at the end;
  ``run`` widens to int64 to match ``torch.argmax``.
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

# ``nisa.max8`` / ``nisa.nc_find_index8`` need >= 8 elements per partition.
MAX8_MIN_ELEMS = 8
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
    def argmax_kernel(a_input, block_size):
        """Row-wise argmax of a ``[n_rows, n_cols]`` HBM tensor -> ``[n_rows, 1]`` int32.

        Args:
            a_input: ``[n_rows, n_cols]`` tensor in HBM (fp16 / bf16 / fp32).
            block_size: columns per chunk (``BLOCK_N``, compile-time constant).
        """
        n_rows, n_cols = a_input.shape
        out = nl.ndarray((n_rows, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        n_chunks = (n_cols + block_size - 1) // block_size

        n_tiles = (n_rows + PMAX - 1) // PMAX
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = ti * PMAX
            rs = min(PMAX, n_rows - r0)
            # Top-8 values / first-occurrence indices of every chunk, side by side.
            top8 = nl.ndarray((rs, n_chunks * 8), dtype=a_input.dtype, buffer=nl.sbuf)
            idx8 = nl.ndarray((rs, n_chunks * 8), dtype=nl.uint16, buffer=nl.sbuf)
            for c in range(n_chunks):
                f0 = c * block_size
                fs = min(block_size, n_cols - f0)
                # max8 needs >= 8 elements: a narrow tail chunk is padded with -inf *after*
                # the data, so the first-occurrence index always lands on a real element.
                tile_sz = max(fs, MAX8_MIN_ELEMS)
                a_tile = nl.ndarray((rs, tile_sz), dtype=a_input.dtype, buffer=nl.sbuf)
                if tile_sz != fs:
                    nisa.memset(dst=a_tile, value=float("-inf"))
                nisa.dma_copy(dst=a_tile[0:rs, 0:fs], src=a_input[r0:r0 + rs, f0:f0 + fs])
                nisa.max8(dst=top8[0:rs, 8 * c:8 * c + 8], src=a_tile)
                nisa.nc_find_index8(dst=idx8[0:rs, 8 * c:8 * c + 8], data=a_tile,
                                    vals=top8[0:rs, 8 * c:8 * c + 8])
            # ---- merge the chunks: gmax, then the smallest global index attaining it ----
            chunk_max = nl.ndarray((rs, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=chunk_max, src=top8[0:rs, 0:n_chunks * 8:8])
            chunk_idx = nl.ndarray((rs, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=chunk_idx, src=idx8[0:rs, 0:n_chunks * 8:8])
            # global index = local index + chunk start (chunk c starts at c * block_size)
            base = nl.ndarray((rs, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=base, pattern=[[block_size, n_chunks]], offset=0, channel_multiplier=0)
            neg_idx = nl.ndarray((rs, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=neg_idx, data=chunk_idx, op0=nl.add, operand0=base,
                                      op1=nl.multiply, operand1=None) if False else None
            nisa.tensor_tensor(dst=neg_idx, data1=chunk_idx, data2=base, op=nl.add)
            nisa.tensor_scalar(dst=neg_idx, data=neg_idx, op0=nl.multiply, operand0=-1.0)
            gmax = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=gmax, op=nl.maximum, data=chunk_max, axis=(1,))
            is_best = nl.ndarray((rs, n_chunks), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=is_best, data=chunk_max, op0=nl.equal, operand0=gmax)
            # max over -idx of the winning chunks == -(smallest winning index): first occurrence.
            scratch = nl.ndarray((rs, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
            neg_best = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=scratch, predicate=is_best, on_true=neg_idx, on_false=NEG_BIG,
                               reduce_res=neg_best, reduce_cmd=nisa.reduce_cmd.reset_reduce,
                               reduce_op=nl.maximum)
            best = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=best, data=neg_best, op0=nl.multiply, operand0=-1.0)
            best_i32 = nl.ndarray((rs, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=best_i32, src=best)
            nisa.dma_copy(dst=out[r0:r0 + rs, 0:1], src=best_i32)
        return out


# Same chunk sizes as impl_triton.py / impl_cutile.py (BLOCK_N default 256; search 256/512/1024/2048).
_DEFAULT_CONFIG = SimpleNamespace(block_size=256)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (256, 512, 1024, 2048)]
_kernel = argmax_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("argmax NKI: int8 not supported")
    if x.dim() != 2 or dim != 1:
        raise NotImplementedError("argmax NKI: expects a 2D input reduced over dim=1")
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
    return _kernel(x, cfg.block_size).reshape(-1).to(torch.int64)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
