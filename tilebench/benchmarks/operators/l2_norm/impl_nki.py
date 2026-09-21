"""NKI (AWS Trainium) implementation of l2_norm: ``y = x * rsqrt(sum(x^2, -1) + eps)``.

Design (mirrors the Triton / cuTile kernels):

* Triton runs one program per row with two passes over ``BLOCK_N``-column blocks:
  pass 1 accumulates ``sum(x^2)`` in fp32, pass 2 reloads each block (from L2 on
  a GPU) and scales it by ``rstd``. NKI puts 128 rows on the 128 SBUF partitions
  and does the same two passes over the same column blocks:
  - pass 1: one ``[128, BLOCK_N]`` DMA + one fused Scalar-engine
    ``activation_reduce(square, sum)`` per block (fp32 partials);
  - ``rstd = rsqrt(sum + eps)`` on the ``[128, 1]`` column (GpSimd rsqrt, fp32 exact);
  - pass 2: ``x * rstd`` (per-partition scalar broadcast) on each block, store.
  The row tile's blocks stay resident in SBUF between the passes whenever the
  row fits (``n_cols * itemsize <= 64 KiB`` per partition, i.e. every benchmark
  case), so each element is read from HBM once -- the SBUF plays the role of
  the GPU L2 for Triton's reload. Longer rows fall back to reloading pass 2.
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


def _itemsize(dtype) -> int:
    return 4 if dtype == nl.float32 else 2


if nki is not None:
    @nki.jit
    def l2_norm_kernel(a_input, eps, block_size):
        """Row-wise L2 normalisation of a ``[n_rows, n_cols]`` HBM tensor.

        Args:
            a_input: ``[n_rows, n_cols]`` tensor in HBM (fp16 / bf16 / fp32).
            eps: epsilon added to the squared norm (compile-time constant).
            block_size: columns per block (``BLOCK_N``, compile-time constant).
        """
        # Any leading batch dims are folded into the rows; the kernel addresses the tensor
        # through flat access patterns so no host-side reshape (an extra XLA copy) is needed.
        n_cols = a_input.shape[-1]
        n_rows = 1
        for d in a_input.shape[:-1]:
            n_rows *= d
        out = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
        n_blocks = (n_cols + block_size - 1) // block_size
        # Keep a row tile resident in SBUF between the passes when it is small enough.
        keep = n_cols * _itemsize(a_input.dtype) <= 64 * 1024

        n_tiles = (n_rows + PMAX - 1) // PMAX
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            r0 = ti * PMAX
            rs = min(PMAX, n_rows - r0)
            # ---- pass 1: sum(x^2) per row, one fp32 partial per column block ----
            parts = nl.ndarray((rs, n_blocks), dtype=nl.float32, buffer=nl.sbuf)
            x_tiles = []
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, n_cols - c0)
                x_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input.ap(pattern=[[n_cols, rs], [1, cs]], offset=r0 * n_cols + c0))
                if keep:
                    x_tiles.append(x_tile)
                sq_tile = nl.ndarray((rs, cs), dtype=nl.float32, buffer=nl.sbuf)
                # square + free-axis sum in one Scalar-engine instruction (fp32 accumulate).
                nisa.activation_reduce(dst=sq_tile, op=nl.square, data=x_tile,
                                       reduce_op=nl.add, reduce_res=parts[0:rs, cb:cb + 1])
            sum_sq = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            if n_blocks == 1:
                nisa.tensor_scalar(dst=sum_sq, data=parts, op0=nl.add, operand0=eps)
            else:
                nisa.tensor_reduce(dst=sum_sq, op=nl.add, data=parts, axis=(1,))
                nisa.tensor_scalar(dst=sum_sq, data=sum_sq, op0=nl.add, operand0=eps)
            # rstd = 1 / sqrt(sum_sq): the GpSimd rsqrt is fp32-exact (~1e-7 rel), unlike the
            # Scalar-engine approximation (~4e-5), which would fail the fp32 tolerance.
            rstd = nl.ndarray((rs, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=sum_sq, op0=nl.rsqrt, operand0=None)
            # ---- pass 2: y = x * rstd ----
            for cb in range(n_blocks):
                c0 = cb * block_size
                cs = min(block_size, n_cols - c0)
                if keep:
                    x_tile = x_tiles[cb]
                else:
                    x_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=x_tile, src=a_input.ap(pattern=[[n_cols, rs], [1, cs]], offset=r0 * n_cols + c0))
                y_tile = nl.ndarray((rs, cs), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=rstd)
                nisa.dma_copy(dst=out.ap(pattern=[[n_cols, rs], [1, cs]], offset=r0 * n_cols + c0), src=y_tile)
        return out


# Same column block sizes as impl_triton.py (BLOCK_N default 1024; search 512/1024/2048).
_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048)]
_kernel = l2_norm_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, eps: float = 1e-6, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("l2_norm NKI: int8 not supported")
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype), float(eps)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x, float(eps), cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    # The kernel takes the (batch, M, K) tensor as-is (no host reshape / XLA copy).
    return _kernel(x, float(eps), cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
