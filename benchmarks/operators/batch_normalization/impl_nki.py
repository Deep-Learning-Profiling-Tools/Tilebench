"""NKI (AWS Trainium) implementation of batch_normalization (training-mode batch statistics).

``y[n, c] = (x[n, c] - mean[c]) * rstd[c] * gamma[c] + beta[c]`` with per-channel
mean / biased variance over the batch axis ``N`` (``F.batch_norm(..., training=True)``).

Design (mirrors the Triton / cuTile kernels):

* Triton uses three kernels: (1) per-row-block partial ``sum`` / ``sum(x^2)``
  over ``BLOCK_N`` rows, (2) ``mean`` / ``inv_std`` per channel from the block
  partials, (3) ``y = x * scale + shift`` over ``ROWS``-row blocks with
  ``scale = inv_std * gamma`` and ``shift = beta - mean * scale``.
  NKI does the same three stages with 128 rows (the SBUF partitions) per block:
  1. the per-block partial sums over rows are a ``ones[128, 1]^T @ x_block``
     matmul on the Tensor engine (``x^2`` from a Scalar-engine ``square``),
     accumulated across all row blocks in PSUM in fp32 -- i.e. the block partials
     and their reduction, in ``BLOCK_N``-column slices (one PSUM bank each);
  2. ``mean``, ``var = max(E[x^2] - mean^2, 0)``, ``rstd``, ``scale``, ``shift`` on
     ``[1, C]`` rows; ``scale`` / ``shift`` are broadcast to all 128 partitions
     with a ``ones[1, 128]^T @ row`` matmul;
  3. ``y = x * scale + shift`` per 128-row block on the Vector engine.
  The NKI tunable is the PSUM column block ``BLOCK_N`` (default 512 = one fp32
  PSUM bank; search 128/256/512); rows per block is the 128-partition hardware
  width (Triton's ``ROWS`` / ``BLOCK_N`` row blocking).
* Under LNC2 the row blocks are split between the two program instances in
  both passes: each core accumulates the partial sums of its own row blocks,
  swaps its ``[1, 2C]`` partial row with the other core (``sendrecv``), and each
  reduces both rows to the full batch statistics (stage 2) before applying
  stage 3 to its own row blocks. Every
  element of ``x`` is therefore read from HBM twice in total (once per pass),
  as in Triton, instead of three times.
* Partial row blocks / column blocks are clamped with ``min()``; nothing is
  padded on the host.
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
    def batch_norm_kernel(x_hbm, gamma_hbm, beta_hbm, eps, block_size):
        """Batch norm of ``[N, C]`` ``x_hbm`` with ``[1, C]`` gamma / beta -> ``[N, C]``.

        Args:
            x_hbm: ``[N, C]`` tensor in HBM (fp16 / bf16 / fp32).
            gamma_hbm, beta_hbm: ``[1, C]`` per-channel scale / shift in HBM.
            eps: variance epsilon (compile-time constant).
            block_size: PSUM column block (``BLOCK_N`` <= 512, compile-time constant).
        """
        N, C = x_hbm.shape
        out = nl.ndarray((N, C), dtype=x_hbm.dtype, buffer=nl.shared_hbm)
        n_row_tiles = (N + PMAX - 1) // PMAX
        n_col_blocks = (C + block_size - 1) // block_size
        num_programs = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (n_row_tiles + num_programs - 1) // num_programs
        t_lo = pid * per_core
        t_hi = min(n_row_tiles, (pid + 1) * per_core)
        my_tiles = range(t_lo, t_hi)

        # ---- gamma / beta -> fp32 rows on partition 0 ----
        gamma_raw = nl.ndarray((1, C), dtype=gamma_hbm.dtype, buffer=nl.sbuf)
        beta_raw = nl.ndarray((1, C), dtype=beta_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=gamma_raw, src=gamma_hbm[0:1, 0:C])
        nisa.dma_copy(dst=beta_raw, src=beta_hbm[0:1, 0:C])
        gamma_f32 = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        beta_f32 = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gamma_f32, src=gamma_raw)
        nisa.tensor_copy(dst=beta_f32, src=beta_raw)
        scale_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        shift_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)

        if N == 1:
            # Batch variance is zero with a single sample: y = beta.
            nisa.memset(dst=scale_row, value=0.0)
            nisa.tensor_copy(dst=shift_row, src=beta_f32)
        else:
            # ---- stage 1: sum(x) and sum(x^2) over N via ones^T @ x on the Tensor engine ----
            ones_col = nl.ndarray((PMAX, 1), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.memset(dst=ones_col, value=1.0)
            ones_col_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_col_f32, value=1.0)
            psum_sum = []
            psum_sq = []
            for cb in range(n_col_blocks):
                cs = min(block_size, C - cb * block_size)
                psum_sum.append(nl.ndarray((1, cs), dtype=nl.float32, buffer=nl.psum))
                psum_sq.append(nl.ndarray((1, cs), dtype=nl.float32, buffer=nl.psum))
            for rt in my_tiles:
                r0 = rt * PMAX
                rs = min(PMAX, N - r0)
                x_tile = nl.ndarray((rs, C), dtype=x_hbm.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=x_hbm[r0:r0 + rs, 0:C])
                sq_tile = nl.ndarray((rs, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sq_tile, op=nl.square, data=x_tile)
                for cb in range(n_col_blocks):
                    c0 = cb * block_size
                    cs = min(block_size, C - c0)
                    # [1, cs] += ones[rs, 1]^T @ x[rs, cs]  (fp32 PSUM accumulation over row tiles)
                    nisa.nc_matmul(dst=psum_sum[cb], stationary=ones_col[0:rs, 0:1],
                                   moving=x_tile[0:rs, c0:c0 + cs], accumulate=(rt > t_lo))
                    nisa.nc_matmul(dst=psum_sq[cb], stationary=ones_col_f32[0:rs, 0:1],
                                   moving=sq_tile[0:rs, c0:c0 + cs], accumulate=(rt > t_lo))
            # This program's partial [sum | sum(x^2)] row.
            part = nl.ndarray((1, 2 * C), dtype=nl.float32, buffer=nl.sbuf)
            if t_hi <= t_lo:
                nisa.memset(dst=part, value=0.0)
            else:
                for cb in range(n_col_blocks):
                    c0 = cb * block_size
                    cs = min(block_size, C - c0)
                    nisa.tensor_copy(dst=part[0:1, c0:c0 + cs], src=psum_sum[cb])
                    nisa.tensor_copy(dst=part[0:1, C + c0:C + c0 + cs], src=psum_sq[cb])
            if num_programs == 1:
                total = part
            else:
                # Swap partial rows with the other core (SBUF -> SBUF point-to-point DMA, self-synchronising).
                other = nl.ndarray((1, 2 * C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.sendrecv(src=part, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
                total = nl.ndarray((1, 2 * C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=total, data1=part, data2=other, op=nl.add)
            total_sum = total[0:1, 0:C]
            total_sq = total[0:1, C:2 * C]
            # ---- stage 2: mean, var, rstd, scale, shift on [1, C] rows ----
            mean_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            mean_sq_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mean_row, data=total_sum, op0=nl.multiply, operand0=1.0 / N)
            nisa.tensor_scalar(dst=mean_sq_row, data=total_sq, op0=nl.multiply, operand0=1.0 / N)
            var_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=var_row, data1=mean_row, data2=mean_row, op=nl.multiply)
            nisa.tensor_tensor(dst=var_row, data1=mean_sq_row, data2=var_row, op=nl.subtract)
            # var = max(var, 0) + eps  (guard against tiny negative round-off)
            nisa.tensor_scalar(dst=var_row, data=var_row, op0=nl.maximum, operand0=0.0,
                               op1=nl.add, operand1=eps)
            rstd_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd_row, data=var_row, op0=nl.rsqrt, operand0=None)
            nisa.tensor_tensor(dst=scale_row, data1=rstd_row, data2=gamma_f32, op=nl.multiply)
            nisa.tensor_tensor(dst=shift_row, data1=mean_row, data2=scale_row, op=nl.multiply)
            nisa.tensor_tensor(dst=shift_row, data1=beta_f32, data2=shift_row, op=nl.subtract)

        # ---- broadcast scale / shift from partition 0 to all 128 partitions (ones^T @ row) ----
        ones_row = nl.ndarray((1, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)
        scale_bcast = nl.ndarray((PMAX, C), dtype=nl.float32, buffer=nl.sbuf)
        shift_bcast = nl.ndarray((PMAX, C), dtype=nl.float32, buffer=nl.sbuf)
        for cb in range(n_col_blocks):
            c0 = cb * block_size
            cs = min(block_size, C - c0)
            ps_scale = nl.ndarray((PMAX, cs), dtype=nl.float32, buffer=nl.psum)
            ps_shift = nl.ndarray((PMAX, cs), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_scale, stationary=ones_row, moving=scale_row[0:1, c0:c0 + cs])
            nisa.nc_matmul(dst=ps_shift, stationary=ones_row, moving=shift_row[0:1, c0:c0 + cs])
            nisa.tensor_copy(dst=scale_bcast[0:PMAX, c0:c0 + cs], src=ps_scale)
            nisa.tensor_copy(dst=shift_bcast[0:PMAX, c0:c0 + cs], src=ps_shift)

        # ---- stage 3: y = x * scale + shift, row tiles split across programs ----
        for rt in my_tiles:
            r0 = rt * PMAX
            rs = min(PMAX, N - r0)
            x_tile = nl.ndarray((rs, C), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x_hbm[r0:r0 + rs, 0:C])
            scaled = nl.ndarray((rs, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=scaled, data1=x_tile, data2=scale_bcast[0:rs, 0:C], op=nl.multiply)
            y_tile = nl.ndarray((rs, C), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y_tile, data1=scaled, data2=shift_bcast[0:rs, 0:C], op=nl.add)
            nisa.dma_copy(dst=out[r0:r0 + rs, 0:C], src=y_tile)
        return out


# PSUM column block: default 512 (one fp32 PSUM bank); search 128/256/512.
_DEFAULT_CONFIG = SimpleNamespace(block_size=512)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (128, 256, 512)]
_kernel = batch_norm_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if input.dtype == torch.int8:
        raise NotImplementedError("batch_normalization NKI: int8 not supported")
    gamma_2d = gamma.reshape(1, -1)
    beta_2d = beta.reshape(1, -1)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(input.shape), str(input.dtype), float(eps)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (input, gamma_2d, beta_2d, float(eps), cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(input, gamma_2d, beta_2d, float(eps), cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
