"""NKI (AWS Trainium) implementation of gaussian_blur: zero-padded 2-D convolution with a
``kernel_rows x kernel_cols`` tap set (single channel), ``out = conv2d(img, w, padding=k//2)``.

Design (mirrors the Triton / cuTile kernels' blocking, mapped onto the Tensor engine):

* Triton computes one ``[BLOCK_R, BLOCK_C]`` output block per program by
  accumulating the ``kernel_rows * kernel_cols`` shifted-and-masked input loads,
  each scaled by its tap weight. NKI computes ``[RT, BLOCK_C]`` output blocks
  with exactly the same arithmetic -- ``sum_{i,j} w[i, j] * img[r + i - pr, c + j - pc]``
  -- but issues it to the Tensor engine: for every kernel column ``j`` the
  vertical taps form a banded ``[128, RT]`` matrix ``B_j`` (``B_j[k, m] = w[k - m, j]``)
  and ``out_block = sum_j B_j^T @ window_j``, where ``window_j`` is the
  ``[128, BLOCK_C]`` input row band shifted by ``j`` columns (an SBUF view, no
  reload). ``RT = 128 - (kernel_rows - 1)`` output rows per tile (122 for 7 taps)
  makes the contraction exactly the 128 SBUF partitions, so one output block
  costs ``kernel_cols`` matmuls instead of ``kernel_rows * kernel_cols``
  Vector-engine multiply-adds. The ``kernel_cols`` contributions accumulate in
  fp32 PSUM. Zero padding is realised by zeroing the halo rows / columns of the
  SBUF window (no host-side padding).
* fp16 inputs are multiplied natively. fp32 operands are split into a bf16
  ``hi`` part and a bf16 ``lo = x - hi`` remainder and multiplied as
  ``hi*hi + lo*hi + hi*lo`` (three bf16 matmuls, fp32 accumulation, ~1e-5
  relative error), which is 25% cheaper on the Tensor engine than the native
  4-pass fp32 matmul.
* ``BLOCK_C`` search space follows ``impl_triton.py`` (64/128/256/512); the
  default is 512 (= one fp32 PSUM bank, the largest matmul moving tile).
* Row tiles are split across the NeuronCores of the logical core (LNC2 on trn2)
  by ``nl.program_id(0)``; the band matrices are built once per launch.
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
    def gaussian_blur_kernel(img, w, H, W, kernel_rows, kernel_cols, block_size):
        """``out[r, c] = sum_{i,j} w[i*kc + j] * img[r + i - kr//2, c + j - kc//2]`` (zero padded).

        Args:
            img: flat ``[H * W]`` row-major image in HBM (fp16 / fp32); addressed through
                ``.ap()`` patterns with row stride ``W`` (no host-side reshape, which XLA
                would materialise as an extra copy pass).
            w: flat ``[kernel_rows * kernel_cols]`` tap weights in HBM (same dtype).
            H, W, kernel_rows, kernel_cols: image / tap-set extents (compile-time constants).
            block_size: output columns per block (``BLOCK_C`` <= 512, compile-time constant).
        """
        dtype = img.dtype
        # fp32 operands are split into bf16 hi + lo parts (3 bf16 matmuls per tap column).
        split = dtype == nl.float32
        mm_dtype = nl.bfloat16 if split else dtype
        pr, pc = kernel_rows // 2, kernel_cols // 2
        RT = PMAX - (kernel_rows - 1)             # output rows per tile: K = RT + kernel_rows - 1 = 128
        Wn = W + kernel_cols - 1                  # window width incl. column halo
        n_taps = kernel_rows * kernel_cols
        out = nl.ndarray((H * W,), dtype=dtype, buffer=nl.shared_hbm)

        # ---- tap weights replicated to all partitions, fp32 [128, n_taps] ----
        w_raw = nl.ndarray((PMAX, n_taps), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_raw, src=w.ap(pattern=[[0, PMAX], [1, n_taps]], offset=0))
        w_b = nl.ndarray((PMAX, n_taps), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_b, src=w_raw)

        # ---- band masks eq_i[k, m] = (k - m == i), k = window row (partition), m = output row ----
        idx = nl.ndarray((PMAX, RT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=idx, pattern=[[-1, RT]], offset=0, channel_multiplier=1)          # k - m
        eq = []
        for i in range(kernel_rows):
            e = nl.ndarray((PMAX, RT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=e, data=idx, op0=nl.equal, operand0=float(i))
            eq.append(e)

        # ---- B_j = sum_i w[i, j] * eq_i  (matmul stationary operands) ----
        B = []
        for j in range(kernel_cols):
            b32 = nl.ndarray((PMAX, RT), dtype=nl.float32, buffer=nl.sbuf)
            for i in range(kernel_rows):
                t = i * kernel_cols + j
                if i == 0:
                    nisa.tensor_scalar(dst=b32, data=eq[i], op0=nl.multiply, operand0=w_b[0:PMAX, t:t + 1])
                else:
                    nisa.scalar_tensor_tensor(dst=b32, data=eq[i], op0=nl.multiply,
                                              operand0=w_b[0:PMAX, t:t + 1], op1=nl.add, operand1=b32)
            b = nl.ndarray((PMAX, RT), dtype=mm_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=b, src=b32)
            if split:
                b_lo = nl.ndarray((PMAX, RT), dtype=nl.bfloat16, buffer=nl.sbuf)
                _bf16_remainder(b_lo, b32, b)
                B.append((b, b_lo))
            else:
                B.append((b, None))

        # ---- output row tiles, split across programs ----
        n_tiles = (H + RT - 1) // RT
        n_col_blocks = (W + block_size - 1) // block_size
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            R0 = ti * RT
            M = min(RT, H - R0)
            K = M + kernel_rows - 1               # window rows used by this tile (<= 128)
            # Window row k <-> image row R0 - pr + k, k in [0, K).
            r_base = R0 - pr
            k_lo = max(0, -r_base)
            k_hi = min(K, H - r_base)
            win_in = nl.ndarray((PMAX, Wn), dtype=dtype, buffer=nl.sbuf)
            # Compute-engine accesses must start at partition 0 (or 32/64/96), so the
            # invalid rows of a boundary tile are zeroed by clearing the whole tile before
            # the load; interior tiles only zero the column halo (full partition range).
            if k_lo > 0 or k_hi < K or K < PMAX:
                nisa.memset(dst=win_in, value=0.0)
            else:
                nisa.memset(dst=win_in[0:PMAX, 0:pc], value=0.0)
                nisa.memset(dst=win_in[0:PMAX, pc + W:Wn], value=0.0)
            if k_hi > k_lo:
                nisa.dma_copy(dst=win_in[k_lo:k_hi, pc:pc + W],
                              src=img.ap(pattern=[[W, k_hi - k_lo], [1, W]], offset=(r_base + k_lo) * W))
            if split:
                win = nl.ndarray((PMAX, Wn), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=win, src=win_in)
                win_lo = nl.ndarray((PMAX, Wn), dtype=nl.bfloat16, buffer=nl.sbuf)
                _bf16_remainder(win_lo, win_in, win)
            else:
                win = win_in
                win_lo = None
            for cb in range(n_col_blocks):
                C0 = cb * block_size
                N = min(block_size, W - C0)
                acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
                for j in range(kernel_cols):
                    # out[m, c] += sum_k B_j[k, m] * window[k, c + j]  (window col c + j = image col c + j - pc)
                    b_hi, b_lo = B[j]
                    nisa.nc_matmul(dst=acc, stationary=b_hi[0:K, 0:M],
                                   moving=win[0:K, C0 + j:C0 + j + N], accumulate=(j > 0))
                    if split:
                        nisa.nc_matmul(dst=acc, stationary=b_lo[0:K, 0:M],
                                       moving=win[0:K, C0 + j:C0 + j + N], accumulate=True)
                        nisa.nc_matmul(dst=acc, stationary=b_hi[0:K, 0:M],
                                       moving=win_lo[0:K, C0 + j:C0 + j + N], accumulate=True)
                o_tile = nl.ndarray((M, N), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_tile, src=acc)
                nisa.dma_copy(dst=out.ap(pattern=[[W, M], [1, N]], offset=R0 * W + C0), src=o_tile)
        return out

    def _bf16_remainder(dst_lo, x32, x_hi):
        """``dst_lo = bf16(x32 - fp32(x_hi))``: the bf16 remainder of the fp32 tile ``x32``."""
        hi32 = nl.ndarray(x32.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hi32, src=x_hi)
        lo32 = nl.ndarray(x32.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=lo32, data1=x32, data2=hi32, op=nl.subtract)
        nisa.tensor_copy(dst=dst_lo, src=lo32)


# Output columns per block: Triton's BLOCK_C values (64/128/256/512); default 512 (one PSUM bank).
_DEFAULT_CONFIG = SimpleNamespace(block_size=512)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (64, 128, 256, 512)]
_kernel = gaussian_blur_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, kernel: torch.Tensor, input_rows: int, input_cols: int,
        kernel_rows: int, kernel_cols: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if kernel_rows > PMAX or kernel_rows < 1 or kernel_cols < 1:
        raise NotImplementedError("gaussian_blur NKI: kernel_rows must be in [1, 128]")
    if input.numel() != input_rows * input_cols:
        raise ValueError("gaussian_blur NKI: input must hold input_rows * input_cols elements")
    img = input.reshape(-1) if input.dim() != 1 else input
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=((input_rows, input_cols), (kernel_rows, kernel_cols), str(input.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (img, kernel, input_rows, input_cols, kernel_rows, kernel_cols, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    # Flat in, flat out: the benchmark hands over a flat image and expects a flat result.
    return _kernel(img, kernel, input_rows, input_cols, kernel_rows, kernel_cols, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
