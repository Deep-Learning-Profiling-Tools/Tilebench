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
        dtype = img.dtype
        split = dtype == nl.float32
        mm_dtype = nl.bfloat16 if split else dtype
        pr, pc = kernel_rows // 2, kernel_cols // 2
        RT = PMAX - (kernel_rows - 1)
        Wn = W + kernel_cols - 1
        n_taps = kernel_rows * kernel_cols
        out = nl.ndarray((H * W,), dtype=dtype, buffer=nl.shared_hbm)

        w_raw = nl.ndarray((PMAX, n_taps), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_raw, src=w.ap(pattern=[[0, PMAX], [1, n_taps]], offset=0))
        w_b = nl.ndarray((PMAX, n_taps), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_b, src=w_raw)

        idx = nl.ndarray((PMAX, RT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=idx, pattern=[[-1, RT]], offset=0, channel_multiplier=1)
        eq = []
        for i in range(kernel_rows):
            e = nl.ndarray((PMAX, RT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=e, data=idx, op0=nl.equal, operand0=float(i))
            eq.append(e)

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

        n_tiles = (H + RT - 1) // RT
        n_col_blocks = (W + block_size - 1) // block_size
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            R0 = ti * RT
            M = min(RT, H - R0)
            K = M + kernel_rows - 1
            r_base = R0 - pr
            k_lo = max(0, -r_base)
            k_hi = min(K, H - r_base)
            win_in = nl.ndarray((PMAX, Wn), dtype=dtype, buffer=nl.sbuf)
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
        hi32 = nl.ndarray(x32.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hi32, src=x_hi)
        lo32 = nl.ndarray(x32.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=lo32, data1=x32, data2=hi32, op=nl.subtract)
        nisa.tensor_copy(dst=dst_lo, src=lo32)


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
    return _kernel(img, kernel, input_rows, input_cols, kernel_rows, kernel_cols, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
