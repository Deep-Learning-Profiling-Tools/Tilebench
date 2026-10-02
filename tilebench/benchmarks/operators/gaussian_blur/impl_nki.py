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
        pr, pc = kernel_rows // 2, kernel_cols // 2
        n_taps = kernel_rows * kernel_cols
        esz = 4 if dtype == nl.float32 else 2
        budget = nl.tile_size.sbuf_fmax_bytes // 2
        nb = max(1, min(block_size, W, budget // (esz * (kernel_rows + 1) + 4) - (kernel_cols - 1)))
        out = nl.ndarray((H * W,), dtype=dtype, buffer=nl.shared_hbm)

        w_raw = nl.ndarray((PMAX, n_taps), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_raw, src=w.ap(pattern=[[0, PMAX], [1, n_taps]], offset=0))
        w_b = nl.ndarray((PMAX, n_taps), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_b, src=w_raw)

        n_tiles = (H + PMAX - 1) // PMAX
        n_col_blocks = (W + nb - 1) // nb
        num_programs = nl.num_programs()
        per_core = (n_tiles + num_programs - 1) // num_programs
        pid = nl.program_id(0)
        for ti in range(pid * per_core, min(n_tiles, (pid + 1) * per_core)):
            R0 = ti * PMAX
            M = min(PMAX, H - R0)
            for cb in range(n_col_blocks):
                C0 = cb * nb
                N = min(nb, W - C0)
                Nw = N + kernel_cols - 1
                c_lo = max(0, pc - C0)
                c_hi = min(Nw, W - C0 + pc)
                acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
                o_tile = nl.ndarray((M, N), dtype=dtype, buffer=nl.sbuf)
                for i in range(kernel_rows):
                    r_base = R0 + i - pr
                    p_lo = max(0, -r_base)
                    p_hi = min(M, H - r_base)
                    win = nl.ndarray((M, Nw), dtype=dtype, buffer=nl.sbuf)
                    if p_lo > 0 or p_hi < M or c_lo > 0 or c_hi < Nw:
                        nisa.memset(dst=win, value=0.0)
                    if p_hi > p_lo:
                        nisa.dma_copy(dst=win[p_lo:p_hi, c_lo:c_hi],
                                      src=img.ap(pattern=[[W, p_hi - p_lo], [1, c_hi - c_lo]],
                                                 offset=(r_base + p_lo) * W + C0 - pc + c_lo))
                    for j in range(kernel_cols):
                        t = i * kernel_cols + j
                        dst = o_tile if t == n_taps - 1 else acc
                        if t == 0:
                            nisa.tensor_scalar(dst=dst, data=win[0:M, j:j + N], op0=nl.multiply,
                                               operand0=w_b[0:M, t:t + 1])
                        else:
                            nisa.scalar_tensor_tensor(dst=dst, data=win[0:M, j:j + N], op0=nl.multiply,
                                                      operand0=w_b[0:M, t:t + 1], op1=nl.add, operand1=acc)
                nisa.dma_copy(dst=out.ap(pattern=[[W, M], [1, N]], offset=R0 * W + C0), src=o_tile)
        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (256, 512, 1024, 2048)]
_kernel = gaussian_blur_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, kernel: torch.Tensor, input_rows: int, input_cols: int,
        kernel_rows: int, kernel_cols: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if kernel_rows < 1 or kernel_cols < 1:
        raise NotImplementedError("gaussian_blur NKI: kernel_rows and kernel_cols must be >= 1")
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
