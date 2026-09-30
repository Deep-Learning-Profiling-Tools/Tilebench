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
except ImportError:
    nki = None

PMAX = 128

FREE_CAP = 8192
FP32_FREE_CAP = 2048
FALLBACK_TILE = 2048


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


def kernel_assert(condition: bool, error_text: str):
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


if nki is not None:
    def _rstd_newton(dst, mean_sq, row_size):
        std = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=std, op=nl.sqrt, data=mean_sq)
        nisa.reciprocal(dst=dst, data=std)

        half_mean_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=half_mean_sq, data=mean_sq, op0=nl.multiply, operand0=0.5)

        correction = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        for _newton_step in range(2):
            nisa.tensor_tensor(dst=correction, data1=dst, data2=dst, op=nl.multiply)
            nisa.tensor_tensor(dst=correction, data1=correction, data2=half_mean_sq,
                               op=nl.multiply)
            nisa.tensor_scalar(dst=correction, data=correction,
                               op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.5)
            nisa.tensor_tensor(dst=dst, data1=dst, data2=correction, op=nl.multiply)

    @nki.jit
    def rmsnorm_kernel(a_input, weight_input, eps, free_cap, block_size):
        kernel_assert(len(a_input.shape) == 2, "input must be 2D [rows, n_cols]")
        n_rows, n_cols = a_input.shape
        kernel_assert(len(weight_input.shape) == 2 and weight_input.shape[0] == 1
                      and weight_input.shape[1] == n_cols,
                      "weight must be [1, n_cols]")
        kernel_assert(min(n_cols, block_size) <= nl.tile_size.sbuf_fmax,
                      "column block exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((n_rows, n_cols), dtype=a_input.dtype, buffer=nl.shared_hbm)
        n_row_tiles = div_ceil(n_rows, PMAX)
        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = div_ceil(n_row_tiles, n_prog)
        t_lo = min(n_row_tiles, pid * per_core)
        t_hi = min(n_row_tiles, t_lo + per_core)


        inv_n_cols = 1.0 / float(n_cols)

        if n_cols <= free_cap:
            w_all = nl.ndarray((PMAX, n_cols), dtype=weight_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_all, src=weight_input.ap(pattern=[[0, PMAX], [1, n_cols]], offset=0))

            for row_tile in range(t_lo, t_hi):
                row_start = row_tile * PMAX
                row_size = min(PMAX, n_rows - row_start)
                row_end = row_start + row_size

                x_tile = nl.ndarray((row_size, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, 0:n_cols])

                sq_tile = nl.ndarray((row_size, n_cols), dtype=nl.float32, buffer=nl.sbuf)
                mean_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq_tile, op=nl.square, data=x_tile,
                                       reduce_op=nl.add, reduce_res=mean_sq)
                nisa.tensor_scalar(dst=mean_sq, data=mean_sq,
                                   op0=nl.multiply, operand0=inv_n_cols,
                                   op1=nl.add, operand1=eps)

                rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                _rstd_newton(rstd, mean_sq, row_size)

                y_tile = nl.ndarray((row_size, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=rstd,
                                          op1=nl.multiply, operand1=w_all[0:row_size, 0:n_cols])

                nisa.dma_copy(dst=out_hbm[row_start:row_end, 0:n_cols], src=y_tile)

            return out_hbm

        n_col_tiles = div_ceil(n_cols, block_size)
        w_all = nl.ndarray((PMAX, n_cols), dtype=weight_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_all, src=weight_input.ap(pattern=[[0, PMAX], [1, n_cols]], offset=0))

        for row_tile in range(t_lo, t_hi):
            row_start = row_tile * PMAX
            row_size = min(PMAX, n_rows - row_start)
            row_end = row_start + row_size

            mean_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=mean_sq, value=0.0)

            for col_tile in range(n_col_tiles):
                col_start = col_tile * block_size
                col_size = min(block_size, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                sq_tile = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
                part_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq_tile, op=nl.square, data=x_tile,
                                       reduce_op=nl.add, reduce_res=part_sq)
                nisa.tensor_tensor(dst=mean_sq, data1=mean_sq, data2=part_sq, op=nl.add)

            nisa.tensor_scalar(dst=mean_sq, data=mean_sq,
                               op0=nl.multiply, operand0=inv_n_cols,
                               op1=nl.add, operand1=eps)

            rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            _rstd_newton(rstd, mean_sq, row_size)

            for col_tile in range(n_col_tiles):
                col_start = col_tile * block_size
                col_size = min(block_size, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                y_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=rstd,
                                          op1=nl.multiply, operand1=w_all[0:row_size, col_start:col_end])

                nisa.dma_copy(dst=out_hbm[row_start:row_end, col_start:col_end], src=y_tile)

        return out_hbm


_DEFAULT_CONFIG = SimpleNamespace(free_cap=0, block_size=1024)
_LNC = _lnc_degree()
_kernel = rmsnorm_kernel[_LNC] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None


def _search_space(n_cols: int, dtype) -> list:
    space = [SimpleNamespace(free_cap=0, block_size=b) for b in (512, 1024, 2048)]
    cap = FP32_FREE_CAP if dtype == torch.float32 else FREE_CAP
    if n_cols <= cap:
        space.append(SimpleNamespace(free_cap=cap, block_size=1024))
    return space
_last_autotune_config: dict = {}


def run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("rmsnorm NKI: int8 not supported")

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    weight_2d = weight.reshape(1, -1)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x_2d.shape), str(x_2d.dtype)),
            search_space=_search_space(x_2d.shape[1], x_2d.dtype),
            args_fn=lambda cfg: (x_2d, weight_2d, float(eps), cfg.free_cap, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    out_2d = _kernel(x_2d, weight_2d, float(eps), cfg.free_cap, cfg.block_size)
    return out_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
