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


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


if nki is not None:
    def _core_range(n: int):
        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per = _div_ceil(n, n_prog)
        lo = min(n, pid * per)
        return lo, min(n, lo + per)

    @nki.jit
    def phi_kernel(x):
        rows, cols = x.shape
        result = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.shared_hbm)
        t_lo, t_hi = _core_range(_div_ceil(rows, PMAX))
        for t in range(t_lo, t_hi):
            r0 = t * PMAX
            rs = min(PMAX, rows - r0)
            x_tile = nl.ndarray((rs, cols), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[r0:r0 + rs, 0:cols])
            pos_val = nl.ndarray((rs, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos_val, data=x_tile, op0=nl.add, operand0=1.0)
            exp_val = nl.ndarray((rs, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_val, op=nl.exp, data=x_tile)
            is_pos = nl.ndarray((rs, cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=is_pos, data=x_tile, op0=nl.greater, operand0=0.0)
            pos_part = nl.ndarray((rs, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=pos_part, predicate=is_pos, on_true=pos_val, on_false=0.0)
            exp_part = nl.ndarray((rs, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=exp_part, predicate=is_pos, on_true=exp_val, on_false=0.0,
                               reverse_pred=True)
            res = nl.ndarray((rs, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=res, data1=pos_part, data2=exp_part, op=nl.add)
            nisa.dma_copy(dst=result[r0:r0 + rs, 0:cols], src=res)
        return result

    @nki.jit
    def stageA_kernel(phi_K, V, tile_d):
        M, D = phi_K.shape
        Dp1 = D + 1
        S_aug = nl.ndarray((D, Dp1), dtype=nl.float32, buffer=nl.shared_hbm)
        j0, j1 = _core_range(Dp1)
        w = j1 - j0
        vw = max(0, min(j1, D) - j0)
        n_m = _div_ceil(M, PMAX)
        if w > 0:
            for d0 in range(0, D, tile_d):
                ds = min(tile_d, D - d0)
                psum = nl.ndarray((ds, w), dtype=nl.float32, buffer=nl.psum)
                for mt in range(n_m):
                    m0 = mt * PMAX
                    ms = min(PMAX, M - m0)
                    stat = nl.ndarray((ms, ds), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=stat, src=phi_K[m0:m0 + ms, d0:d0 + ds])
                    mov = nl.ndarray((ms, w), dtype=nl.float32, buffer=nl.sbuf)
                    if vw > 0:
                        nisa.dma_copy(dst=mov[0:ms, 0:vw], src=V[m0:m0 + ms, j0:j0 + vw])
                    if j1 > D:
                        nisa.memset(dst=mov[0:ms, w - 1:w], value=1.0)
                    nisa.nc_matmul(dst=psum, stationary=stat, moving=mov, accumulate=(mt > 0))
                res = nl.ndarray((ds, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=res, src=psum)
                nisa.dma_copy(dst=S_aug[d0:d0 + ds, j0:j1], src=res)
        return S_aug

    @nki.jit
    def stageB_kernel(phi_Q, S_aug, eps, tile_m):
        M, D = phi_Q.shape
        Dp1 = D + 1
        out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.shared_hbm)
        t_lo, t_hi = _core_range(_div_ceil(M, tile_m))
        n_d = _div_ceil(D, PMAX)
        for t in range(t_lo, t_hi):
            m0 = t * tile_m
            ms = min(tile_m, M - m0)
            psum = nl.ndarray((ms, Dp1), dtype=nl.float32, buffer=nl.psum)
            for dt in range(n_d):
                d0 = dt * PMAX
                dc = min(PMAX, D - d0)
                q_tile = nl.ndarray((ms, dc), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_tile, src=phi_Q[m0:m0 + ms, d0:d0 + dc])
                qt_ps = nl.ndarray((dc, ms), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=qt_ps, data=q_tile)
                qt = nl.ndarray((dc, ms), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qt, src=qt_ps)
                s_tile = nl.ndarray((dc, Dp1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=s_tile, src=S_aug[d0:d0 + dc, 0:Dp1])
                nisa.nc_matmul(dst=psum, stationary=qt, moving=s_tile, accumulate=(dt > 0))
            o_aug = nl.ndarray((ms, Dp1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_aug, src=psum)
            denom = nl.ndarray((ms, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=denom, data=o_aug[0:ms, D:Dp1], op0=nl.add, operand0=eps)
            inv = nl.ndarray((ms, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv, data=denom)
            res = nl.ndarray((ms, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=res, data=o_aug[0:ms, 0:D], op0=nl.multiply, operand0=inv)
            nisa.dma_copy(dst=out[m0:m0 + ms, 0:D], src=res)
        return out


_LNC = _lnc_degree()
_phi = phi_kernel[_LNC] if nki is not None else None
_stage_a = stageA_kernel[_LNC] if nki is not None else None
_stage_b = stageB_kernel[_LNC] if nki is not None else None
_DEFAULT_CONFIG = SimpleNamespace(tile_d=64, tile_m=64)
_SEARCH_A = [SimpleNamespace(tile_d=d) for d in (32, 64, 128)]
_SEARCH_B = [SimpleNamespace(tile_m=m) for m in (32, 64, 128)]
_tuner_a = NkiAutotuner(_stage_a) if nki is not None else None
_tuner_b = NkiAutotuner(_stage_b) if nki is not None else None
_last_autotune_config: dict = {}


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    M, D = Q.shape
    phi_Q = _phi(Q)
    phi_K = _phi(K)
    if autotune:
        cfg_a = _tuner_a.tune_or_cached(
            shape_key=((M, D), str(Q.dtype), "A"),
            search_space=[c for c in _SEARCH_A if c.tile_d <= max(D, 32)],
            args_fn=lambda c: (phi_K, V, c.tile_d))
    else:
        cfg_a = _DEFAULT_CONFIG
    S_aug = _stage_a(phi_K, V, cfg_a.tile_d)
    if autotune:
        cfg_b = _tuner_b.tune_or_cached(
            shape_key=((M, D), str(Q.dtype), "B"),
            search_space=_SEARCH_B,
            args_fn=lambda c: (phi_Q, S_aug, float(eps), c.tile_m))
        _last_autotune_config.clear()
        _last_autotune_config.update({"tile_d": cfg_a.tile_d, "tile_m": cfg_b.tile_m})
    else:
        cfg_b = _DEFAULT_CONFIG
    return _stage_b(phi_Q, S_aug, float(eps), cfg_b.tile_m)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
