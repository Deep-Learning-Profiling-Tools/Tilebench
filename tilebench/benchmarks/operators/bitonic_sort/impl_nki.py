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

SPAN_CAP = 1024


if nki is not None:
    @nki.jit
    def bitonic_local_kernel(src, dst, k, log2k, j, span, P, n_blocks):
        two_j = 2 * j
        C = span // two_j
        pat = [[span, P], [1, span]]

        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (n_blocks + n_prog - 1) // n_prog
        b_lo = min(n_blocks, pid * per_core)
        b_hi = min(n_blocks, b_lo + per_core)

        for b in range(b_lo, b_hi):
            base = b * P * span

            tile = nl.ndarray((P, C, two_j), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile.ap(pattern=pat), src=src.ap(pattern=pat, offset=base))
            lower = tile[:, :, 0:j]
            upper = tile[:, :, j:two_j]

            idx = nl.ndarray((P, C, j), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=idx, pattern=[[two_j, C], [1, j]], offset=base,
                      channel_multiplier=span)
            kbit = nl.add(nl.right_shift(nl.bitwise_and(idx, k), log2k), 0.0,
                          dtype=nl.float32)
            descending = nl.greater(kbit, 0.5)

            keep_lower = nl.logical_xor(nl.less_equal(lower, upper), descending)

            out = nl.ndarray((P, C, two_j), dtype=src.dtype, buffer=nl.sbuf)
            out[:, :, 0:j] = nl.where(keep_lower, lower, upper)
            out[:, :, j:two_j] = nl.where(keep_lower, upper, lower)

            nisa.dma_copy(dst=dst.ap(pattern=pat, offset=base), src=out.ap(pattern=pat))

        return dst

    @nki.jit
    def bitonic_stride_kernel(src, dst, k, log2k, j, W, P, part_stride,
                              n_outer, outer_stride, n_inner, inner_stride):
        pat = [[part_stride, P], [1, W]]

        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (n_outer + n_prog - 1) // n_prog
        a_lo = min(n_outer, pid * per_core)
        a_hi = min(n_outer, a_lo + per_core)

        for a in range(a_lo, a_hi):
            for c in range(n_inner):
                base = a * outer_stride + c * inner_stride

                lower = nl.ndarray((P, W), dtype=src.dtype, buffer=nl.sbuf)
                upper = nl.ndarray((P, W), dtype=src.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=lower, src=src.ap(pattern=pat, offset=base))
                nisa.dma_copy(dst=upper, src=src.ap(pattern=pat, offset=base + j))

                idx = nl.ndarray((P, W), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(dst=idx, pattern=[[1, W]], offset=base,
                          channel_multiplier=part_stride)
                kbit = nl.add(nl.right_shift(nl.bitwise_and(idx, k), log2k), 0.0,
                              dtype=nl.float32)
                descending = nl.greater(kbit, 0.5)

                keep_lower = nl.logical_xor(nl.less_equal(lower, upper), descending)
                new_lower = nl.where(keep_lower, lower, upper)
                new_upper = nl.where(keep_lower, upper, lower)

                nisa.dma_copy(dst=dst.ap(pattern=pat, offset=base), src=new_lower)
                nisa.dma_copy(dst=dst.ap(pattern=pat, offset=base + j), src=new_upper)

        return dst


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


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _plan(M: int, j: int, span_cap: int = SPAN_CAP):
    if 2 * j <= span_cap:
        span = min(span_cap, max(2 * j, M // PMAX))
        P = min(PMAX, M // span)
        return ("local", span, P, M // (P * span))

    W = span_cap
    p_across_runs = min(PMAX, M // (2 * j))
    p_within_run = min(PMAX, j // W)
    if p_across_runs >= p_within_run:
        P = p_across_runs
        return ("stride", W, P, 2 * j, M // (2 * j * P), P * 2 * j, j // W, W)
    P = p_within_run
    return ("stride", W, P, W, M // (2 * j), 2 * j, j // (W * P), P * W)


def _bitonic_sort_1d(data: torch.Tensor, span_cap: int = SPAN_CAP) -> torch.Tensor:
    N = data.numel()
    if N <= 1:
        return data.clone()

    M = max(_next_pow2(N), PMAX)
    PAD_VALUE = 65504.0 if data.dtype == torch.float16 else 1e30
    work_a = torch.full((M, 1), PAD_VALUE, dtype=data.dtype, device=data.device)
    work_a[:N, 0] = data
    work_b = torch.empty_like(work_a)

    src, dst = work_a, work_b
    k = 2
    while k <= M:
        j = k // 2
        log2k = k.bit_length() - 1
        while j > 0:
            plan = _plan(M, j, span_cap)
            if plan[0] == "local":
                _, span, P, n_blocks = plan
                dst = bitonic_local_kernel[_LNC](src, dst, k, log2k, j, span, P, n_blocks)
            else:
                _, W, P, part_stride, n_outer, outer_stride, n_inner, inner_stride = plan
                dst = bitonic_stride_kernel[_LNC](src, dst, k, log2k, j, W, P, part_stride,
                                            n_outer, outer_stride, n_inner, inner_stride)
            src, dst = dst, src
            j //= 2
        k *= 2

    return src[:N, 0]


def _first_stage_args(data: torch.Tensor, span_cap: int) -> tuple:
    N = data.numel()
    M = max(_next_pow2(N), PMAX)
    pad = 65504.0 if data.dtype == torch.float16 else 1e30
    work_a = torch.full((M, 1), pad, dtype=data.dtype, device=data.device)
    work_a[:N, 0] = data
    work_b = torch.empty_like(work_a)
    _, span, P, n_blocks = _plan(M, 1, span_cap)
    return (work_a, work_b, 2, 1, 1, span, P, n_blocks)


_DEFAULT_CONFIG = SimpleNamespace(block_size=SPAN_CAP)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (512, 1024, 2048, 4096)]
_LNC = _lnc_degree()
_tuner = NkiAutotuner(bitonic_local_kernel[_LNC]) if nki is not None else None
_last_autotune_config: dict = {}


def run(data: torch.Tensor, N: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=((N,), str(data.dtype)), search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: _first_stage_args(data, cfg.block_size))
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _bitonic_sort_1d(data, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
