import functools
import math
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

FLUSH_EVERY = 1 << 20
SBUF_ONEHOT_BYTES = 32 * 1024


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


def _digits(num_bins: int):
    lo_bits = 0
    while (1 << (lo_bits + 1)) <= int(math.isqrt(num_bins)) and num_bins % (1 << (lo_bits + 1)) == 0:
        lo_bits += 1
    n_lo = 1 << lo_bits
    n_hi = num_bins // n_lo
    if n_hi * n_lo != num_bins or n_hi > PMAX or n_lo > PMAX:
        raise NotImplementedError(f"histogramming NKI: unsupported num_bins={num_bins}")
    return n_hi, n_lo, lo_bits


def _chunk_cols(block_size: int, n_hi: int, n_lo: int) -> int:
    hp = max(32, n_hi)
    f = min(block_size, SBUF_ONEHOT_BYTES // (2 * (hp + n_lo)))
    return max(PMAX // hp, f - f % (PMAX // hp))


if nki is not None:
    @nki.jit
    def histogram_kernel(values, num_bins, n_hi, n_lo, lo_bits, chunk_cols):
        N = values.shape[0]
        num_programs = nl.num_programs()
        pid = nl.program_id(0)
        out = nl.ndarray((num_programs, num_bins), dtype=nl.int32, buffer=nl.shared_hbm)
        HP = max(32, n_hi)
        G = PMAX // HP

        iota_hi = nl.ndarray((PMAX, n_hi), dtype=nl.bfloat16, buffer=nl.sbuf)
        _iota_bf16(iota_hi, n_hi)
        iota_lo = nl.ndarray((PMAX, n_lo), dtype=nl.bfloat16, buffer=nl.sbuf)
        _iota_bf16(iota_lo, n_lo)
        oh_hi = []
        oh_lo = []
        for _ in range(2):
            t = nl.ndarray((PMAX, chunk_cols, HP), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=t, value=0.0)
            oh_hi.append(t)
            oh_lo.append(nl.ndarray((PMAX, chunk_cols, n_lo), dtype=nl.bfloat16, buffer=nl.sbuf))
        acc = nl.ndarray((n_hi, n_lo), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0)
        psum = nl.ndarray((PMAX, G * n_lo), dtype=nl.float32, buffer=nl.psum)

        per_core = (N + num_programs - 1) // num_programs
        lo = pid * per_core
        hi = min(N, lo + per_core)
        chunk = PMAX * chunk_cols
        since_flush = 0
        group_start = True
        for ci in range(max(0, (hi - lo + chunk - 1) // chunk)):
            start = lo + ci * chunk
            count = min(chunk, hi - start)
            q = count // PMAX
            r = count - q * PMAX
            buf = ci % 2
            if q > 0:
                _chunk(values, PMAX, q, start, n_hi, n_lo, lo_bits, HP, G,
                       iota_hi, iota_lo, oh_hi[buf], oh_lo[buf], psum, group_start)
                group_start = False
            if r > 0:
                _chunk(values, 1, r, start + q * PMAX, n_hi, n_lo, lo_bits, HP, G,
                       iota_hi, iota_lo, oh_hi[buf], oh_lo[buf], psum, group_start)
                group_start = False
            since_flush += count
            if since_flush >= FLUSH_EVERY:
                _flush(psum, acc, n_hi, n_lo, HP, G)
                since_flush = 0
                group_start = True
        if not group_start:
            _flush(psum, acc, n_hi, n_lo, HP, G)
        nisa.dma_copy(dst=out.ap(pattern=[[n_lo, n_hi], [1, n_lo]], offset=pid * num_bins), src=acc)
        return out

    def _iota_bf16(dst, n):
        tmp = nl.ndarray((PMAX, n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=tmp, pattern=[[1, n]], offset=0, channel_multiplier=0)
        nisa.tensor_copy(dst=dst, src=tmp)

    def _chunk(values, p, f, start, n_hi, n_lo, lo_bits, HP, G, iota_hi, iota_lo, oh_hi, oh_lo, psum, group_start):
        v = nl.ndarray((p, f), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v, src=values.ap(pattern=[[f, p], [1, f]], offset=start))
        hi_i = nl.ndarray((p, f), dtype=nl.int32, buffer=nl.sbuf)
        lo_i = nl.ndarray((p, f), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hi_i, data=v, op0=nl.right_shift, operand0=lo_bits)
        nisa.tensor_scalar(dst=lo_i, data=v, op0=nl.bitwise_and, operand0=n_lo - 1)
        hi_d = nl.ndarray((p, f), dtype=nl.bfloat16, buffer=nl.sbuf)
        lo_d = nl.ndarray((p, f), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hi_d, src=hi_i)
        nisa.tensor_copy(dst=lo_d, src=lo_i)
        nisa.tensor_tensor(dst=oh_hi[0:p, 0:f, 0:n_hi],
                           data1=hi_d.ap(pattern=[[f, p], [1, f], [0, n_hi]]),
                           data2=iota_hi.ap(pattern=[[n_hi, p], [0, f], [1, n_hi]]),
                           op=nl.equal)
        nisa.tensor_tensor(dst=oh_lo[0:p, 0:f, 0:n_lo],
                           data1=lo_d.ap(pattern=[[f, p], [1, f], [0, n_lo]]),
                           data2=iota_lo.ap(pattern=[[n_lo, p], [0, f], [1, n_lo]]),
                           op=nl.equal)
        for f0 in range(0, f, G):
            g = min(G, f - f0)
            nisa.nc_matmul(dst=psum[0:g * HP, 0:g * n_lo],
                           stationary=oh_hi.ap(pattern=[[chunk_stride(oh_hi), p], [1, g * HP]], offset=f0 * HP),
                           moving=oh_lo.ap(pattern=[[chunk_stride(oh_lo), p], [1, g * n_lo]], offset=f0 * n_lo),
                           accumulate=(f0 > 0 or not group_start))

    def chunk_stride(t):
        return t.shape[1] * t.shape[2]

    def _flush(psum, acc, n_hi, n_lo, HP, G):
        for g in range(G):
            tmp = nl.ndarray((n_hi, n_lo), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp, src=psum[g * HP:g * HP + n_hi, g * n_lo:(g + 1) * n_lo])
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)


_DEFAULT_CONFIG = SimpleNamespace(block_size=1024)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048)]
_kernel = histogram_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, N: int, num_bins: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    assert input.ndim == 1 and input.shape[0] == N and input.dtype == torch.int32
    n_hi, n_lo, lo_bits = _digits(num_bins)
    if autotune:
        seen = set()
        space = [c for c in _SEARCH_SPACE
                 if not (_chunk_cols(c.block_size, n_hi, n_lo) in seen or seen.add(_chunk_cols(c.block_size, n_hi, n_lo)))]
        cfg = _tuner.tune_or_cached(
            shape_key=(N, num_bins),
            search_space=space,
            args_fn=lambda cfg: (input, num_bins, n_hi, n_lo, lo_bits, _chunk_cols(cfg.block_size, n_hi, n_lo)),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    partials = _kernel(input, num_bins, n_hi, n_lo, lo_bits, _chunk_cols(cfg.block_size, n_hi, n_lo))
    return partials.sum(dim=0, dtype=torch.int32)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
