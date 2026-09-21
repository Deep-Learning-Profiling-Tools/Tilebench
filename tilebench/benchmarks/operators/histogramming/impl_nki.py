"""NKI (AWS Trainium) implementation of histogramming: ``hist[b] = #{i : x[i] == b}`` for ``b < num_bins``.

Design (mirrors the Triton / cuTile two-stage structure):

* Triton shards the input across workers, each worker accumulates a private
  partial histogram with ``atomic_add`` (one read per element), and a second
  kernel reduces the partials. NeuronCores have no atomic scatter-add, so the
  per-element "+1 into bin ``v``" is expressed as a one-hot outer product that
  the Tensor engine accumulates:
  - a bin id is split into two digits ``v = hi * n_lo + lo``
    (``n_hi * n_lo = num_bins``, e.g. 32 x 32 for 1024 bins);
  - for a chunk of ``[128, F]`` values, one-hot codes ``OH_hi[128, F, HP]`` and
    ``OH_lo[128, F, n_lo]`` (bf16, exact 0/1) are built by one broadcast compare
    each on the Vector engine (``HP = max(32, n_hi)`` columns per value, the
    unused ones zero, so that the ``G = 128 / HP`` results below land on
    32-aligned partitions);
  - ``G`` consecutive value columns are contracted by a single ``nc_matmul``:
    ``stationary = OH_hi[:, f:f+G, :]`` (``[128, 128]``), ``moving = OH_lo[:, f:f+G, :]``
    (``[128, G * n_lo]``), whose ``[128, G * n_lo]`` PSUM result holds the wanted
    ``[n_hi, n_lo]`` count blocks on its block diagonal (the off-diagonal blocks
    pair values of different columns and are ignored). One instruction thus
    consumes ``128 * G`` values; the fp32 PSUM is flushed into an exact int32
    SBUF accumulator every ``2^20`` values, so counts stay exact for any ``N``.
  - each program instance handles its own contiguous share of the input and
    writes its partial ``[num_bins]`` histogram; ``run`` sums the per-program
    partials (the same "reduce the partials" stage as Triton's second kernel).
  Every element is read exactly once (``O(N)`` DMA); compute is
  ``O(N * (n_hi + n_lo))`` Vector-engine work + ``N / (128 * G)`` matmuls.
  ``BLOCK`` (values per partition per chunk) defaults to Triton's
  ``partial_BLOCK_SIZE`` (1024); search 1024/2048. It is capped so that the
  one-hot tiles of a chunk fit in SBUF.
* Requires ``num_bins`` to factor as ``n_hi * n_lo`` with ``n_lo`` a power of two
  and ``n_hi, n_lo <= 128`` (all benchmark cases are powers of two); values are
  assumed to lie in ``[0, num_bins)`` as in the Triton kernel's valid range.
"""
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

FLUSH_EVERY = 1 << 20   # values accumulated in fp32 PSUM before flushing to the int32 accumulator
SBUF_ONEHOT_BYTES = 32 * 1024   # per-partition budget for one chunk's pair of one-hot tiles


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


def _digits(num_bins: int):
    """Split ``num_bins`` into ``(n_hi, n_lo, lo_bits)`` with ``n_lo = 2**lo_bits`` closest to ``sqrt(num_bins)``."""
    lo_bits = 0
    while (1 << (lo_bits + 1)) <= int(math.isqrt(num_bins)) and num_bins % (1 << (lo_bits + 1)) == 0:
        lo_bits += 1
    n_lo = 1 << lo_bits
    n_hi = num_bins // n_lo
    if n_hi * n_lo != num_bins or n_hi > PMAX or n_lo > PMAX:
        raise NotImplementedError(f"histogramming NKI: unsupported num_bins={num_bins}")
    return n_hi, n_lo, lo_bits


def _chunk_cols(block_size: int, n_hi: int, n_lo: int) -> int:
    """Values per partition per chunk: ``block_size`` capped by the one-hot SBUF budget."""
    hp = max(32, n_hi)
    f = min(block_size, SBUF_ONEHOT_BYTES // (2 * (hp + n_lo)))
    return max(PMAX // hp, f - f % (PMAX // hp))


if nki is not None:
    @nki.jit
    def histogram_kernel(values, num_bins, n_hi, n_lo, lo_bits, chunk_cols):
        """Per-program partial histograms of the flat ``(N,)`` int32 ``values`` -> ``[lnc, num_bins]`` int32."""
        N = values.shape[0]
        num_programs = nl.num_programs()
        pid = nl.program_id(0)
        out = nl.ndarray((num_programs, num_bins), dtype=nl.int32, buffer=nl.shared_hbm)
        HP = max(32, n_hi)              # one-hot width of the hi digit (32-aligned PSUM blocks)
        G = PMAX // HP                  # value columns contracted per matmul

        # digit iotas [128, n_d] (bf16: exact small integers, enables the 2x Vector-engine mode)
        iota_hi = nl.ndarray((PMAX, n_hi), dtype=nl.bfloat16, buffer=nl.sbuf)
        _iota_bf16(iota_hi, n_hi)
        iota_lo = nl.ndarray((PMAX, n_lo), dtype=nl.bfloat16, buffer=nl.sbuf)
        _iota_bf16(iota_lo, n_lo)
        # ping-pong one-hot buffers, zeroed once (the compare only writes the first n_hi columns)
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

        # This program's contiguous share of the values.
        per_core = (N + num_programs - 1) // num_programs
        lo = pid * per_core
        hi = min(N, lo + per_core)
        chunk = PMAX * chunk_cols
        since_flush = 0
        group_start = True                  # next matmul overwrites (rather than accumulates into) PSUM
        for ci in range(max(0, (hi - lo + chunk - 1) // chunk)):
            start = lo + ci * chunk
            count = min(chunk, hi - start)
            q = count // PMAX               # [128, q] values
            r = count - q * PMAX            # tail values (< 128): processed as a [1, r] chunk
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
                # fp32 PSUM stays exact below 2^24: flush into the int32 accumulator.
                _flush(psum, acc, n_hi, n_lo, HP, G)
                since_flush = 0
                group_start = True
        if not group_start:
            _flush(psum, acc, n_hi, n_lo, HP, G)
        # counts[hi * n_lo + lo] <- acc[hi, lo], written to this program's row of ``out``.
        nisa.dma_copy(dst=out.ap(pattern=[[n_lo, n_hi], [1, n_lo]], offset=pid * num_bins), src=acc)
        return out

    def _iota_bf16(dst, n):
        """``dst[p, d] = d`` (bf16) on every partition."""
        tmp = nl.ndarray((PMAX, n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=tmp, pattern=[[1, n]], offset=0, channel_multiplier=0)
        nisa.tensor_copy(dst=dst, src=tmp)

    def _chunk(values, p, f, start, n_hi, n_lo, lo_bits, HP, G, iota_hi, iota_lo, oh_hi, oh_lo, psum, group_start):
        """Accumulate the ``[p, f]`` chunk of values at flat offset ``start`` into ``psum``."""
        v = nl.ndarray((p, f), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v, src=values.ap(pattern=[[f, p], [1, f]], offset=start))
        # digits hi = v >> lo_bits, lo = v & (n_lo - 1), converted to bf16 (exact: < 128)
        hi_i = nl.ndarray((p, f), dtype=nl.int32, buffer=nl.sbuf)
        lo_i = nl.ndarray((p, f), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hi_i, data=v, op0=nl.right_shift, operand0=lo_bits)
        nisa.tensor_scalar(dst=lo_i, data=v, op0=nl.bitwise_and, operand0=n_lo - 1)
        hi_d = nl.ndarray((p, f), dtype=nl.bfloat16, buffer=nl.sbuf)
        lo_d = nl.ndarray((p, f), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hi_d, src=hi_i)
        nisa.tensor_copy(dst=lo_d, src=lo_i)
        # one-hots oh[p, f, d] = (digit[p, f] == d): one broadcast compare per digit
        nisa.tensor_tensor(dst=oh_hi[0:p, 0:f, 0:n_hi],
                           data1=hi_d.ap(pattern=[[f, p], [1, f], [0, n_hi]]),
                           data2=iota_hi.ap(pattern=[[n_hi, p], [0, f], [1, n_hi]]),
                           op=nl.equal)
        nisa.tensor_tensor(dst=oh_lo[0:p, 0:f, 0:n_lo],
                           data1=lo_d.ap(pattern=[[f, p], [1, f], [0, n_lo]]),
                           data2=iota_lo.ap(pattern=[[n_lo, p], [0, f], [1, n_lo]]),
                           op=nl.equal)
        # G value columns per matmul: psum[g*HP + hi, g*n_lo + lo] += #{values in column f+g with digits (hi, lo)}
        for f0 in range(0, f, G):
            g = min(G, f - f0)
            nisa.nc_matmul(dst=psum[0:g * HP, 0:g * n_lo],
                           stationary=oh_hi.ap(pattern=[[chunk_stride(oh_hi), p], [1, g * HP]], offset=f0 * HP),
                           moving=oh_lo.ap(pattern=[[chunk_stride(oh_lo), p], [1, g * n_lo]], offset=f0 * n_lo),
                           accumulate=(f0 > 0 or not group_start))

    def chunk_stride(t):
        """Elements per partition of a ``[128, F, D]`` SBUF tile (partition stride of its flat view)."""
        return t.shape[1] * t.shape[2]

    def _flush(psum, acc, n_hi, n_lo, HP, G):
        """``acc[hi, lo] += sum_g psum[g*HP + hi, g*n_lo + lo]`` (diagonal blocks), then PSUM is reused."""
        for g in range(G):
            tmp = nl.ndarray((n_hi, n_lo), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp, src=psum[g * HP:g * HP + n_hi, g * n_lo:(g + 1) * n_lo])
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)


# Values per partition per chunk: Triton's partial_BLOCK_SIZE (default 1024; search 1024/2048).
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
        # The SBUF cap can map several block sizes onto the same chunk width: keep one config each.
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
    # Stage 2 (Triton's reduce kernel): sum the per-program partial histograms.
    return partials.sum(dim=0, dtype=torch.int32)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
