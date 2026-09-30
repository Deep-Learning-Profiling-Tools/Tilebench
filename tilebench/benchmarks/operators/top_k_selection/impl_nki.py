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

PAD_VALUE = -1e30


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


if nki is not None:

    def _stage_desc(cur, nxt, P, L, k, j):
        G = k // (2 * j)
        R = L // (2 * k)
        if R == 0:
            src = cur.reshape((P, G, 2, j))
            dst = nxt.reshape((P, G, 2, j))
            lo = src.select(2, 0)
            hi = src.select(2, 1)
            nisa.tensor_tensor(dst=dst.select(2, 0), data1=lo, data2=hi, op=nl.maximum)
            nisa.tensor_tensor(dst=dst.select(2, 1), data1=lo, data2=hi, op=nl.minimum)
        else:
            src = cur.reshape((P, R, 2, G, 2, j))
            dst = nxt.reshape((P, R, 2, G, 2, j))
            d0_lo = src.select(4, 0).select(2, 0)
            d0_hi = src.select(4, 1).select(2, 0)
            nisa.tensor_tensor(dst=dst.select(4, 0).select(2, 0), data1=d0_lo, data2=d0_hi, op=nl.maximum)
            nisa.tensor_tensor(dst=dst.select(4, 1).select(2, 0), data1=d0_lo, data2=d0_hi, op=nl.minimum)
            d1_lo = src.select(4, 0).select(2, 1)
            d1_hi = src.select(4, 1).select(2, 1)
            nisa.tensor_tensor(dst=dst.select(4, 0).select(2, 1), data1=d1_lo, data2=d1_hi, op=nl.minimum)
            nisa.tensor_tensor(dst=dst.select(4, 1).select(2, 1), data1=d1_lo, data2=d1_hi, op=nl.maximum)

    def _merge_chunks(cur, P, L, K2, log2K2):
        for jb in range(log2K2 - 1, -1, -1):
            nxt = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
            _stage_desc(cur, nxt, P, L, K2, 1 << jb)
            cur = nxt
        return cur

    def _block_topk(cur, P, B, K2, log2K2):
        for kb in range(1, log2K2 + 1):
            for jb in range(kb - 1, -1, -1):
                nxt = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
                _stage_desc(cur, nxt, P, B, 1 << kb, 1 << jb)
                cur = nxt
        L = B
        while L > K2:
            half = L // 2
            pairs = cur.reshape((P, half // K2, 2, K2))
            nxt = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=nxt.reshape((P, half // K2, K2)),
                               data1=pairs.select(2, 0), data2=pairs.select(2, 1), op=nl.maximum)
            cur = _merge_chunks(nxt, P, half, K2, log2K2)
            L = half
        return cur

    def _level(src, src_off, n, dst, dst_off, B, K2, log2K2):
        nb = (n + B - 1) // B
        last = None
        for g in range((nb + PMAX - 1) // PMAX):
            b0 = g * PMAX
            p = min(PMAX, nb - b0)
            full = min(p, (n - b0 * B) // B)
            tile = nl.ndarray((p, B), dtype=nl.float32, buffer=nl.sbuf)
            if full < p:
                nisa.memset(dst=tile, value=PAD_VALUE)
            if full > 0:
                nisa.dma_copy(dst=tile[0:full, 0:B],
                              src=src.ap(pattern=[[B, full], [1, B]], offset=src_off + b0 * B))
            if full < p:
                r = n - (b0 + full) * B
                if r > 0:
                    nisa.dma_copy(dst=tile[full:full + 1, 0:r],
                                  src=src.ap(pattern=[[r, 1], [1, r]], offset=src_off + (b0 + full) * B))
            last = _block_topk(tile, p, B, K2, log2K2)
            if nb > 1:
                nisa.dma_copy(dst=dst.ap(pattern=[[K2, p], [1, K2]], offset=dst_off + b0 * K2), src=last)
        return last

    @nki.jit
    def topk_kernel(x, N, k, K2, log2K2, B):
        out = nl.ndarray((k,), dtype=nl.float32, buffer=nl.shared_hbm)
        num_programs = nl.num_programs()
        pid = nl.program_id(0)

        nb0 = (N + B - 1) // B
        per_core = (nb0 + num_programs - 1) // num_programs
        b_lo = min(nb0, pid * per_core)
        b_hi = min(nb0, b_lo + per_core)
        n = max(0, min(N, b_hi * B) - b_lo * B)

        if n == 0:
            mine = nl.ndarray((1, K2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=mine, value=PAD_VALUE)
        else:
            cap = per_core * K2
            scratch = [nl.ndarray((num_programs * cap,), dtype=nl.float32, buffer=nl.shared_hbm),
                       nl.ndarray((num_programs * cap,), dtype=nl.float32, buffer=nl.shared_hbm)]
            mine = _level(x, b_lo * B, n, scratch[0], pid * cap, B, K2, log2K2)
            n = ((n + B - 1) // B) * K2
            lvl = 0
            while n > K2:
                mine = _level(scratch[lvl % 2], pid * cap, n, scratch[(lvl + 1) % 2], pid * cap,
                              B, K2, log2K2)
                n = ((n + B - 1) // B) * K2
                lvl += 1

        if num_programs > 1:
            other = nl.ndarray((1, K2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=mine, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
            other_rev = nl.ndarray((1, K2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=other_rev, src=other.ap(pattern=[[K2, 1], [-1, K2]], offset=K2 - 1))
            both = nl.ndarray((1, K2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=both, data1=mine, data2=other_rev, op=nl.maximum)
            mine = _merge_chunks(both, 1, K2, K2, log2K2)

        if pid == 0:
            nisa.dma_copy(dst=out, src=mine[0:1, 0:k])
        return out


_DEFAULT_CONFIG = SimpleNamespace(block_size=2048)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048, 4096)]
_kernel = topk_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def _block(block_size: int, K2: int) -> int:
    return max(int(block_size), 2 * K2)


def run(input: torch.Tensor, N: int, k: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if input.dtype != torch.float32 or input.dim() != 1 or input.shape[0] != N:
        raise NotImplementedError("top_k_selection NKI: expects a 1-D fp32 tensor of length N")
    if k < 1 or k > N:
        raise ValueError(f"k={k} out of range for N={N}")
    k = int(k)
    K2 = _next_pow2(k)
    log2K2 = K2.bit_length() - 1
    if autotune:
        seen = set()
        space = [c for c in _SEARCH_SPACE
                 if not (_block(c.block_size, K2) in seen or seen.add(_block(c.block_size, K2)))]
        cfg = _tuner.tune_or_cached(
            shape_key=(int(N), k),
            search_space=space,
            args_fn=lambda cfg: (input, int(N), k, K2, log2K2, _block(cfg.block_size, K2)),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(input, int(N), k, K2, log2K2, _block(cfg.block_size, K2))


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
