"""NKI (AWS Trainium) implementation of top_k_selection: the k largest values of a 1-D fp32
tensor, descending (``torch.topk(x, k).values``).

Design (mirrors the Triton / cuTile kernels):

* Triton runs a hierarchy of block-local top-k passes: every pass splits its input into
  ``BLOCK_SIZE``-element blocks, each program keeps the top ``K2 = next_pow2(k)`` of its block
  (``tl.topk``, a partial bitonic network), and the ``nb * K2`` survivors feed the next pass until
  one block is left. NKI does the same hierarchy with the same ``BLOCK_SIZE``
  (``B = max(BLOCK_SIZE, 2 * K2)``; default 2048, search 1024/2048/4096), with one block per
  SBUF partition so that 128 blocks run through the network in lock-step on the Vector engine:
  - the first ``log2(K2)`` phases of a bitonic sort leave every ``K2``-chunk of a block sorted,
    directions alternating (descending first);
  - each reduction round takes the element-wise ``max`` of neighbouring chunks (a descending and
    an ascending run -> the top ``K2`` of the pair, as a bitonic run), halving the block, and
    re-sorts the new chunks with a ``log2(K2)``-stage bitonic merge (directions alternating
    again), until one descending chunk -- the block's top ``K2`` -- is left.
  This is the partial network of ``tl.topk``: ``O(B log^2 K2)`` compare-exchanges per block
  instead of the ``O(B log^2 B)`` of a full sort.
* The whole hierarchy runs inside ONE kernel launch: survivors of a level are stored to an HBM
  scratch range and reloaded as the next level's blocks. Partial blocks are padded inside the
  kernel (the SBUF tile is pre-filled with a large negative sentinel) -- no host-side padding,
  slicing or per-level relaunch.
* LNC2: the level-0 blocks are split between the two program instances and each core reduces
  its own share down to one descending ``K2`` run; the two runs are swapped with ``sendrecv``
  and merged (``max`` against the reversed partner + one bitonic merge), and program 0 writes
  the ``k`` results.
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

# Sentinel for padded block tails. A large *finite* negative value, never -inf: literal
# infinities corrupt the compare-exchange network on this hardware/SDK. The benchmark draws
# standard-normal values, so -1e30 is safely below anything real.
PAD_VALUE = -1e30


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


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


if nki is not None:

    def _stage_desc(cur, nxt, P, L, k, j):
        """One bitonic compare-exchange stage over P rows of length L (descending-first).

        Row-local element ``i`` pairs with ``i + j``; the direction bit is bit ``log2(k)`` of
        ``i``, so directions come in runs of ``k`` elements that alternate (run 0 descending).
        Viewing a row as ``(R, 2, G, 2, j)`` puts that bit on its own axis, so each direction is
        a plain min/max pair over a strided view -- no index tile, no predicated select.
        """
        G = k // (2 * j)   # pair-groups sharing one direction
        R = L // (2 * k)   # direction-run pairs; 0 when the row is a single k-run
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
            d0_lo = src.select(4, 0).select(2, 0)      # direction bit 0: descending
            d0_hi = src.select(4, 1).select(2, 0)
            nisa.tensor_tensor(dst=dst.select(4, 0).select(2, 0), data1=d0_lo, data2=d0_hi, op=nl.maximum)
            nisa.tensor_tensor(dst=dst.select(4, 1).select(2, 0), data1=d0_lo, data2=d0_hi, op=nl.minimum)
            d1_lo = src.select(4, 0).select(2, 1)      # direction bit 1: ascending
            d1_hi = src.select(4, 1).select(2, 1)
            nisa.tensor_tensor(dst=dst.select(4, 0).select(2, 1), data1=d1_lo, data2=d1_hi, op=nl.minimum)
            nisa.tensor_tensor(dst=dst.select(4, 1).select(2, 1), data1=d1_lo, data2=d1_hi, op=nl.maximum)

    def _merge_chunks(cur, P, L, K2, log2K2):
        """Bitonic merge of every K2-chunk of ``cur`` (P rows of length L), directions alternating."""
        for jb in range(log2K2 - 1, -1, -1):
            nxt = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
            _stage_desc(cur, nxt, P, L, K2, 1 << jb)
            cur = nxt
        return cur

    def _block_topk(cur, P, B, K2, log2K2):
        """Top K2 (descending) of each of the P rows of length B -> ``[P, K2]`` tile."""
        # Phases 1..log2(K2) of a bitonic sort: every K2-chunk sorted, directions alternating.
        for kb in range(1, log2K2 + 1):
            for jb in range(kb - 1, -1, -1):
                nxt = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
                _stage_desc(cur, nxt, P, B, 1 << kb, 1 << jb)
                cur = nxt
        # Reduction rounds: max of neighbouring (descending, ascending) chunks keeps the pair's
        # top K2 as a bitonic run; a bitonic merge re-sorts it. The row halves every round.
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
        """One hierarchy level over ``src[src_off : src_off + n]``: returns the last group's tile.

        The top K2 of block ``b`` is stored at ``dst[dst_off + b*K2 ...]`` unless the level is a
        single block, whose ``[1, K2]`` result is only returned.
        """
        nb = (n + B - 1) // B
        last = None
        for g in range((nb + PMAX - 1) // PMAX):
            b0 = g * PMAX
            p = min(PMAX, nb - b0)
            full = min(p, (n - b0 * B) // B)          # blocks of this group that are complete
            tile = nl.ndarray((p, B), dtype=nl.float32, buffer=nl.sbuf)
            if full < p:
                nisa.memset(dst=tile, value=PAD_VALUE)
            if full > 0:
                nisa.dma_copy(dst=tile[0:full, 0:B],
                              src=src.ap(pattern=[[B, full], [1, B]], offset=src_off + b0 * B))
            if full < p:
                r = n - (b0 + full) * B               # elements of the partial block
                if r > 0:
                    nisa.dma_copy(dst=tile[full:full + 1, 0:r],
                                  src=src.ap(pattern=[[r, 1], [1, r]], offset=src_off + (b0 + full) * B))
            last = _block_topk(tile, p, B, K2, log2K2)
            if nb > 1:
                nisa.dma_copy(dst=dst.ap(pattern=[[K2, p], [1, K2]], offset=dst_off + b0 * K2), src=last)
        return last

    @nki.jit
    def topk_kernel(x, N, k, K2, log2K2, B):
        """The k largest of the flat ``(N,)`` fp32 tensor ``x``, descending -> ``(k,)``."""
        out = nl.ndarray((k,), dtype=nl.float32, buffer=nl.shared_hbm)
        num_programs = nl.num_programs()
        pid = nl.program_id(0)

        # Level 0: this program's contiguous share of the B-element blocks.
        nb0 = (N + B - 1) // B
        per_core = (nb0 + num_programs - 1) // num_programs
        b_lo = min(nb0, pid * per_core)
        b_hi = min(nb0, b_lo + per_core)
        n = max(0, min(N, b_hi * B) - b_lo * B)

        if n == 0:
            mine = nl.ndarray((1, K2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=mine, value=PAD_VALUE)
        else:
            # Survivors ping-pong between two HBM scratch tensors, one private range per program.
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
            # Swap the per-core runs and keep the top K2 of their union: both runs descend, so
            # max(mine, reversed(other)) is the union's top K2 as a bitonic run.
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


# Same block sizes as impl_triton.py (BLOCK_SIZE default 2048; search 1024/2048/4096).
_DEFAULT_CONFIG = SimpleNamespace(block_size=2048)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (1024, 2048, 4096)]
_kernel = topk_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def _block(block_size: int, K2: int) -> int:
    """Triton's rule: a block must hold at least two K2 runs."""
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
