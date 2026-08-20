import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128

# Largest block a single hierarchy level may sort. One block lives entirely in
# one SBUF partition (4096 fp32 = 16KB of the 192KB partition, and the kernel
# keeps two such tiles live for the compare-exchange ping-pong).
MAX_BLOCK = 4096

# Smallest block worth launching for: below this the per-level launch overhead
# dominates, and a block this small already shrinks its input by >= 16x.
MIN_BLOCK = 256

# Sentinel used to pad a partial block up to B elements. A large *finite*
# negative value, never -inf: literal infinities were confirmed to corrupt the
# bitonic compare-exchange network on this hardware/SDK (see the PAD_VALUE note
# in benchmarks/operators/bitonic_sort/impl_nki.py). The benchmark generator
# draws standard-normal values, so -1e30 is safely below anything real.
PAD_VALUE = -1e30

_last_config: dict = {}


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _log2(n: int) -> int:
    return n.bit_length() - 1


if nki is not None:

    def _stage_desc(cur, nxt, P, B, k, j):
        """One bitonic compare-exchange stage, descending, over P independent rows.

        Row-local element index ``i`` pairs with ``i + j``; the bitonic direction
        bit is bit ``log2(k)`` of ``i``, so directions come in runs of
        ``G = k / 2j`` consecutive pair-groups that alternate every ``k``
        elements.  Viewing a row as ``(R, 2, G, 2, j)`` puts that direction bit
        on its own axis, which lets each direction be handled by a plain min/max
        pair over a strided view -- no index tile and no predicated select.

        The order is *descending*, i.e. the mirror image of the textbook
        ascending network: direction bit 0 means "this pair ends up descending".
        """
        G = k // (2 * j)   # pair-groups sharing one direction
        R = B // (2 * k)   # direction-run pairs; 0 only on the final k == B pass
        if R == 0:
            # k == B: bit log2(k) of every in-row index is 0, so the whole row
            # is a single descending run.
            src = cur.reshape((P, G, 2, j))
            dst = nxt.reshape((P, G, 2, j))
            src_lo = src.select(2, 0)
            src_hi = src.select(2, 1)
            nisa.tensor_tensor(dst=dst.select(2, 0), data1=src_lo, data2=src_hi,
                               op=nl.maximum)
            nisa.tensor_tensor(dst=dst.select(2, 1), data1=src_lo, data2=src_hi,
                               op=nl.minimum)
        else:
            src = cur.reshape((P, R, 2, G, 2, j))
            dst = nxt.reshape((P, R, 2, G, 2, j))
            # Direction bit 0: pair ends descending (larger element first).
            d0_lo = src.select(4, 0).select(2, 0)
            d0_hi = src.select(4, 1).select(2, 0)
            nisa.tensor_tensor(dst=dst.select(4, 0).select(2, 0),
                               data1=d0_lo, data2=d0_hi, op=nl.maximum)
            nisa.tensor_tensor(dst=dst.select(4, 1).select(2, 0),
                               data1=d0_lo, data2=d0_hi, op=nl.minimum)
            # Direction bit 1: pair ends ascending.
            d1_lo = src.select(4, 0).select(2, 1)
            d1_hi = src.select(4, 1).select(2, 1)
            nisa.tensor_tensor(dst=dst.select(4, 0).select(2, 1),
                               data1=d1_lo, data2=d1_hi, op=nl.minimum)
            nisa.tensor_tensor(dst=dst.select(4, 1).select(2, 1),
                               data1=d1_lo, data2=d1_hi, op=nl.maximum)

    @nki.jit
    def block_topk_kernel(src, B, log2B, K2, P, n_groups, nb):
        """One hierarchy level: descending-sort each B-element block, keep top K2.

        ``src`` is ``(nb * B, 1)`` in HBM, block ``b`` being the contiguous span
        ``[b*B, (b+1)*B)``.  Block ``b`` is loaded into partition ``b % P`` of
        group ``b // P``; every compare-exchange partner of a block-local bitonic
        network lies inside the same block, so the whole network is
        partition-local and all P blocks of a group sort simultaneously (SIMD
        across partitions replaces the grid-over-blocks parallelism of the GPU
        implementations).

        Returns ``(nb * K2, 1)`` in HBM: the top K2 elements of every block,
        descending.
        """
        out = nl.ndarray((nb * K2, 1), dtype=nl.float32, buffer=nl.shared_hbm)

        for g in range(n_groups):
            p = min(P, nb - g * P)

            buf_a = nl.ndarray((p, B), dtype=nl.float32, buffer=nl.sbuf)
            buf_b = nl.ndarray((p, B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=buf_a,
                          src=src.ap(pattern=[[B, p], [1, B]], offset=g * P * B))

            cur = buf_a
            nxt = buf_b
            for kb in range(1, log2B + 1):
                k = 1 << kb
                for jb in range(kb - 1, -1, -1):
                    _stage_desc(cur, nxt, p, B, k, 1 << jb)
                    tmp = cur
                    cur = nxt
                    nxt = tmp

            nisa.dma_copy(dst=out.ap(pattern=[[K2, p], [1, K2]],
                                     offset=g * P * K2),
                          src=cur[:, 0:K2])

        return out


def _level_block(n: int, K2: int) -> int:
    """Block size (a power of two) for one hierarchy level of ``n`` elements.

    A block is sorted inside one partition, so ``B <= MAX_BLOCK``.  ``B`` must be
    at least ``2*K2`` for the level to make progress; ``4*K2`` (shrink by >= 4x
    per level) bounds the number of levels, and hence kernel launches.  Within
    those bounds the smallest block that still fills the 128 partitions is
    preferred, since a bitonic sort costs ``O(B log^2 B)`` per block.
    """
    if 2 * K2 > MAX_BLOCK:
        raise ValueError(
            f"top-k with k'={K2} needs a block of at least {2 * K2} elements, "
            f"but a block is capped at {MAX_BLOCK}."
        )
    lo = min(max(MIN_BLOCK, 4 * K2), MAX_BLOCK)
    if n <= lo:
        # Everything fits in one exactly-sized block: nothing has to be padded.
        return max(_next_pow2(n), K2, 2)
    return min(MAX_BLOCK, max(lo, _next_pow2((n + PMAX - 1) // PMAX)))


def _hierarchical_topk(data: torch.Tensor, k: int) -> torch.Tensor:
    """Repeated block-local top-k until a single block holds the answer."""
    K2 = _next_pow2(k)
    cur = data.reshape(-1, 1)
    n = cur.shape[0]
    levels = 0
    first_block = None

    while True:
        B = _level_block(n, K2)
        nb = (n + B - 1) // B
        if nb * B != n:
            padded = torch.full((nb * B, 1), PAD_VALUE, dtype=torch.float32,
                                device=cur.device)
            padded[:n] = cur
            cur = padded
        P = min(PMAX, nb)
        n_groups = (nb + P - 1) // P

        cur = block_topk_kernel(cur, B, _log2(B), K2, P, n_groups, nb)
        levels += 1
        if first_block is None:
            first_block = B

        if nb == 1:
            _last_config.clear()
            _last_config.update({"block": int(first_block), "K2": int(K2),
                                 "levels": int(levels)})
            return cur[:k, 0]
        n = nb * K2


def run(input: torch.Tensor, N: int, k: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    """The k largest values of a 1-D tensor, descending (matches torch.topk).

    ``block_size`` is accepted for API compatibility only: the block size of
    every hierarchy level is derived from that level's own length and from k,
    both of which change as the hierarchy contracts (see ``_level_block``).
    """
    if k < 1 or k > N:
        raise ValueError(f"k={k} out of range for N={N}")
    return _hierarchical_topk(input.contiguous(), int(k))


def get_last_config() -> dict | None:
    return dict(_last_config) if _last_config else None
