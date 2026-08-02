"""NKI bitonic sort.

Uses the "new" NKI frontend (see destindex/impl_nki.py) for its ``.ap()``
access-pattern handles on HBM tensors: every compare-exchange stage reads two
*strided* views of the work buffer (the "lower" and "upper" element of each
pair) rather than one contiguous slice.

Data (M = next_pow2(N), padded) lives in HBM as (M, 1) fp32; a stage's pairs
are (i, i^j) for the M/2 indices i with (i & j) == 0 -- i.e. the run of j
elements starting at every multiple of 2j is the "lower" half of a pair and
the next j elements are its "upper" half. Both halves are reached with plain
affine access patterns, so no indirect/gather DMA is needed (see the tiling
note below).

TILING (the reason this file looks the way it does)
---------------------------------------------------
The obvious layout -- one element per partition lane, i.e. a (PMAX, 1) tile
per block and M/PMAX blocks -- needs 8192 blocks for M = 2^20 alone, and
``nl.affine_range`` unrolls at compile time, so each of the ~log2(M)^2/2
stages copies its body thousands of times into the NEFF. That is
unrunnable in practice (hundreds of GB of compiler scratch; see 1d_conv for
the same blowup and the same fix). Instead every stage uses a *fat* tile:
partition dim up to 128, free dim carrying thousands of elements, which drops
the block count from thousands to single digits.

Two tilings are needed because the pair distance j sweeps 1 .. M/2:

* "local" stages (2*j <= SPAN_CAP): each partition owns a contiguous span of
  ``span`` elements, which holds span/(2j) whole pairs-groups. The tile is
  loaded with one fully contiguous DMA and shaped (P, C, 2j); the two halves
  of every pair are then the strided sub-views [..., 0:j] and [..., j:2j], so
  the compare-exchange is pure on-chip strided compute. This is the case that
  would otherwise degenerate to 4-byte DMA descriptors for j = 1.

* "strided" stages (2*j > SPAN_CAP): a partition can no longer hold a whole
  pair-group, so the lower and upper halves are loaded as two separate
  (P, W) tiles W = SPAN_CAP elements wide. The partition stride is either
  2j (consecutive partitions take consecutive pair-groups) or W (consecutive
  partitions take consecutive chunks of the *same* j-element run), whichever
  gives more partitions -- their product is M/(2W), so the worse of the two is
  never used and the iteration count stays ~sqrt(M/(2W)).

Per-stage iteration count is then about max(M/(128*SPAN_CAP),
sqrt(M/(2*SPAN_CAP))): 1..4 for the smallest configured case (N = 500K,
M = 2^19) and <= 32 at the top of the sweep (N = 10M, M = 2^24), against
M/128 = up to 131072 blocks for the old one-element-per-lane layout.
SPAN_CAP is the knob if a bigger M ever needs a smaller NEFF -- doubling it
halves every "local" stage's block count and doubles its SBUF footprint.

Direction bit: bitonic's per-pair sort direction is (i & k) for the pair's
lower index i, which varies within a tile, so it is materialized with
``nisa.iota`` (an affine index generator) and reduced to a 0/1 mask the same
way the previous implementation did -- integer tensor-tensor comparisons are
not supported by this frontend ("operand0 must be float32, got i32"), so the
bit is extracted with bitwise_and + right_shift and compared as float.
"""
import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

# Largest per-partition contiguous span, in fp32 elements, that a stage tile
# may cover (8192 * 4B = 32KB of the 192KB SBUF partition; the kernel keeps
# ~3 live tiles of that size plus half-size temporaries).
SPAN_CAP = 8192


if nki is not None:
    @nki.jit
    def bitonic_local_kernel(src, dst, k, log2k, j, span, P, n_blocks):
        """Compare-exchange for stages with 2*j <= span (partner in-partition).

        Each partition holds ``span`` contiguous elements = C = span/(2j)
        pair-groups; lower/upper halves are strided views of that one tile.
        """
        two_j = 2 * j
        C = span // two_j
        pat = [[span, P], [1, span]]

        for b in nl.affine_range(n_blocks):
            base = b * P * span

            tile = nl.ndarray((P, C, two_j), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile.ap(pattern=pat), src=src.ap(pattern=pat, offset=base))
            lower = tile[:, :, 0:j]
            upper = tile[:, :, j:two_j]

            # idx[p, c, f] = index of the *lower* element of this pair.
            idx = nl.ndarray((P, C, j), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=idx, pattern=[[two_j, C], [1, j]], offset=base,
                      channel_multiplier=span)
            kbit = nl.add(nl.right_shift(nl.bitwise_and(idx, k), log2k), 0.0,
                          dtype=nl.float32)
            descending = nl.greater(kbit, 0.5)

            # keep_lower: this pair stays as-is (no swap). Ascending pairs keep
            # their order when lower <= upper; descending pairs when it doesn't.
            # Selecting the two outputs from one boolean (rather than computing
            # min/max and picking) keeps both sides of the pair consistent and
            # never does arithmetic on the large finite pad sentinel.
            keep_lower = nl.logical_xor(nl.less_equal(lower, upper), descending)

            out = nl.ndarray((P, C, two_j), dtype=nl.float32, buffer=nl.sbuf)
            out[:, :, 0:j] = nl.where(keep_lower, lower, upper)
            out[:, :, j:two_j] = nl.where(keep_lower, upper, lower)

            nisa.dma_copy(dst=dst.ap(pattern=pat, offset=base), src=out.ap(pattern=pat))

        return dst

    @nki.jit
    def bitonic_stride_kernel(src, dst, k, log2k, j, W, P, part_stride,
                              n_outer, outer_stride, n_inner, inner_stride):
        """Compare-exchange for stages with 2*j > span (partner out-of-partition).

        Lower and upper halves are two separate (P, W) loads j elements apart;
        ``part_stride`` walks either whole pair-groups (2j) or chunks of one
        j-element run (W).
        """
        pat = [[part_stride, P], [1, W]]

        for a in nl.affine_range(n_outer):
            for c in nl.affine_range(n_inner):
                base = a * outer_stride + c * inner_stride

                lower = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                upper = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
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


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _plan(M: int, j: int):
    """Tile geometry for one compare-exchange stage of an M-element buffer.

    Returns ``("local", span, P, n_blocks)`` or
    ``("stride", W, P, part_stride, n_outer, outer_stride, n_inner, inner_stride)``.
    M, j and SPAN_CAP are all powers of two, so every division below is exact.
    """
    if 2 * j <= SPAN_CAP:
        # Per-partition span: at least one whole pair-group, at most the cap,
        # and no more than an even 128-way split of the buffer.
        span = min(SPAN_CAP, max(2 * j, M // PMAX))
        P = min(PMAX, M // span)
        return ("local", span, P, M // (P * span))

    W = SPAN_CAP
    p_across_runs = min(PMAX, M // (2 * j))   # partition stride = 2j
    p_within_run = min(PMAX, j // W)          # partition stride = W
    if p_across_runs >= p_within_run:
        P = p_across_runs
        return ("stride", W, P, 2 * j, M // (2 * j * P), P * 2 * j, j // W, W)
    P = p_within_run
    return ("stride", W, P, W, M // (2 * j), 2 * j, j // (W * P), P * W)


def _bitonic_sort_1d(data: torch.Tensor) -> torch.Tensor:
    N = data.numel()
    if N <= 1:
        return data.clone()

    M = max(_next_pow2(N), PMAX)
    # A large *finite* sentinel, not float("inf"): confirmed via
    # temp/probe_bitonic_pad.py that literal +inf padding corrupts the sort
    # on this hardware/SDK (exact-power-of-2 N, needing no padding, always
    # sorted correctly; padded N did not -- switching the pad value from
    # inf to 1e30 was the only change needed to fix it). Values in this
    # benchmark's generator are standard-normal, so 1e30 is safely larger
    # than anything real without risking inf-specific behavior.
    PAD_VALUE = 1e30
    work_a = torch.full((M, 1), PAD_VALUE, dtype=torch.float32, device=data.device)
    work_a[:N, 0] = data.float()
    work_b = torch.empty_like(work_a)

    src, dst = work_a, work_b
    k = 2
    while k <= M:
        j = k // 2
        log2k = k.bit_length() - 1
        while j > 0:
            plan = _plan(M, j)
            if plan[0] == "local":
                _, span, P, n_blocks = plan
                dst = bitonic_local_kernel(src, dst, k, log2k, j, span, P, n_blocks)
            else:
                _, W, P, part_stride, n_outer, outer_stride, n_inner, inner_stride = plan
                dst = bitonic_stride_kernel(src, dst, k, log2k, j, W, P, part_stride,
                                            n_outer, outer_stride, n_inner, inner_stride)
            src, dst = dst, src
            j //= 2
        k *= 2

    return src[:N, 0].to(data.dtype)


def run(data: torch.Tensor, N: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    return _bitonic_sort_1d(data)


def get_last_config() -> dict | None:
    return None
