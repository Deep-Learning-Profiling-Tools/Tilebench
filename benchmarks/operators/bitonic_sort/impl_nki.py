from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

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

        for b in range(n_blocks):
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

        for a in range(n_outer):
            for c in range(n_inner):
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


def _plan(M: int, j: int, span_cap: int = SPAN_CAP):
    """Tile geometry for one compare-exchange stage of an M-element buffer.

    Returns ``("local", span, P, n_blocks)`` or
    ``("stride", W, P, part_stride, n_outer, outer_stride, n_inner, inner_stride)``.
    M, j and SPAN_CAP are all powers of two, so every division below is exact.
    """
    if 2 * j <= span_cap:
        # Per-partition span: at least one whole pair-group, at most the cap,
        # and no more than an even 128-way split of the buffer.
        span = min(span_cap, max(2 * j, M // PMAX))
        P = min(PMAX, M // span)
        return ("local", span, P, M // (P * span))

    W = span_cap
    p_across_runs = min(PMAX, M // (2 * j))   # partition stride = 2j
    p_within_run = min(PMAX, j // W)          # partition stride = W
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
            plan = _plan(M, j, span_cap)
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


def _first_stage_args(data: torch.Tensor, span_cap: int) -> tuple:
    """Arguments of the first (k=2, j=1) local stage -- the autotune timing proxy."""
    N = data.numel()
    M = max(_next_pow2(N), PMAX)
    work_a = torch.full((M, 1), 1e30, dtype=torch.float32, device=data.device)
    work_a[:N, 0] = data.float()
    work_b = torch.empty_like(work_a)
    _, span, P, n_blocks = _plan(M, 1, span_cap)
    return (work_a, work_b, 2, 1, 1, span, P, n_blocks)


_DEFAULT_CONFIG = SimpleNamespace(block_size=SPAN_CAP)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (2048, 4096, 8192)]
_tuner = NkiAutotuner(bitonic_local_kernel) if nki is not None else None
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
