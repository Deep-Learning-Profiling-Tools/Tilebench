import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
    SPAN_CAP = 4096
except ImportError:
    nki = None



if nki is not None:
    @nki.jit
    def radix_local_kernel(src, dst, M, k, log2k, j, span, P, n_blocks):
        two_j = 2 * j
        C = span // two_j
        pat = [[span, P], [1, span]]

        for b in nl.affine_range(n_blocks):
            base = b * P * span

            tile_h = nl.ndarray((P, C, two_j), dtype=nl.int32, buffer=nl.sbuf)
            tile_l = nl.ndarray((P, C, two_j), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile_h.ap(pattern=pat), src=src.ap(pattern=pat, offset=base))
            nisa.dma_copy(dst=tile_l.ap(pattern=pat), src=src.ap(pattern=pat, offset=M + base))
            lo_h = tile_h[:, :, 0:j]
            up_h = tile_h[:, :, j:two_j]
            lo_l = tile_l[:, :, 0:j]
            up_l = tile_l[:, :, j:two_j]

            lo_hf = nl.add(lo_h, 0.0, dtype=nl.float32)
            up_hf = nl.add(up_h, 0.0, dtype=nl.float32)
            lo_lf = nl.add(lo_l, 0.0, dtype=nl.float32)
            up_lf = nl.add(up_l, 0.0, dtype=nl.float32)
            in_order = nl.where(nl.equal(lo_hf, up_hf),
                                nl.less_equal(lo_lf, up_lf),
                                nl.less(lo_hf, up_hf))

            idx = nl.ndarray((P, C, j), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=idx, pattern=[[two_j, C], [1, j]], offset=base,
                      channel_multiplier=span)
            kbit = nl.add(nl.right_shift(nl.bitwise_and(idx, k), log2k), 0.0,
                          dtype=nl.float32)
            descending = nl.greater(kbit, 0.5)

            keep_lower = nl.logical_xor(in_order, descending)

            out_h = nl.ndarray((P, C, two_j), dtype=nl.int32, buffer=nl.sbuf)
            out_l = nl.ndarray((P, C, two_j), dtype=nl.int32, buffer=nl.sbuf)
            out_h[:, :, 0:j] = nl.where(keep_lower, lo_h, up_h)
            out_h[:, :, j:two_j] = nl.where(keep_lower, up_h, lo_h)
            out_l[:, :, 0:j] = nl.where(keep_lower, lo_l, up_l)
            out_l[:, :, j:two_j] = nl.where(keep_lower, up_l, lo_l)

            nisa.dma_copy(dst=dst.ap(pattern=pat, offset=base), src=out_h.ap(pattern=pat))
            nisa.dma_copy(dst=dst.ap(pattern=pat, offset=M + base), src=out_l.ap(pattern=pat))

        return dst

    @nki.jit
    def radix_stride_kernel(src, dst, M, k, log2k, j, W, P, part_stride,
                            n_outer, outer_stride, n_inner, inner_stride):
        pat = [[part_stride, P], [1, W]]

        for a in nl.affine_range(n_outer):
            for c in nl.affine_range(n_inner):
                base = a * outer_stride + c * inner_stride

                lo_h = nl.ndarray((P, W), dtype=nl.int32, buffer=nl.sbuf)
                lo_l = nl.ndarray((P, W), dtype=nl.int32, buffer=nl.sbuf)
                up_h = nl.ndarray((P, W), dtype=nl.int32, buffer=nl.sbuf)
                up_l = nl.ndarray((P, W), dtype=nl.int32, buffer=nl.sbuf)
                nisa.dma_copy(dst=lo_h, src=src.ap(pattern=pat, offset=base))
                nisa.dma_copy(dst=lo_l, src=src.ap(pattern=pat, offset=M + base))
                nisa.dma_copy(dst=up_h, src=src.ap(pattern=pat, offset=base + j))
                nisa.dma_copy(dst=up_l, src=src.ap(pattern=pat, offset=M + base + j))

                lo_hf = nl.add(lo_h, 0.0, dtype=nl.float32)
                up_hf = nl.add(up_h, 0.0, dtype=nl.float32)
                lo_lf = nl.add(lo_l, 0.0, dtype=nl.float32)
                up_lf = nl.add(up_l, 0.0, dtype=nl.float32)
                in_order = nl.where(nl.equal(lo_hf, up_hf),
                                    nl.less_equal(lo_lf, up_lf),
                                    nl.less(lo_hf, up_hf))

                idx = nl.ndarray((P, W), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(dst=idx, pattern=[[1, W]], offset=base,
                          channel_multiplier=part_stride)
                kbit = nl.add(nl.right_shift(nl.bitwise_and(idx, k), log2k), 0.0,
                              dtype=nl.float32)
                descending = nl.greater(kbit, 0.5)

                keep_lower = nl.logical_xor(in_order, descending)

                new_lo_h = nl.where(keep_lower, lo_h, up_h)
                new_up_h = nl.where(keep_lower, up_h, lo_h)
                new_lo_l = nl.where(keep_lower, lo_l, up_l)
                new_up_l = nl.where(keep_lower, up_l, lo_l)

                nisa.dma_copy(dst=dst.ap(pattern=pat, offset=base), src=new_lo_h)
                nisa.dma_copy(dst=dst.ap(pattern=pat, offset=M + base), src=new_lo_l)
                nisa.dma_copy(dst=dst.ap(pattern=pat, offset=base + j), src=new_up_h)
                nisa.dma_copy(dst=dst.ap(pattern=pat, offset=M + base + j), src=new_up_l)

        return dst


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _plan(M: int, j: int):
    if 2 * j <= SPAN_CAP:
        span = min(SPAN_CAP, max(2 * j, M // PMAX))
        P = min(PMAX, M // span)
        return ("local", span, P, M // (P * span))

    W = SPAN_CAP
    p_across_runs = min(PMAX, M // (2 * j))  
    p_within_run = min(PMAX, j // W)          
    if p_across_runs >= p_within_run:
        P = p_across_runs
        return ("stride", W, P, 2 * j, M // (2 * j * P), P * 2 * j, j // W, W)
    P = p_within_run
    return ("stride", W, P, W, M // (2 * j), 2 * j, j // (W * P), P * W)


def _radix_sort_1d(data: torch.Tensor) -> torch.Tensor:
    N = data.numel()
    if N <= 1:
        return data.clone()

    M = max(_next_pow2(N), PMAX)
    data_i64 = data.to(torch.int64)
    high = (data_i64 >> 16) & 0xFFFF
    low = data_i64 & 0xFFFF

    PAD_HIGH, PAD_LOW = 0xFFFF, 0xFFFF  
    
    work_a = torch.empty((2 * M, 1), dtype=torch.int32, device=data.device)
    work_a[:M, 0] = PAD_HIGH
    work_a[M:, 0] = PAD_LOW
    work_a[:N, 0] = high.to(torch.int32)
    work_a[M:M + N, 0] = low.to(torch.int32)
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
                dst = radix_local_kernel(src, dst, M, k, log2k, j, span, P, n_blocks)
            else:
                _, W, P, part_stride, n_outer, outer_stride, n_inner, inner_stride = plan
                dst = radix_stride_kernel(src, dst, M, k, log2k, j, W, P, part_stride,
                                          n_outer, outer_stride, n_inner, inner_stride)
            src, dst = dst, src
            j //= 2
        k *= 2

    sorted_high = src[:N, 0].to(torch.int64)
    sorted_low = src[M:M + N, 0].to(torch.int64)
    result = (sorted_high << 16) | sorted_low
    return result.to(data.dtype)


def run(input: torch.Tensor, N: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    return _radix_sort_1d(input)


def get_last_config() -> dict | None:
    return None
