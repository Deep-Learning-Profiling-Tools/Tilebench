import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"num_warps": 4}
_BLOCK_SIZE = 1024
_RADIX_BITS = 2
_RADIX = 1 << _RADIX_BITS
_FIELD_BITS = 16
_FIELD_MASK = (1 << _FIELD_BITS) - 1


@triton.jit
def radix_histogram_kernel(input, hist, N, K, shift,
                           BLOCK_SIZE: tl.constexpr, RADIX: tl.constexpr,
                           FIELD_BITS: tl.constexpr, FIELD_MASK: tl.constexpr):
    input = input.to(tl.pointer_type(tl.uint32))
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    block = tl.load(input + offset, mask=mask, other=0)
    digit = ((block >> shift) & (RADIX - 1)).to(tl.int32)


    packed = tl.where(mask, 1, 0).to(tl.int64) << (digit * FIELD_BITS)
    total = tl.sum(packed)


    counts = ((total >> (tl.arange(0, RADIX) * FIELD_BITS)) & FIELD_MASK).to(tl.int32)
    tl.store(hist + tl.arange(0, RADIX) * K + pid, counts)


@triton.jit
def radix_sum_chunks_kernel(src, dst, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    vals = tl.load(src + offset, mask=mask, other=0)
    tl.store(dst + pid, tl.sum(vals))


@triton.jit
def radix_scan_chunk_sums_kernel(sums, L, BLOCK_BB: tl.constexpr):

    offset = tl.arange(0, BLOCK_BB)
    mask = offset < L

    vals = tl.load(sums + offset, mask=mask, other=0)
    tl.store(sums + offset, tl.cumsum(vals) - vals, mask=mask)


@triton.jit
def radix_scan_chunks_kernel(src, chunk_offsets, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    vals = tl.load(src + offset, mask=mask, other=0)
    base = tl.load(chunk_offsets + pid)
    tl.store(src + offset, tl.cumsum(vals) - vals + base, mask=mask)


@triton.jit
def radix_scatter_kernel(input, output, hist, N, K, shift,
                         BLOCK_SIZE: tl.constexpr, RADIX: tl.constexpr,
                         FIELD_BITS: tl.constexpr, FIELD_MASK: tl.constexpr):
    input = input.to(tl.pointer_type(tl.uint32))
    output = output.to(tl.pointer_type(tl.uint32))
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    block = tl.load(input + offset, mask=mask, other=0)
    digit = ((block >> shift) & (RADIX - 1)).to(tl.int32)


    packed = tl.where(mask, 1, 0).to(tl.int64) << (digit * FIELD_BITS)
    excl = tl.cumsum(packed) - packed
    rank = ((excl >> (digit * FIELD_BITS)) & FIELD_MASK).to(tl.int32)

    base = tl.load(hist + digit * K + pid, mask=mask, other=0)
    tl.store(output + base + rank, block, mask=mask)


_radix_scatter_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw)
        for nw in [2, 4, 8]
    ],
    key=["N"],
    warmup=1,
    rep=3,
)(radix_scatter_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    if N <= 1:
        return input.clone()

    work = input.clone()
    output = torch.empty_like(input)

    K = triton.cdiv(N, _BLOCK_SIZE)
    M = _RADIX * K
    G2 = triton.cdiv(M, _BLOCK_SIZE)
    BB = max(1024, triton.next_power_of_2(G2))

    hist = torch.empty((M,), dtype=torch.int32, device=input.device)
    chunk_sums = torch.empty((G2,), dtype=torch.int32, device=input.device)

    cfg = _DEFAULT_CONFIG

    for shift in range(0, 32, _RADIX_BITS):
        radix_histogram_kernel[(K,)](work, hist, N, K, shift,
                                     _BLOCK_SIZE, _RADIX, _FIELD_BITS, _FIELD_MASK)
        radix_sum_chunks_kernel[(G2,)](hist, chunk_sums, M, _BLOCK_SIZE)
        radix_scan_chunk_sums_kernel[(1,)](chunk_sums, G2, BB)
        radix_scan_chunks_kernel[(G2,)](hist, chunk_sums, M, _BLOCK_SIZE)

        if autotune:
            _radix_scatter_kernel_autotuned[(K,)](
                work, output, hist, N, K, shift,
                _BLOCK_SIZE, _RADIX, _FIELD_BITS, _FIELD_MASK,
            )
        else:
            radix_scatter_kernel[(K,)](
                work, output, hist, N, K, shift,
                _BLOCK_SIZE, _RADIX, _FIELD_BITS, _FIELD_MASK,
                num_warps=cfg["num_warps"],
            )


        work, output = output, work

    return work


def get_last_config() -> dict | None:
    cfg = getattr(_radix_scatter_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps}
